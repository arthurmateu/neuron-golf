"""Run the GPU jobs on Modal (80 GB cards instead of the 12 GB one).

    modal run modal_census.py::download [--model Qwen/Qwen3.5-9B]   # weights -> Volume, once, CPU only
    modal run modal_census.py::bench --gpu H100 --batches 256,512,1024
    modal run modal_census.py::census --layers 0-31 --gpu H100 --containers 8 --batch 1024
    modal run modal_census.py::screen --gpu H100                     # screen.py --probes census, both signs
    modal run modal_census.py::run --cmd "census.py verify miku --top 16"   # any repo script
    modal volume get neuron-golf-out census_gen/4b data/census_gen/  # results -> local

`run` copies the repo into the container, runs the command there, and writes back every file it
created or changed under data/ and results/. data/screen and data/census_gen live on the
neuron-golf-out Volume (symlinked into the container's copy). Weights live on neuron-golf-hf.
Everything imports golf, so the steering recipe has one source of truth.
"""

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("neuron-golf")
hf_vol = modal.Volume.from_name("neuron-golf-hf", create_if_missing=True)
out_vol = modal.Volume.from_name("neuron-golf-out", create_if_missing=True)
VOLS = {"/hf": hf_vol, "/out": out_vol}
REPO = "/root/ng"
MODEL = "Qwen/Qwen3.5-4B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "libc6-dev")  # triton compiles a small C helper at first use
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "flash-linear-attention==0.6.0", "fla-core==0.6.0",
                    "accelerate==1.15.0", "numpy==2.5.4", "safetensors==0.8.0", "huggingface-hub==2.3.0")
    .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "TRITON_CACHE_DIR": "/out/triton"})
    .add_local_dir(Path(__file__).parent, REPO,
                   ignore=[".venv", ".git", "build", "data/screen", "data/census_gen", "**/__pycache__", "*.gguf"])
)


def workdir(model_id: str) -> Path:
    """A writable copy of the repo for model_id, with the big outputs on the Volume."""
    os.environ["GOLF_MODEL"] = model_id
    tag = model_id.rsplit("-", 1)[-1].lower()
    work = Path("/work")
    if not work.exists():
        shutil.copytree(REPO, work)
        for name, target in (("screen", f"/out/screen/{tag}"), ("census_gen", "/out/census_gen")):
            Path(target).mkdir(parents=True, exist_ok=True)
            (work / "data" / name).symlink_to(target)
    sys.path.insert(0, str(work))
    os.chdir(work)
    return work


def snapshot(work: Path) -> dict:
    """mtime of every real file under data/ and results/ (symlinked Volume dirs skipped)."""
    out = {}
    for sub in ("data", "results"):
        for root, _, files in os.walk(work / sub):  # does not follow symlinks
            for f in files:
                p = Path(root) / f
                out[str(p.relative_to(work))] = p.stat().st_mtime
    return out


@app.function(image=image, volumes=VOLS, timeout=3600, cpu=4)
def download_weights(model_id: str):
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id)
    hf_vol.commit()
    return sorted(os.listdir(path))


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=3600)
def bench_fn(model_id: str, batches: list[int], sample_layers: list[int], tokens: int = 40, fast: bool = True) -> dict:
    """Seconds per batch for each batch size, and how many rows each dose rung retries."""
    workdir(model_id)
    import torch

    import gen_census
    import golf

    t0 = time.time()
    model, tok = golf.load()
    norms = golf.get_norms(model, tok)
    colnorm = golf.down_col_norms(model)
    res = {"gpu": torch.cuda.get_device_name(), "load_s": round(time.time() - t0, 1)}
    d_mlp = colnorm.shape[1]
    check = [([(10, n, 1.25)], gen_census.PROMPT) for n in range(64)]
    plain = golf.generate_rows(model, tok, check, norms, max_new_tokens=tokens, batch=64, colnorm=colnorm)
    gen_census.swap_recurrent_state()
    swapped = golf.generate_rows(model, tok, check, norms, max_new_tokens=tokens, batch=64, colnorm=colnorm)
    quick = golf.generate_rows(model, tok, check, norms, max_new_tokens=tokens, batch=64, colnorm=colnorm, fast=True)
    res["swap_identical"] = sum(a == b for a, b in zip(plain, swapped))
    res["greedy_identical"] = sum(a == b for a, b in zip(plain, quick))
    res["greedy_diffs"] = [(a[:70], b[:70]) for a, b in zip(plain, quick) if a != b][:4]
    print({k: res[k] for k in ("swap_identical", "greedy_identical", "greedy_diffs")}, flush=True)
    for b in batches:
        rows = [([(10, n, 1.25)], gen_census.PROMPT) for n in range(b)]
        golf.generate_rows(model, tok, rows, norms, max_new_tokens=4, batch=b, colnorm=colnorm, fast=fast)  # warm-up
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        golf.generate_rows(model, tok, rows, norms, max_new_tokens=tokens, batch=b, colnorm=colnorm, fast=fast)
        torch.cuda.synchronize()
        dt = time.time() - t
        res[f"batch{b}"] = {"s": round(dt, 2), "rows_per_s": round(b / dt, 1),
                            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        print(res[f"batch{b}"], flush=True)
    best = max(batches)
    sample = range(0, d_mlp, d_mlp // 128)
    for L in sample_layers:
        t = time.time()
        recs, rungs = gen_census.census_layer(model, tok, norms, colnorm, L, best, tokens, neurons=sample, fast=fast)
        res[f"L{L}"] = {"rungs": rungs, "s": round(time.time() - t, 1),
                        "degenerate": sum(r["degenerate"] for r in recs), "rows": len(recs),
                        "examples": [(r["neuron"], r["sign"], r["dose"], r["reply"][:80]) for r in recs[:6]]}
        print(L, res[f"L{L}"], flush=True)
    res["total_s"] = round(time.time() - t0, 1)
    return res


@app.cls(image=image, gpu="H100", volumes=VOLS, timeout=3600, scaledown_window=30)
class Census:
    model_id: str = modal.parameter(default=MODEL)
    batch: int = modal.parameter(default=512)
    tokens: int = modal.parameter(default=40)

    @modal.enter()
    def load(self):
        workdir(self.model_id)
        import gen_census
        import golf

        self.golf, self.gc = golf, gen_census
        gen_census.swap_recurrent_state()
        self.model, self.tok = golf.load()
        self.norms = golf.get_norms(self.model, self.tok)
        self.colnorm = golf.down_col_norms(self.model)
        self.out = Path("/out/census_gen") / golf.MODEL_TAG
        self.out.mkdir(parents=True, exist_ok=True)
        if not (self.out / "baseline.json").exists():
            gen_census.baseline(self.model, self.tok, self.out, self.tokens)
            out_vol.commit()

    @modal.method()
    def layer(self, L: int) -> dict:
        path = self.out / f"L{L:02d}.jsonl.gz"
        if path.exists():
            return {"layer": L, "skipped": True}
        t = time.time()
        recs, rungs = self.gc.census_layer(self.model, self.tok, self.norms, self.colnorm, L, self.batch, self.tokens)
        self.gc.write_gz(path, recs)
        out_vol.commit()
        return {"layer": L, "s": round(time.time() - t), "rungs": rungs,
                "degenerate": sum(r["degenerate"] for r in recs)}


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=3 * 3600)
def script_fn(cmd: str, model_id: str) -> dict:
    """Run `python <cmd>` in a copy of the repo; return {path: bytes} of new/changed data/ + results/ files."""
    work = workdir(model_id)
    before = snapshot(work)
    rc = subprocess.run([sys.executable, "-u", *shlex.split(cmd)], cwd=work).returncode
    out_vol.commit()
    after = snapshot(work)
    files = {p: (work / p).read_bytes() for p, m in after.items() if before.get(p) != m}
    return {"rc": rc, "files": files}


@app.local_entrypoint()
def download(model: str = MODEL):
    print(download_weights.remote(model))


@app.local_entrypoint()
def bench(gpu: str = "H100", batches: str = "256,512,1024", layers: str = "3,12,22,30", model: str = MODEL,
          tokens: int = 40, slow: bool = False):
    res = bench_fn.with_options(gpu=gpu).remote(model, [int(b) for b in batches.split(",")],
                                                [int(x) for x in layers.split(",")], tokens, not slow)
    import json

    print(json.dumps(res, indent=1, ensure_ascii=False))


@app.local_entrypoint()
def census(layers: str = "0-31", gpu: str = "H100", containers: int = 8, batch: int = 512, model: str = MODEL,
           tokens: int = 40):
    lo, hi = map(int, layers.split("-"))
    c = Census.with_options(gpu=gpu, max_containers=containers)(model_id=model, batch=batch, tokens=tokens)
    t = time.time()
    for r in c.layer.map(range(lo, hi + 1), order_outputs=False):
        print(f"[{time.time() - t:.0f}s] {r}", flush=True)


@app.local_entrypoint()
def screen(gpu: str = "H100", jobs_per_dose: int = 4, chunk: int = 256, model: str = MODEL, doses: str = "1,-1",
           n_layers: int = 32):
    """screen.py --probes census over every layer at each dose. Layer L runs layers L..end, so the
    layer ranges are cut to equal cost, not equal length."""
    cost = [n_layers - L for L in range(n_layers)]
    cuts, acc = [0], 0
    for L, c in enumerate(cost):
        acc += c
        if acc >= sum(cost) * len(cuts) / jobs_per_dose and len(cuts) < jobs_per_dose:
            cuts.append(L + 1)
    ranges = [(a, b - 1) for a, b in zip(cuts, cuts[1:] + [n_layers]) if a < b]
    jobs = [f"screen.py --dose {d} --probes census --layers {a}-{b} --chunk {chunk}"
            for d in doses.split(",") for a, b in ranges]
    print("\n".join(jobs))
    t = time.time()
    for r in script_fn.with_options(gpu=gpu).map(jobs, kwargs={"model_id": model}, order_outputs=False):
        print(f"[{time.time() - t:.0f}s] rc {r['rc']}", flush=True)


@app.local_entrypoint()
def run(cmd: str, gpu: str = "H100", model: str = MODEL):
    r = script_fn.with_options(gpu=gpu).remote(cmd, model)
    root = Path(__file__).parent
    for p, data in r["files"].items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_bytes(data)
        print("wrote", p)
    if r["rc"]:
        sys.exit(r["rc"])
