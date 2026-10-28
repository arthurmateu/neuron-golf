"""Run the Mistral scripts (mgolf, mhunt, msteer) on Modal, with the model loaded once and kept warm.

    modal run modal_mistral.py::download                      # weights -> Volume, once, CPU only (~121 GB FP8)
    modal run modal_mistral.py::selftest                      # test_mgolf.py, CPU
    modal run modal_mistral.py::dryrun --cmd "mhunt.py sap"   # a real script on test_mgolf.tiny(), CPU
    modal deploy modal_mistral.py                             # the GPU class; scales to zero after 5 idle min
    python modal_mistral.py mhunt.py sap lebron --top 12      # run in the warm container (needs `modal` importable)
    modal app stop neuron-golf-mistral                        # stop paying right away

A call ships the local *.py files and data/mistral/ to the container, runs the script inside the
process that already holds the model (mgolf._loaded), and writes back every file the script
created or changed under data/mistral/ and results/mistral/. Mistral Small 4 is ~238 GB in bf16:
one B300 (288 GB). GOLF_GPU / GOLF_MODEL at deploy time pick others (Large 4 will need 8 GPUs).
"""

import contextlib
import io
import os
import runpy
import shlex
import sys
import time
import traceback
from pathlib import Path

import modal

APP, MODEL = "neuron-golf-mistral", os.environ.get("GOLF_MODEL", "mistralai/Mistral-Small-4-119B-2603")
GPU = os.environ.get("GOLF_GPU", "B300")
SYNC = ("data/mistral", "results/mistral")
LOCAL = Path(__file__).parent
WORK = Path("/work")

app = modal.App(APP)
hf_vol = modal.Volume.from_name("neuron-golf-mistral-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "accelerate==1.15.0", "numpy==2.5.4",
                    "safetensors==0.8.0", "huggingface-hub==2.3.0")
    .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "GOLF_MODEL": MODEL})
    .add_local_dir(LOCAL, str(WORK), ignore=[".venv", ".git", "build", "data", "results", "**/__pycache__", "*.gguf"])
)


def local_files() -> dict[str, bytes]:
    files = {p.name: p.read_bytes() for p in LOCAL.glob("*.py")}
    return files | {str(p.relative_to(LOCAL)): p.read_bytes() for p in (LOCAL / "data" / "mistral").rglob("*") if p.is_file()}


def write_back(changed: dict[str, bytes]):
    for name, data in changed.items():
        (LOCAL / name).parent.mkdir(parents=True, exist_ok=True)
        (LOCAL / name).write_bytes(data)
        print("wrote", name)


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def snapshot() -> dict[str, float]:
    return {str(p.relative_to(WORK)): p.stat().st_mtime for d in SYNC for p in (WORK / d).rglob("*") if p.is_file()}


def execute(loaded, argv: list[str], files: dict[str, bytes]) -> tuple[str, dict[str, bytes]]:
    """Run argv[0] from WORK as __main__ with mgolf.load() returning `loaded`; return stdout + changed files."""
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))
    for name, data in files.items():
        (WORK / name).parent.mkdir(parents=True, exist_ok=True)
        (WORK / name).write_bytes(data)
    before = snapshot()
    for name in [n for n, m in sys.modules.items() if (getattr(m, "__file__", None) or "").startswith(str(WORK))]:
        del sys.modules[name]  # re-import the shipped versions of mgolf, personas, ...
    import mgolf

    mgolf._loaded = loaded
    buf = io.StringIO()
    sys.argv = argv
    with contextlib.redirect_stdout(Tee(sys.__stdout__, buf)):
        t = time.time()
        try:
            runpy.run_path(str(WORK / argv[0]), run_name="__main__")
        except BaseException:  # SystemExit from argparse included
            print(traceback.format_exc())
        print(f"[{argv[0]} took {time.time() - t:.0f}s]")
    after = snapshot()
    return buf.getvalue(), {n: (WORK / n).read_bytes() for n, m in after.items() if before.get(n) != m}


@app.function(image=image, volumes={"/hf": hf_vol}, timeout=7200, cpu=8, memory=16384)
def download_weights(model_id: str):
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id, ignore_patterns=["consolidated*"])  # skip the duplicate mistral-native shards
    hf_vol.commit()
    return sorted(os.listdir(path))


@app.local_entrypoint()
def download(model: str = MODEL):
    print(download_weights.remote(model))


@app.function(image=image, timeout=600)
def selftest_fn() -> str:
    import subprocess

    r = subprocess.run([sys.executable, "test_mgolf.py"], cwd=WORK, capture_output=True, text=True)
    return r.stdout + r.stderr


@app.local_entrypoint()
def selftest():
    print(selftest_fn.remote())


@app.function(image=image, volumes={"/hf": hf_vol}, timeout=1800, cpu=4, memory=8192)
def dryrun_fn(argv: list[str], files: dict[str, bytes]) -> tuple[str, dict[str, bytes]]:
    """The real script on a random 3-layer Mistral 4 with the real tokenizer: catches bugs for cents."""
    os.environ["GOLF_MODEL"] = "tiny"
    sys.path.insert(0, str(WORK))
    from transformers import AutoTokenizer

    import test_mgolf

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    return execute((test_mgolf.tiny(len(tok)), tok), argv, files)


@app.local_entrypoint()
def dryrun(cmd: str):
    out, changed = dryrun_fn.remote(shlex.split(cmd), local_files())
    print(out)
    print("(dry run: not writing back)", sorted(changed))


@app.cls(image=image, gpu=GPU, volumes={"/hf": hf_vol}, timeout=3 * 3600, memory=32768, scaledown_window=300)
class Golf:
    @modal.enter()
    def load(self):
        sys.path.insert(0, str(WORK))
        import mgolf

        t = time.time()
        self.loaded = mgolf.load()
        print(f"loaded {mgolf.MODEL_ID} in {time.time() - t:.0f}s")

    @modal.method()
    def run(self, argv: list[str], files: dict[str, bytes]) -> tuple[str, dict[str, bytes]]:
        return execute(self.loaded, argv, files)


if __name__ == "__main__":  # local client for the deployed class
    out, changed = modal.Cls.from_name(APP, "Golf")().run.remote(sys.argv[1:], local_files())
    print(out)
    write_back(changed)
