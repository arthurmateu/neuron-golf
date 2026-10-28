"""Run the Mistral scripts (mgolf, mhunt, msteer) on Modal GPUs, with the model loaded once and kept warm.

    modal run modal_mistral.py::download        # weights -> Volume, once, CPU only (~121 GB of FP8)
    modal run modal_mistral.py::selftest        # test_mgolf.py on a CPU container
    modal deploy modal_mistral.py               # the GPU class; scales to zero after 20 idle minutes
    uv run --no-project --with modal python modal_mistral.py mhunt.py sap lebron --top 12

A call ships the local *.py files and data/mistral/ to the container, runs the script inside the
process that already holds the model (mgolf._loaded), and writes back every file the script
created or changed under data/mistral/ and results/mistral/. Mistral Small 4 is ~238 GB in bf16,
so it needs two H200s. GOLF_MODEL at deploy time picks another checkpoint (Large 4 will need 8 GPUs).
"""

import contextlib
import io
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

import modal

APP, MODEL = "neuron-golf-mistral", os.environ.get("GOLF_MODEL", "mistralai/Mistral-Small-4-119B-2603")
GPU = os.environ.get("GOLF_GPU", "H200:2")
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
    return {str(p.relative_to(WORK)): p.stat().st_mtime
            for d in SYNC for p in (WORK / d).rglob("*") if p.is_file()}


@app.cls(image=image, gpu=GPU, volumes={"/hf": hf_vol}, timeout=3 * 3600, memory=65536, scaledown_window=1200)
class Golf:
    @modal.enter()
    def load(self):
        sys.path.insert(0, str(WORK))
        os.chdir(WORK)
        import mgolf

        t = time.time()
        self.loaded = mgolf.load()
        print(f"loaded {mgolf.MODEL_ID} in {time.time() - t:.0f}s, devices {set(self.loaded[0].hf_device_map.values())}")

    @modal.method()
    def run(self, argv: list[str], files: dict[str, bytes]) -> tuple[str, dict[str, bytes]]:
        for name, data in files.items():
            (WORK / name).parent.mkdir(parents=True, exist_ok=True)
            (WORK / name).write_bytes(data)
        before = snapshot()
        for name in [n for n, m in sys.modules.items() if getattr(m, "__file__", None) and m.__file__.startswith(str(WORK))]:
            del sys.modules[name]  # re-import the shipped versions of mgolf, personas, ...
        import mgolf

        mgolf._loaded = self.loaded
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


if __name__ == "__main__":  # local client for the deployed class
    files = {p.name: p.read_bytes() for p in LOCAL.glob("*.py")}
    files |= {str(p.relative_to(LOCAL)): p.read_bytes() for p in (LOCAL / "data" / "mistral").rglob("*") if p.is_file()}
    out, changed = modal.Cls.from_name(APP, "Golf")().run.remote(sys.argv[1:], files)
    print(out)
    for name, data in changed.items():
        (LOCAL / name).parent.mkdir(parents=True, exist_ok=True)
        (LOCAL / name).write_bytes(data)
        print("wrote", name)
