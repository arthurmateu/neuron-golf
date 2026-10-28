"""Download only the tensors the weight-space analysis needs (unembedding, final norm, every
down_proj), with HTTP range requests into the checkpoint's safetensors shards. CPU, no transformers.

usage: python paper/fetch.py [--input | --gate | --readout] qwen35_4b qwen35_9b qwen3_8b
       --input also fetches the input side: embedding, every gate_proj and post-attention norm
       --gate fetches only every gate_proj (for lineage.py's second check)
       --readout fetches only the unembedding and final norm (for lineage.py's channel map)
out:   paper/weights/<tag>/{unembed,norm,down_LXX}.pt  (bf16, as stored)
       paper/weights/<tag>/{embed,gate_LXX,ln_LXX}.pt   with --input
"""

import json
import struct
import sys
from pathlib import Path

import requests
import torch

OUT = Path(__file__).parent / "weights"
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}

# tag: (repo, layers, unembedding tensor, final norm tensor, layer prefix, embedding tensor)
Q35 = "model.language_model."
M3 = "language_model."
MODELS = {
    "qwen35_4b": ("Qwen/Qwen3.5-4B", 32, Q35 + "embed_tokens.weight", Q35 + "norm.weight", Q35 + "layers.{}.",
                  None),  # tied
    "qwen35_9b": ("Qwen/Qwen3.5-9B", 32, "lm_head.weight", Q35 + "norm.weight", Q35 + "layers.{}.",
                  Q35 + "embed_tokens.weight"),
    "qwen35_08b": ("Qwen/Qwen3.5-0.8B", 24, Q35 + "embed_tokens.weight", Q35 + "norm.weight", Q35 + "layers.{}.",
                   None),  # tied
    "qwen35_2b": ("Qwen/Qwen3.5-2B", 24, Q35 + "embed_tokens.weight", Q35 + "norm.weight", Q35 + "layers.{}.",
                  None),  # tied
    "qwen35_27b": ("Qwen/Qwen3.5-27B", 64, "lm_head.weight", Q35 + "norm.weight", Q35 + "layers.{}.",
                   Q35 + "embed_tokens.weight"),
    "qwen3_8b": ("Qwen/Qwen3-8B", 36, "lm_head.weight", "model.norm.weight", "model.layers.{}.",
                 "model.embed_tokens.weight"),
    "ministral3_3b": ("mistralai/Ministral-3-3B-Base-2512", 26, M3 + "model.embed_tokens.weight",  # tied
                      M3 + "model.norm.weight", M3 + "model.layers.{}.", None),
    "ministral3_8b": ("mistralai/Ministral-3-8B-Base-2512", 34, M3 + "lm_head.weight", M3 + "model.norm.weight",
                      M3 + "model.layers.{}.", M3 + "model.embed_tokens.weight"),
    "ministral3_14b": ("mistralai/Ministral-3-14B-Base-2512", 40, M3 + "lm_head.weight", M3 + "model.norm.weight",
                       M3 + "model.layers.{}.", M3 + "model.embed_tokens.weight"),
}


def url(repo, f):
    return f"https://huggingface.co/{repo}/resolve/main/{f}"


def get(u, start=None, end=None):
    h = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for _ in range(5):
        try:
            r = requests.get(u, headers=h, timeout=600)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print("retry", e, flush=True)
    raise RuntimeError(u)


def header(repo, shard):
    n = struct.unpack("<Q", get(url(repo, shard), 0, 7))[0]
    return 8 + n, json.loads(get(url(repo, shard), 8, 8 + n - 1))


def tensor(repo, shard, hdr, name):
    base, h = hdr
    meta = h[name]
    a, b = meta["data_offsets"]
    buf = bytearray(get(url(repo, shard), base + a, base + b - 1))
    return torch.frombuffer(buf, dtype=DT[meta["dtype"]]).reshape(meta["shape"]).clone()


def fetch(tag, inputs=False, readout=False, gate=False):
    repo, n_layers, unemb, norm, layer, emb = MODELS[tag]
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    try:
        wmap = json.loads(get(url(repo, "model.safetensors.index.json")))["weight_map"]
    except RuntimeError:
        wmap = None
    want = {"unembed": unemb, "norm": norm} | {f"down_L{i:02d}": layer.format(i) + "mlp.down_proj.weight"
                                               for i in range(n_layers)}
    if readout:
        want = {"unembed": unemb, "norm": norm}
    if gate:
        want = {f"gate_L{i:02d}": layer.format(i) + "mlp.gate_proj.weight" for i in range(n_layers)}
    if inputs:
        want = ({"embed": emb} if emb else {}) | {
            k: layer.format(i) + v for i in range(n_layers)
            for k, v in ((f"ln_L{i:02d}", "post_attention_layernorm.weight"), (f"gate_L{i:02d}", "mlp.gate_proj.weight"))}
    hdrs = {}
    for key, name in want.items():
        p = d / f"{key}.pt"
        if p.exists():
            continue
        shard = wmap[name] if wmap else "model.safetensors"
        if shard not in hdrs:
            hdrs[shard] = header(repo, shard)
        torch.save(tensor(repo, shard, hdrs[shard], name), p)
        print(tag, key, flush=True)


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        if not tag.startswith("--"):
            fetch(tag, "--input" in sys.argv, "--readout" in sys.argv, "--gate" in sys.argv)
