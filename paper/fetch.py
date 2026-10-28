"""Download only the tensors the weight-space analysis needs (unembedding, final norm, every
down_proj), with HTTP range requests into the checkpoint's safetensors shards. CPU, no transformers.

usage: python paper/fetch.py [--input | --gate | --readout] qwen35_4b qwen35_9b qwen3_8b mistral_s4
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


# Mistral Small 4 (MoE, FP8 with one scale per expert): the shared expert plus a fixed set of routed experts per
# layer, chosen to include the five that held docs/mistral.md's working persona neurons (E1, E10, E19, E34, E52)
MOE = {"mistral_s4": ("mistralai/Mistral-Small-4-119B-2603", 36, [0, 1, 2, 10, 19, 34, 52, 100])}
M3L = "language_model.model.layers.{}.mlp."


def fetch_moe(tag):
    """down_LXX.pt = [shared expert | routed experts in MOE order], each 2,048 neurons, dequantized to bf16."""
    from concurrent.futures import ThreadPoolExecutor
    repo, n_layers, experts = MOE[tag]
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    wmap = json.loads(get(url(repo, "model.safetensors.index.json")))["weight_map"]
    hdrs = {sh: header(repo, sh) for sh in set(wmap.values())}

    def piece(name, lo=None, hi=None):
        base, h = hdrs[wmap[name]]
        meta = h[name]
        a, b = meta["data_offsets"]
        if lo is None:
            buf = get(url(repo, wmap[name]), base + a, base + b - 1)
            return torch.frombuffer(bytearray(buf), dtype=DT[meta["dtype"]]).reshape(meta["shape"]).clone()
        per = (b - a) // meta["shape"][0]  # bytes per expert: the expert axis comes first
        buf = get(url(repo, wmap[name]), base + a + lo * per, base + a + hi * per - 1)
        return torch.frombuffer(bytearray(buf), dtype=DT[meta["dtype"]]).reshape([hi - lo] + meta["shape"][1:]).clone()

    for key, name in (("unembed", "language_model.lm_head.weight"), ("norm", "language_model.model.norm.weight")):
        if not (d / f"{key}.pt").exists():
            torch.save(piece(name), d / f"{key}.pt")

    def layer(L):
        p = d / f"down_L{L:02d}.pt"
        if p.exists():
            return
        sh = piece(M3L.format(L) + "shared_experts.down_proj.weight").float()
        sh *= piece(M3L.format(L) + "shared_experts.down_proj.weight_scale_inv").float()
        scale = piece(M3L.format(L) + "experts.down_proj_scale_inv").float()  # [128, 1, 1]
        cols = [sh]
        for e in experts:
            cols.append(piece(M3L.format(L) + "experts.down_proj", e, e + 1)[0].float() * scale[e])
        torch.save(torch.cat(cols, 1).to(torch.bfloat16), p)  # [4096, 2048 * (1 + len(experts))]
        print(tag, f"down_L{L:02d}", flush=True)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(layer, range(n_layers)))
    (d / "experts.json").write_text(json.dumps({"columns": ["shared"] + [f"E{e}" for e in experts], "width": 2048}))


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
        if tag in MOE:
            fetch_moe(tag)
        elif not tag.startswith("--"):
            fetch(tag, "--input" in sys.argv, "--readout" in sys.argv, "--gate" in sys.argv)
