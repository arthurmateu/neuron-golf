"""The Qwen3.5 family tree from unembeddings alone: which models keep a subset of which others' residual
channels (lineage.py's channel map, for every pair of sizes).

A channel "has a partner" in the other model if its best |correlation| (over the shared vocabulary)
beats the 99th percentile of what randomly rotated channels reach. If 4B was pruned from 9B, nearly
every 4B channel has a 9B partner. If 4B and 9B were both pruned from 27B instead, some 4B channels
have a 27B partner but none in 9B.

usage: python paper/family.py     (needs paper/fetch.py --readout for each size)
out:   paper/data/family.json
"""

import itertools
import json
from pathlib import Path

import torch

from lineage import zcols
from weights import W

HERE = Path(__file__).parent
SIZES = ["qwen35_08b", "qwen35_2b", "qwen35_4b", "qwen35_9b", "qwen35_27b"]  # one tokenizer: all 248k tokens
torch.set_num_threads(4)


def unembed(tag):
    return zcols(torch.load(W / tag / "unembed.pt"))


def best(A, B):
    """per channel of A: best |corr| with any channel of B, and which."""
    v, i = torch.zeros(A.shape[1]), torch.zeros(A.shape[1], dtype=torch.long)
    for s in range(0, B.shape[1], 1024):  # [V, dA]^T [V, 1024] in chunks: keeps memory flat
        c = (A.T @ B[:, s:s + 1024]).abs()
        m, j = c.max(1)
        upd = m > v
        v[upd], i[upd] = m[upd], j[upd] + s
    return v, i


def null_p99(A, B, k=512):
    """99th percentile of the best |corr| reached by k random rotations of A's channels."""
    Q, _ = torch.linalg.qr(torch.randn(A.shape[1], k, generator=torch.Generator().manual_seed(0)))
    return float(best(zcols(A @ Q), B)[0].quantile(0.99))


if __name__ == "__main__":
    tags = [t for t in SIZES if (W / t / "unembed.pt").exists()]
    out, keep = {"pairs": {}}, {}
    for a, b in itertools.combinations(tags, 2):  # a smaller than b; two models in memory at a time
        A, B = unembed(a), unembed(b)
        v, i = best(A, B)
        thr = null_p99(A, B)
        del A, B
        out["pairs"][f"{a}~{b}"] = {"median_best": float(v.median()), "null_p99": thr,
                                    "frac_with_partner": float((v > thr).float().mean()),
                                    "distinct_partners": len(set(i[v > thr].tolist())), "channels": len(v)}
        keep[(a, b)] = v > thr
        print(a, b, out["pairs"][f"{a}~{b}"], flush=True)
    # cascade or siblings? 4B channels with a 27B partner but no 9B partner
    if ("qwen35_4b", "qwen35_9b") in keep and ("qwen35_4b", "qwen35_27b") in keep:
        p9, p27 = keep[("qwen35_4b", "qwen35_9b")], keep[("qwen35_4b", "qwen35_27b")]
        out["4b_channels"] = {"partner_in_9b_and_27b": float((p9 & p27).float().mean()),
                              "only_27b": float((~p9 & p27).float().mean()),
                              "only_9b": float((p9 & ~p27).float().mean()),
                              "neither": float((~p9 & ~p27).float().mean())}
        print(out["4b_channels"])
    (HERE / "data" / "family.json").write_text(json.dumps(out, indent=1))
