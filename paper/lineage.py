"""Are two models' neurons related by descent, not just by function? Two checks in raw coordinates:

1. channel map: correlate every residual channel of A's unembedding with every channel of B's over the
   shared tokens. Independent training only agrees up to a rotation, so the best single-channel match is
   no better than for a randomly rotated A (the null). A width-pruned model keeps a subset of its parent's
   channels, so each channel finds a partner.
2. weights through the map: for neuron pairs matched by weights.py (logit lens only), compare the raw
   down_proj column, and the gate_proj row if fetched (paper/fetch.py --input), after reindexing A's
   channels to B's. Random pairs give the baseline.

usage: python paper/lineage.py <A> <B>
out:   paper/data/lineage.json (one entry per pair, merged)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from weights import W, shared  # noqa: E402

torch.set_num_threads(4)


def zcols(x):
    x = x.float()
    x -= x.mean(0)
    return x.div_(x.norm(dim=0).clamp_min(1e-9))  # in place: a 27B unembedding is 5 GB in fp32


def channel_map(a, b):
    ia, ib = shared(a, b)
    A = zcols(torch.load(W / a / "unembed.pt")[ia])
    B = zcols(torch.load(W / b / "unembed.pt")[ib])
    C = A.T @ B
    best, pi = C.abs().max(1)
    sign = torch.sign(C.gather(1, pi[:, None])[:, 0])
    Q, _ = torch.linalg.qr(torch.randn(A.shape[1], A.shape[1], generator=torch.Generator().manual_seed(0)))
    null = (zcols(A @ Q).T @ B).abs().max(1).values
    stats = {"best_corr_median": float(best.median()), "null_median": float(null.median()),
             "frac_above_null_max": float((best > null.max()).float().mean()),
             "distinct_partners": len(set(pi.tolist())), "channels": A.shape[1]}
    return pi, sign, stats


def through_map(a, b, pi, sign, kind):
    """median |cos| of raw weights for matched pairs vs random pairs, per A layer."""
    if not (HERE / "data" / f"match_{a}_{b}.npz").exists():
        return {}
    z = np.load(HERE / "data" / f"match_{a}_{b}.npz")
    rng = np.random.default_rng(0)
    out = {}
    for L in np.unique(z["a_layer"]):
        fa, fb = W / a / f"{kind}_L{L:02d}.pt", None
        m = z["a_layer"] == L
        if not fa.exists():
            continue
        XA = torch.load(fa).float()
        XA = XA if kind == "down" else XA.T  # columns = neurons, rows = residual channels
        res = {"twin": [], "random": []}
        for Lb in np.unique(z["b_layer"][m]):
            fb = W / b / f"{kind}_L{Lb:02d}.pt"
            if not fb.exists():
                continue
            XB = torch.load(fb).float()
            XB = (XB if kind == "down" else XB.T)[pi] * sign[:, None]
            sel = np.flatnonzero(m & (z["b_layer"] == Lb))
            x = XA[:, z["a_neuron"][sel]]
            for k, cols in (("twin", z["b_neuron"][sel]), ("random", rng.integers(0, XB.shape[1], len(sel)))):
                res[k] += torch.nn.functional.cosine_similarity(x, XB[:, cols], dim=0).abs().tolist()
        if res["twin"]:
            out[str(int(L))] = {k: float(np.median(v)) for k, v in res.items()}
    return out


if __name__ == "__main__":
    a, b = sys.argv[1:3]
    pi, sign, stats = channel_map(a, b)
    entry = {"channel_map": stats, "down": through_map(a, b, pi, sign, "down"),
             "gate": through_map(a, b, pi, sign, "gate")}
    p = HERE / "data" / "lineage.json"
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[f"{a}~{b}"] = entry
    p.write_text(json.dumps(allr, indent=1))
    print(json.dumps(entry, indent=1))
