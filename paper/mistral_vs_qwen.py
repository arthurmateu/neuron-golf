"""Mistral Small 4's working persona neurons (docs/mistral.md): what they write, and their best weight-space match
in Qwen3.5-4B and 9B, with what the Qwen censuses say that match does.

Needs paper/fetch.py mistral_s4 (shared expert + 8 routed experts per layer, including the experts below).

usage: python paper/mistral_vs_qwen.py
out:   paper/data/mistral_s4_personas.json
"""

import gzip
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from weights import W, best_matches, down, readout, vocab  # noqa: E402

COLS = json.loads((W / "mistral_s4" / "experts.json").read_text())["columns"]
PERSONAS = [  # (layer, expert, neuron, dose, what it did in docs/mistral.md)
    (15, "E19", 1133, 1.25, "SAP Fiori application"), (18, "E10", 382, -1.0, "pirate"),
    (20, "E52", 228, 0.75, "AI assistant created by IBM"), (15, "E34", 1511, 1.25, "student of the Mises Institute"),
    (25, "E1", 1003, 1.25, "crying baby")]


def census(tag):
    out = {}
    for p in sorted((HERE.parent / "data" / "census_gen" / tag).glob("L*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[(r["layer"], r["neuron"], r["sign"])] = " ".join(r["reply"].split())[:160]
    return out


if __name__ == "__main__":
    cols = torch.stack([(1 if d > 0 else -1) * down("mistral_s4", L)[:, 2048 * COLS.index(e) + n].clone()
                        for L, e, n, d, _ in PERSONAS], 1)
    U = readout("mistral_s4")
    U -= U.mean(0)
    inv = {i: t for t, i in vocab("mistral_s4").items()}
    out = [{"neuron": f"L{L} {e} N{n} {d:+g}", "persona": what,
            "top_tokens": [inv[i].replace("Ġ", " ") for i in (U @ c).topk(10).indices.tolist()]}
           for (L, e, n, d, what), c in zip(PERSONAS, cols.T)]
    del U
    for tag, ct in (("qwen35_4b", "4b"), ("qwen35_9b", "9b")):
        idx, cos, tenth, _, lab = best_matches("mistral_s4", tag, cols)
        reply = census(ct)
        for o, i, c, t in zip(out, idx, cos, tenth):
            L, n = (int(x) for x in lab[i])
            s = 1 if c > 0 else -1
            o[tag] = {"match": f"L{L} N{n} {s:+d}", "cos": round(float(abs(c)), 3), "tenth": round(float(t), 3),
                      "census_reply": reply[(L, n, s)]}
    (HERE / "data" / "mistral_s4_personas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
