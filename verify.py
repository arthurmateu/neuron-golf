"""Generate with candidate neurons at several doses and score persona keywords.

usage:
  uv run verify.py "L13 N4810, L17 N123" --doses 1,1.5,2 [--persona sap] [--tokens 100]
Each comma-separated neuron is tested *alone*; the dose list overrides any dose
given in the spec (the spec's sign is kept).
"""

import argparse
import json

import golf
from personas import EVAL_PROMPTS, PERSONAS, keyword_score

ap = argparse.ArgumentParser()
ap.add_argument("neurons")
ap.add_argument("--doses", default="1,1.5,2")
ap.add_argument("--persona", default=None)
ap.add_argument("--tokens", type=int, default=100)
ap.add_argument("--prompts", default=None, help="'|'-separated prompts (default: EVAL_PROMPTS[:4])")
ap.add_argument("--show", type=int, default=2, help="prompts to print in full")
args = ap.parse_args()

prompts = args.prompts.split("|") if args.prompts else EVAL_PROMPTS[:4]
model, tok = golf.load()
norms = golf.get_norms(model, tok)
out_path = golf.ROOT / "data" / "verify.jsonl"

def parse(item):
    """'L13 N4810' or 'L13 N4810 -1' -> (13, 4810, sign)"""
    parts = item.split()
    return int(parts[0][1:]), int(parts[1][1:]), float(parts[2]) if len(parts) > 2 else 1.0


for L, n, d0 in map(parse, args.neurons.split(",")):
    sign = 1 if d0 >= 0 else -1
    print(f"\n{'=' * 30} L{L} N{n} {'=' * 30}")
    for d in (float(x) for x in args.doses.split(",")):
        dose = sign * d
        with golf.steer(model, [(L, n, dose)], norms):
            outs = golf.generate(model, tok, prompts, max_new_tokens=args.tokens)
        kws = {p: sum(keyword_score(o, v["keywords"]) for o in outs) / len(outs) for p, v in PERSONAS.items()}
        best = max(kws, key=kws.get)
        focus = f"{args.persona}={kws[args.persona]:.1f} " if args.persona else ""
        print(f"\n--- dose {dose:+g}  {focus}(top kw: {best}={kws[best]:.1f})")
        for p, o in list(zip(prompts, outs))[: args.show]:
            print(f"  >>> {p}\n  {o.strip()[:400]!r}")
        for p, o in list(zip(prompts, outs))[args.show:]:
            print(f"  >>> {p} :: {o.strip()[:120]!r}")
        with out_path.open("a") as f:
            f.write(json.dumps({"layer": L, "neuron": n, "dose": dose, "kw": kws,
                                "prompts": prompts, "outs": outs}, ensure_ascii=False) + "\n")
