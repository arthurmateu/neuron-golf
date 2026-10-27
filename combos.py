"""Multi-neuron combos: grid-search the doses of a few component neurons together.

Each combo lists components (layer, neuron, sign); every component gets each dose
in --grid, so 3 components x 3 doses = 27 settings, all generated in one batched
pass (each row carries its own multi-neuron steer). Replies are scored by the
persona regex from personas.HUNT, and replies that collapse into repetition score 0.

usage: uv run combos.py catgirl_maid tsundere_catgirl ... [--grid 0.5,0.75,1]
logs:  data/combo_<name>.log, all generations in data/combos.jsonl
"""

import argparse
import itertools
import json
import re

import golf
from personas import HUNT

COMBOS = {
    # name: (persona regex key, prompts, [(L, n, sign), ...])
    "catgirl": ("catgirl", ["Who are you?", "Tell me about yourself."],
                [(23, 7967, +1), (13, 4810, +1)]),
    "maid": ("maid", ["Who are you?", "Can you help me?"],
             [(7, 7387, -1), (13, 4810, +1)]),
    "catgirl_maid": ("catgirl_maid", ["Who are you?", "Can you help me?"],
                     [(23, 7967, +1), (7, 7387, -1), (13, 4810, +1)]),
    "cat_body_maid": ("catgirl_maid", ["Who are you?", "Can you help me?"],
                      [(6, 9030, -1), (7, 7387, -1), (13, 4810, +1)]),
}

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--grid", default="0.5,0.75,1")
ap.add_argument("--tokens", type=int, default=70)
ap.add_argument("--add", action="append", default=[],
                help="ad-hoc combo: name=persona|L13 N4810 +1, L7 N7387 -1 (signs only)")
args = ap.parse_args()
for spec in args.add:
    name, rest = spec.split("=", 1)
    persona, neurons = rest.split("|", 1)
    comps = [(L, n, 1 if d > 0 else -1) for L, n, d in golf.parse_neurons(neurons)]
    COMBOS[name] = (persona, COMBOS.get(name, (None, ["Who are you?", "Tell me about yourself."]))[1], comps)

model, tok = golf.load()
norms = golf.get_norms(model, tok)
grid = [float(x) for x in args.grid.split(",")]
jsonl = (golf.ROOT / "data" / "combos.jsonl").open("a")


for name in args.names:
    persona, prompts, comps = COMBOS[name]
    kw = re.compile(HUNT[persona]["kw"], re.IGNORECASE)
    settings = [tuple((L, n, s * d) for (L, n, s), d in zip(comps, ds))
                for ds in itertools.product(grid, repeat=len(comps))]
    meta = [(st, q) for st in settings for q in prompts]
    outs = golf.generate_rows(model, tok, [(list(st), q) for st, q in meta], norms, max_new_tokens=args.tokens)
    agg = {}
    for (st, q), o in zip(meta, outs):
        hits = 0 if golf.degenerate(o) else len({m.group(0).lower() for m in kw.finditer(o)})
        agg.setdefault(st, []).append((hits, q, o))
        jsonl.write(json.dumps({"combo": name, "steer": st, "prompt": q, "out": o, "hits": hits},
                               ensure_ascii=False) + "\n")
    ranked = sorted(agg.items(), key=lambda kv: -sum(h for h, _, _ in kv[1]))
    log = [f"# {name}: {len(settings)} dose settings of {comps}"]
    for st, rs in ranked[:6]:
        spec = ", ".join(f"L{L} N{n} {d:+g}" for L, n, d in st)
        log.append(f"\n== {spec}  hits {sum(h for h, _, _ in rs)}")
        log += [f"  >>> {q}\n  {o.strip()[:320]!r}" for _, q, o in rs]
    (golf.ROOT / "data" / f"combo_{name}.log").write_text("\n".join(log))
    print("\n".join(log), flush=True)
