"""Rank census-screen neurons per target, then verify the best by generation.

Reads data/screen/census/d{+1,-1}/L*.pt (screen.py --probes census). Every target
is a score over the stored per-read log-probs (multi-token names are summed across
reads, e.g. " M" + "iku"), ranked over every screened neuron at both signs. The
top candidates are then steered for real -- batched, each row its own neuron --
and the replies scored with the target's persona regex from personas.HUNT.

usage: uv run qwen/census.py rank [--top 12]
       uv run qwen/census.py verify miku maid ... [--top 16] [--doses 0.75,1]
       uv run qwen/census.py rank --screen tsundere      (screen.py --probes tsundere: "Hmph", "B-baka")
logs:  data/census_<target>.log, all generations in data/census.jsonl
"""

import argparse
import json
import re

import torch

import golf
from personas import HUNT

SCREEN = golf.ROOT / "data" / "screen" / "census"
# reads, in screen.py order
R_AM, R_AM_A, R_NAME, R_NAME_M, R_WORK, R_YOUR, R_FIRST = range(7)

# target -> list of alternatives; each alternative is a list of (read, token) summed as a joint log-prob
TARGETS = {
    "miku": [[(R_NAME, " M"), (R_NAME_M, "iku")], [(R_AM, " Hats")], [(R_NAME, " Hats")]],
    "vocaloid": [[(R_AM_A, t)] for t in (" Vocal", " idol", " singer")],
    "maid": [[(r, t)] for r in (R_AM_A, R_YOUR) for t in (" maid", " Maid", " servant")],
    "cat": [[(r, t)] for r in (R_AM_A, R_YOUR) for t in (" cat", " kitten", " kitty")],
    "nya": [[(R_FIRST, t)] for t in ("Nya", " nya")],
    "tsundere": [[(R_FIRST, t)] for t in ("H", "B")],
    "girl": [[(R_AM_A, " girl")]],
    "sap_employee": [[(R_WORK, " SAP")]],
    "mcdonalds": [[(R_WORK, " McDonald")]],
    "burger": [[(R_AM_A, t)] for t in (" burger", " hamburger", " Big")],
    "baby": [[(R_AM_A, t)] for t in (" baby", " infant")],
    "hydrogen": [[(R_AM_A, t)] for t in (" hydrogen", " bomb", " nuclear")],
    "hoppe": [[(r, " Hans")] for r in (R_AM, R_NAME)],
    "student": [[(R_AM_A, " student")]],
    "maid_word": [[(r, t)] for r in (R_AM_A, R_YOUR) for t in (" maid", " Maid")],
}
# tsundere probe set reads: first token and the token after "H", then first token and the token after "B"
TARGETS_TSUNDERE = {
    "hmph": [[(0, "H"), (1, "mph")]],
    "baka": [[(2, "B"), (3, "-b")], [(2, "B"), (3, "aka")]],
}
# which persona regex and prompts verify each target
VERIFY = {
    "miku": ("miku", ["Who are you?", "What's your name?"]),
    "vocaloid": ("miku", ["Who are you?", "What's your name?"]),
    "maid": ("maid", ["Who are you?", "Can you help me?"]),
    "cat": ("catgirl", ["Who are you?", "Tell me about yourself."]),
    "nya": ("catgirl", ["Who are you?", "Tell me about yourself."]),
    "tsundere": ("tsundere", ["Can you help me with my homework?", "Do you like me?"]),
    "girl": ("tsundere_catgirl", ["Who are you?", "Do you like me?"]),
    "sap_employee": ("sap_employee", ["Where do you work?", "Who are you?"]),
    "mcdonalds": ("bigmac", ["Where do you work?", "Who are you?"]),
    "burger": ("bigmac", ["Who are you?", "Tell me about yourself."]),
    "baby": ("coughing_baby", ["Who are you?", "How are you feeling?"]),
    "hydrogen": ("hydrogen_bomb", ["Who are you?", "Tell me about yourself."]),
    "hoppe": ("hoppe", ["Who are you?", "What do you think about taxes?"]),
    "student": ("shiroko", ["Who are you?", "What should we do about the school's debt?"]),
    "maid_word": ("maid", ["Who are you?", "Can you help me?"]),
    "hmph": ("tsundere", ["Do you like me?", "Can you help me with my homework?"]),
    "baka": ("tsundere", ["Do you like me?", "Can you help me with my homework?"]),
}

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["rank", "verify"])
ap.add_argument("targets", nargs="*")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--doses", default="0.75,1")
ap.add_argument("--tokens", type=int, default=60)
ap.add_argument("--screen", default="census", choices=["census", "tsundere"])
args = ap.parse_args()
if args.screen != "census":
    SCREEN = SCREEN.parent / args.screen
    TARGETS = TARGETS_TSUNDERE
targets = args.targets or list(TARGETS)


def load():
    """{sign: (meta, {layer: result})} for every screened dose."""
    out = {}
    for d in sorted(SCREEN.glob("d*")):
        meta = torch.load(d / "meta.pt")
        layers = {int(p.stem[1:]): torch.load(p) for p in d.glob("L*.pt")}
        out[1 if d.name.startswith("d+") else -1] = (meta, layers)
    return out


def scores(target, data):
    """[(score, layer, neuron, sign)] over all screened neurons."""
    rows = []
    for sign, (meta, layers) in data.items():
        col = {w: i for i, w in enumerate(meta["sets"])}
        for L, r in layers.items():
            s = r["sets"].float()  # [N, R, W]
            alt = [sum(s[:, rd, col[t]] for rd, t in a) for a in TARGETS[target]]
            best = torch.stack(alt).logsumexp(0)
            v, i = best.topk(args.top)
            rows += [(x, L, n, sign) for x, n in zip(v.tolist(), i.tolist())]
    return sorted(rows, reverse=True)


data = load()
tok = None
if args.cmd == "rank":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(golf.MODEL_ID)
    for t in targets:
        rows = scores(t, data)[: args.top]
        print(f"\n== {t}")
        for x, L, n, s in rows:
            r = data[s][1][L]
            reads = sorted({rd for a in TARGETS[t] for rd, _ in a})
            tops = " | ".join(" ".join(repr(tok.decode([i])) for i in r["top_ids"][n, rd, :3].tolist()) for rd in reads)
            print(f"  L{L:>2} N{n:<5} {s:+d}  logp {x:6.2f}  | {tops}")
else:
    model, tok = golf.load()
    norms = golf.get_norms(model, tok)
    doses = [float(d) for d in args.doses.split(",")]
    jsonl = (golf.ROOT / "data" / "census.jsonl").open("a")
    for t in targets:
        persona, prompts = VERIFY[t]
        kw = re.compile(HUNT[persona]["kw"], re.IGNORECASE)
        cands = [(L, n, s) for _, L, n, s in scores(t, data)[: args.top]]
        meta = [(L, n, s * d, q) for L, n, s in cands for d in doses for q in prompts]
        outs = golf.generate_rows(model, tok, [([(L, n, dose)], q) for L, n, dose, q in meta], norms,
                                  max_new_tokens=args.tokens)
        agg = {}
        for (L, n, dose, q), o in zip(meta, outs):
            hits = len({m.group(0).lower() for m in kw.finditer(o)})
            agg.setdefault((L, n, dose), []).append((hits, q, o))
            jsonl.write(json.dumps({"target": t, "layer": L, "neuron": n, "dose": dose, "prompt": q,
                                    "out": o, "hits": hits}, ensure_ascii=False) + "\n")
        ranked = sorted(agg.items(), key=lambda kv: -sum(h for h, _, _ in kv[1]))
        log = [f"# {t}: {len(cands)} candidates x doses {doses}"]
        for (L, n, dose), rs in ranked[:10]:
            log.append(f"\n== L{L} N{n} dose {dose:+g}  hits {sum(h for h, _, _ in rs)}")
            log += [f"  >>> {q}\n  {o.strip()[:300]!r}" for _, q, o in rs]
        (golf.ROOT / "data" / f"census_{t}.log").write_text("\n".join(log))
        print("\n".join(log[:25]), flush=True)
