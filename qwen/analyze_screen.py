"""Rank screened neurons per persona and run the identity census.

usage:
  uv run qwen/analyze_screen.py rank sap [--dose +1.5] [--top 30]
  uv run qwen/analyze_screen.py census [--dose +1.5] [--probe 0]
  uv run qwen/analyze_screen.py token " pirate" [--probe 0]     # who says this word?
"""

import argparse
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer

import golf

ap = argparse.ArgumentParser()
ap.add_argument("cmd")
ap.add_argument("arg", nargs="?")
ap.add_argument("--dose", default="+1.5")
ap.add_argument("--set", default="", help="probe set subdir, e.g. roleplay")
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--probe", type=int, default=0)
ap.add_argument("--probes", default="0,1,2,3", help="probes to average for rank")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(golf.MODEL_ID)


def load(dose):
    d = golf.ROOT / "data" / "screen" / args.set / f"d{dose}"
    meta, base = torch.load(d / "meta.pt"), torch.load(d / "base.pt")
    layers = {int(p.stem[1:]): torch.load(p) for p in sorted(d.glob("L*.pt"))}
    return meta, base, layers


def show(ids, lps=None, k=5):
    toks = [tok.decode([i]) for i in ids[:k].tolist()]
    if lps is None:
        return " ".join(repr(t) for t in toks)
    return " ".join(f"{t!r}:{p:.1f}" for t, p in zip(toks, lps[:k].float().tolist()))


meta, base, layers = load(args.dose)
probes = meta["probes"]
print(f"dose {args.dose}: layers {sorted(layers)}")

if args.cmd == "rank":
    s = list(meta["sets"]).index(args.arg)
    ps = [int(p) for p in args.probes.split(",")]
    rows = []
    for L, r in layers.items():
        # absolute log-prob on the best probe: a gain from -19 to -10 is still ~0
        gain = r["sets"][:, ps, s].float().amax(1)  # [N]
        best = gain.topk(min(args.top, len(gain)))
        rows += [(g, L, n) for g, n in zip(best.values.tolist(), best.indices.tolist())]
    rows.sort(reverse=True)
    print("base set logp per probe:", [round(x, 2) for x in base["sets"][:, s].float().tolist()])
    for g, L, n in rows[: args.top]:
        r = layers[L]
        print(f"L{L:>2} N{n:<5} logp {g:+.2f} | p0 {show(r['top_ids'][n, 0], k=4)} | "
              f"p2 {show(r['top_ids'][n, 2], k=4)} | last {show(r['top_ids'][n, -1], k=3)}")

elif args.cmd == "census":
    p = args.probe
    print("probe:", probes[p], "| base top:", show(base["top_ids"][p], base["top_lp"][p]))
    base_top = base["top_ids"][p, 0].item()
    who = defaultdict(list)
    for L, r in layers.items():
        top1, lp1 = r["top_ids"][:, p, 0], r["top_lp"][:, p, 0]
        for n in (top1 != base_top).nonzero().flatten().tolist():
            who[top1[n].item()].append((lp1[n].item(), L, n))
    cnt = Counter({t: len(v) for t, v in who.items()})
    for t, c in cnt.most_common(args.top * 4):
        ex = sorted(who[t], reverse=True)[:3]
        print(f"{tok.decode([t])!r:>16} x{c:<5} e.g. " + ", ".join(f"L{L} N{n} ({lp:.1f})" for lp, L, n in ex))

elif args.cmd == "token":
    t = tok(args.arg, add_special_tokens=False).input_ids[0]
    p = args.probe
    rows = []
    for L, r in layers.items():
        hit = (r["top_ids"][:, p] == t)
        lp = torch.where(hit, r["top_lp"][:, p].float(), torch.full_like(r["top_lp"][:, p].float(), -99)).amax(1)
        for n in (lp > -99).nonzero().flatten().tolist():
            rows.append((lp[n].item(), L, n))
    rows.sort(reverse=True)
    print(f"token {tok.decode([t])!r} in top-10 at probe {probes[p]} for {len(rows)} neurons")
    for lp, L, n in rows[: args.top]:
        print(f"  L{L:>2} N{n:<5} logp {lp:.2f} | p0 {show(layers[L]['top_ids'][n, 0], k=4)} | last {show(layers[L]['top_ids'][n, -1], k=3)}")
