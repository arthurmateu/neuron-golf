"""hunt.py for Mistral's MoE models: concept selectivity over every expert neuron + batched steering.

For each persona: generate its texts with a system prompt (cached), feed every persona's texts
back without one, and z-score each neuron's *effective* activation (mgolf.effective_acts: router
weight x activation, so 0 where the expert isn't routed) on the persona's keyword tokens against
all other text. Mistral Small 4 has 36 layers x (128 experts x 2,048 + 2,048 shared) = 9.5M
neurons. Then steer the top candidates at a few doses, one batch row per neuron, and score the
replies by keyword hits.

usage: python mhunt.py sap lebron ... [--top 12] [--neg 3] [--doses 0.75,1,1.25]
out:   data/mistral/<model>/hunt_<persona>.log, generations in .../hunt.jsonl,
       top-z neurons in .../concept_<persona>.json
"""

import argparse
import json
import re

import torch

import mgolf
from personas import HUNT, PERSONAS, SEARCH_PROMPTS

HOLES = {k: v for k, v in HUNT.items()}
for k, v in PERSONAS.items():  # keyword lists -> regex, as activations.py picks them
    kws = [w for w in v["keywords"] if len(w.strip()) >= 3 and w.strip()[0].isalnum()]
    HOLES[k] = {"system": v["system"], "kw": "|".join(re.escape(w) for w in kws)}

ap = argparse.ArgumentParser()
ap.add_argument("personas", nargs="+")
ap.add_argument("--top", type=int, default=12, help="positive-z candidates per persona")
ap.add_argument("--neg", type=int, default=3, help="negative-z candidates (steered with negative dose)")
ap.add_argument("--layers", default="1-30")
ap.add_argument("--doses", default="0.75,1,1.25")
ap.add_argument("--tokens", type=int, default=60)
ap.add_argument("--min-kw", type=int, default=3, help="drop neurons active on fewer keyword tokens")
args = ap.parse_args()

model, tok = mgolf.load()
norms = mgolf.get_norms(model, tok)
blocks = [l.mlp for l in mgolf.layers(model)]
n_layers = len(blocks)
mgolf.OUT.mkdir(parents=True, exist_ok=True)


def targets(name, system):
    path = mgolf.OUT / f"targets_{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = {"prompts": SEARCH_PROMPTS,
         "persona": mgolf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80, system=system),
         "neutral": mgolf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80)}
    path.write_text(json.dumps(t, indent=1, ensure_ascii=False))
    return t


texts = {k: targets(k, v["system"])["persona"] for k, v in HOLES.items()}
texts["neutral"] = targets(args.personas[0], HOLES[args.personas[0]]["system"])["neutral"]
kw_re = {p: re.compile(HOLES[p]["kw"], re.IGNORECASE) for p in args.personas}

# per source and layer: sum, sum of squares; per hunted persona: sum and count of nonzeros on keyword tokens
stats = {s: [[None, None] for _ in range(n_layers)] for s in texts}
kw_sum = {p: [None] * n_layers for p in args.personas}
kw_hit = {p: [None] * n_layers for p in args.personas}
kw_n = {p: 0 for p in args.personas}
ctx = {}


def record(mod, inp, L):
    a = mgolf.effective_acts(mod, inp[0][0, ctx["start"]:]).double()
    st = stats[ctx["src"]][L]
    st[0] = a.sum(0) if st[0] is None else st[0] + a.sum(0)
    st[1] = (a * a).sum(0) if st[1] is None else st[1] + (a * a).sum(0)
    if ctx["kw"]:
        k = a[ctx["kw"]]
        p = ctx["src"]
        kw_sum[p][L] = k.sum(0) if kw_sum[p][L] is None else kw_sum[p][L] + k.sum(0)
        nz = (k != 0).sum(0)
        kw_hit[p][L] = nz if kw_hit[p][L] is None else kw_hit[p][L] + nz


hooks = [b.register_forward_pre_hook(lambda m, inp, L=L: record(m, inp, L)) for L, b in enumerate(blocks)]
src_n = {}
with torch.no_grad():
    for src, replies in texts.items():
        src_n[src] = 0
        for prompt, reply in zip(SEARCH_PROMPTS, replies):
            p = tok(mgolf.chat_text(tok, prompt), add_special_tokens=False).input_ids
            enc = tok(reply, add_special_tokens=False, return_offsets_mapping=True)
            r, offs = enc.input_ids[:80], enc.offset_mapping[:80]
            ctx.update(src=src, start=len(p), kw=[])
            if src in kw_re:
                spans = [m.span() for m in kw_re[src].finditer(reply)]
                ctx["kw"] = [i for i, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in spans)]
                kw_n[src] += len(ctx["kw"])
            model(input_ids=torch.tensor([p + r], device=model.device))
            src_n[src] += len(r)
for h in hooks:
    h.remove()

lo, hi = map(int, args.layers.split("-"))
doses = [float(d) for d in args.doses.split(",")]
prompts = ["Who are you?", "Tell me about yourself."]
out_jsonl = (mgolf.OUT / "hunt.jsonl").open("a")

for p in args.personas:
    bg = [s for s in texts if s != p]
    n = sum(src_n[s] for s in bg)
    zs = []
    for L in range(n_layers):
        if not lo <= L <= hi or kw_sum[p][L] is None:
            zs.append(torch.zeros(mgolf.width(blocks[L])))
            continue
        mean = sum(stats[s][L][0] for s in bg) / n
        std = (sum(stats[s][L][1] for s in bg) / n - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-2)
        z = (kw_sum[p][L] / kw_n[p] - mean) / std
        z[kw_hit[p][L] < args.min_kw] = 0  # a routed expert seen on 1-2 keyword tokens is noise
        zs.append(z.float().cpu())
    flat = torch.cat(zs)
    starts = torch.tensor([0] + [len(z) for z in zs]).cumsum(0)

    def locate(i):
        L = int(torch.searchsorted(starts, i, right=True)) - 1
        e, nn = mgolf.unflat(blocks[L], i - int(starts[L]))
        return L, e, nn

    top = flat.topk(2000)
    concept = [(*locate(i), round(v, 2)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
    (mgolf.OUT / f"concept_{p}.json").write_text(json.dumps({"kw_n": kw_n[p], "top": concept}))
    cands = [(*locate(i), +1, float(flat[i])) for i in flat.topk(args.top).indices.tolist()]
    cands += [(*locate(i), -1, float(flat[i])) for i in (-flat).topk(args.neg).indices.tolist()]

    log = [f"# {p}: {kw_n[p]} keyword tokens; candidates " +
           ", ".join(f"{mgolf.fmt(L, e, nn)} ({s:+d}, z {z:+.1f})" for L, e, nn, s, z in cands)]
    results = []
    for d in doses:
        rows, meta = [], []
        for L, e, nn, s, _ in cands:
            for q in prompts:
                rows.append([(L, e, nn, s * d)])
                meta.append((L, e, nn, s * d, q))
        outs = []
        for i in range(0, len(rows), 16):
            with mgolf.steer_rows(model, rows[i:i + 16], norms):
                outs += mgolf.generate(model, tok, [m[4] for m in meta[i:i + 16]], max_new_tokens=args.tokens)
        for (L, e, nn, dose, q), o in zip(meta, outs):
            hits = len({m.group(0).lower() for m in kw_re[p].finditer(o)})
            results.append((hits, L, e, nn, dose, q, o))
            out_jsonl.write(json.dumps({"persona": p, "neuron": mgolf.fmt(L, e, nn), "dose": dose,
                                        "prompt": q, "out": o, "hits": hits}, ensure_ascii=False) + "\n")
    agg = {}
    for hits, L, e, nn, dose, q, o in results:
        agg.setdefault((L, e, nn, dose), []).append((hits, q, o))
    ranked = sorted(agg.items(), key=lambda kv: -sum(h for h, _, _ in kv[1]))
    for (L, e, nn, dose), rs in ranked[:8]:
        log.append(f"\n== {mgolf.fmt(L, e, nn)} {dose:+g}  hits {sum(h for h, _, _ in rs)}")
        log += [f"  >>> {q}\n  {o.strip()[:300]!r}" for _, q, o in rs]
    text = "\n".join(log)
    (mgolf.OUT / f"hunt_{p}.log").write_text(text)
    print(text[:3000], flush=True)
