"""One-shot persona hunt: concept selectivity + batched steering, many personas at once.

For each persona in HUNT: generate its texts with a system prompt (cached), rank
all 294,912 neurons by z-scored activation on the persona's keyword tokens vs.
every other text (activations.py, method 3), then steer the top candidates at a
few doses -- one batched generate per dose, each batch row carrying its own
neuron -- and score the replies by keyword hits.

usage: uv run hunt.py miku lebron ... [--top 12] [--neg 3] [--doses 0.75,1,1.25]
log:   data/hunt_<persona>.log, all generations in data/hunt.jsonl
"""

import argparse
import json
import re

import torch

import golf
from personas import HUNT, PERSONAS, SEARCH_PROMPTS

ap = argparse.ArgumentParser()
ap.add_argument("personas", nargs="+")
ap.add_argument("--top", type=int, default=12, help="positive-z candidates per persona")
ap.add_argument("--neg", type=int, default=3, help="negative-z candidates (steered with negative dose)")
ap.add_argument("--layers", default="1-24")
ap.add_argument("--doses", default="0.75,1,1.25")
ap.add_argument("--tokens", type=int, default=60)
args = ap.parse_args()

model, tok = golf.load()
norms = golf.get_norms(model, tok)
dev = model.device
n_layers, d_mlp = len(golf.layers(model)), golf.layers(model)[0].mlp.down_proj.in_features


def targets(name, system):
    path = golf.ROOT / "data" / f"targets_{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = {"prompts": SEARCH_PROMPTS,
         "persona": golf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80, system=system),
         "neutral": golf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80)}
    path.write_text(json.dumps(t, indent=1, ensure_ascii=False))
    return t


# every persona's texts (old + new) serve as background for the others
systems = {k: v["system"] for k, v in PERSONAS.items()} | {k: v["system"] for k, v in HUNT.items()}
texts = {k: targets(k, s)["persona"] for k, s in systems.items()}
texts["neutral"] = targets(args.personas[0], systems[args.personas[0]])["neutral"]

buf = [None] * n_layers
hooks = [l.mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: buf.__setitem__(L, a[0][0].float()))
         for L, l in enumerate(golf.layers(model))]
zeros = lambda: torch.zeros(n_layers, d_mlp, device=dev, dtype=torch.float64)
src_sum, src_sq, src_n = {}, {}, {}
kw_sum = {p: zeros() for p in args.personas}
kw_n = {p: 0 for p in args.personas}
kw_re = {p: re.compile(HUNT[p]["kw"], re.IGNORECASE) for p in args.personas}

with torch.no_grad():
    for src, replies in texts.items():
        src_sum[src], src_sq[src], src_n[src] = zeros(), zeros(), 0
        for prompt, reply in zip(SEARCH_PROMPTS, replies):
            p = tok(golf.chat_text(tok, prompt), add_special_tokens=False).input_ids
            enc = tok(reply, add_special_tokens=False, return_offsets_mapping=True)
            r, offs = enc.input_ids[:80], enc.offset_mapping[:80]
            model(input_ids=torch.tensor([p + r], device=dev))
            acts = torch.stack([b[len(p):] for b in buf]).double()  # [L, T, d]
            src_sum[src] += acts.sum(1)
            src_sq[src] += (acts ** 2).sum(1)
            src_n[src] += acts.shape[1]
            if src in kw_re:
                spans = [m.span() for m in kw_re[src].finditer(reply)]
                idx = [i for i, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in spans)]
                if idx:
                    kw_sum[src] += acts[:, idx].sum(1)
                    kw_n[src] += len(idx)
for h in hooks:
    h.remove()

lo, hi = map(int, args.layers.split("-"))
doses = [float(d) for d in args.doses.split(",")]
prompts = ["Who are you?", "Tell me about yourself."]
out_jsonl = (golf.ROOT / "data" / "hunt.jsonl").open("a")

for p in args.personas:
    bg = [s for s in texts if s != p]
    n = sum(src_n[s] for s in bg)
    mean = sum(src_sum[s] for s in bg) / n
    std = (sum(src_sq[s] for s in bg) / n - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-2)
    z = ((kw_sum[p] / max(kw_n[p], 1) - mean) / std).float()
    z[:lo] = 0
    z[hi + 1:] = 0
    torch.save({"z": z.cpu(), "kw_n": kw_n[p]}, golf.ROOT / "data" / f"concept_{p}.pt")
    cands = [(i // d_mlp, i % d_mlp, +1) for i in z.flatten().topk(args.top).indices.tolist()]
    cands += [(i // d_mlp, i % d_mlp, -1) for i in (-z).flatten().topk(args.neg).indices.tolist()]

    log = [f"# {p}: {kw_n[p]} keyword tokens; candidates " +
           ", ".join(f"L{L} N{nn} ({s:+d}, z {z[L, nn]:+.1f})" for L, nn, s in cands)]
    results = []
    for d in doses:
        rows, meta = [], []
        for L, nn, s in cands:
            for q in prompts:
                rows.append([(L, nn, s * d)])
                meta.append((L, nn, s * d, q))
        outs = []
        for i in range(0, len(rows), 16):  # keep recurrent state small: 16 rows per generate
            with golf.steer_rows(model, rows[i:i + 16], norms):
                outs += golf.generate(model, tok, [m[3] for m in meta[i:i + 16]], max_new_tokens=args.tokens)
        for (L, nn, dose, q), o in zip(meta, outs):
            hits = len({m.group(0).lower() for m in kw_re[p].finditer(o)})
            results.append((hits, L, nn, dose, q, o))
            out_jsonl.write(json.dumps({"persona": p, "layer": L, "neuron": nn, "dose": dose,
                                        "prompt": q, "out": o, "hits": hits}, ensure_ascii=False) + "\n")
    # aggregate per (neuron, dose)
    agg = {}
    for hits, L, nn, dose, q, o in results:
        agg.setdefault((L, nn, dose), []).append((hits, q, o))
    ranked = sorted(agg.items(), key=lambda kv: -sum(h for h, _, _ in kv[1]))
    for (L, nn, dose), rs in ranked[:8]:
        log.append(f"\n== L{L} N{nn} dose {dose:+g}  hits {sum(h for h, _, _ in rs)}")
        log += [f"  >>> {q}\n  {o.strip()[:300]!r}" for _, q, o in rs]
    text = "\n".join(log)
    (golf.ROOT / "data" / f"hunt_{p}.log").write_text(text)
    print(text[:3000], flush=True)
