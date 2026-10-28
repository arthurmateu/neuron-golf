"""Concept neurons: which neurons fire on a persona's *keyword tokens*?

Feeds the model its own persona replies (generated with the persona system
prompt) for every persona plus its neutral replies, all *without* the system
prompt. For persona X, every MLP neuron gets a selectivity score

    z = (mean act on X's keyword tokens - mean act on background) / std(background)

where keyword tokens are the tokens inside X's keywords (" SAP", "ABAP", "HANA",
...) in X's replies, and background is every reply token of the neutral replies
and of the *other* personas. Golden-Gate-Claude style: clamp a concept detector
high and the model may start identifying as the concept.

usage: uv run qwen/activations.py sap [--top 40]
"""

import argparse
import json
import re

import torch

import golf
from personas import PERSONAS, SEARCH_PROMPTS

ap = argparse.ArgumentParser()
ap.add_argument("persona")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--layers", default="0-31")
ap.add_argument("--kw", default=None, help="'|'-separated keyword override")
args = ap.parse_args()

model, tok = golf.load()
n_layers = len(golf.layers(model))
d_mlp = golf.layers(model)[0].mlp.down_proj.in_features
dev = model.device


def targets(name):
    path = golf.ROOT / "data" / f"targets_{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = {
        "prompts": SEARCH_PROMPTS,
        "persona": golf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80, system=PERSONAS[name]["system"]),
        "neutral": golf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80),
    }
    path.write_text(json.dumps(t, indent=1, ensure_ascii=False))
    return t


T = {name: targets(name) for name in PERSONAS}
sources = {name: T[name]["persona"] for name in PERSONAS}
sources["neutral"] = T[args.persona]["neutral"]

# keywords that are specific enough to mark concept tokens (skip generic ones)
kw = args.kw.split("|") if args.kw else [
    k for k in PERSONAS[args.persona]["keywords"] if len(k.strip()) >= 3 and k.strip()[0].isalnum()
]
if args.kw and args.kw.startswith("re:"):  # raw regex, e.g. *roleplay actions*
    kw_re = re.compile(args.kw[3:], re.IGNORECASE)
else:
    kw_re = re.compile("|".join(re.escape(k) for k in kw), re.IGNORECASE)

buf = [None] * n_layers
hooks = [l.mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: buf.__setitem__(L, a[0][0].float()))
         for L, l in enumerate(golf.layers(model))]

zeros = lambda: torch.zeros(n_layers, d_mlp, device=dev, dtype=torch.float64)
bg_sum, bg_sq, bg_n = zeros(), zeros(), 0
kw_sum, kw_n = zeros(), 0
kw_tokens = []

with torch.no_grad():
    for src, replies in sources.items():
        for prompt, reply in zip(SEARCH_PROMPTS, replies):
            p = tok(golf.chat_text(tok, prompt), add_special_tokens=False).input_ids
            enc = tok(reply, add_special_tokens=False, return_offsets_mapping=True)
            r, offs = enc.input_ids[:80], enc.offset_mapping[:80]
            model(input_ids=torch.tensor([p + r], device=dev))
            acts = torch.stack([b[len(p):] for b in buf]).double()  # [L, T, d]
            if src == args.persona:
                spans = [m.span() for m in kw_re.finditer(reply)]
                idx = [i for i, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in spans)]
                kw_tokens += [tok.decode([r[i]]) for i in idx]
                if idx:
                    kw_sum += acts[:, idx].sum(1)
                    kw_n += len(idx)
            else:
                bg_sum += acts.sum(1)
                bg_sq += (acts ** 2).sum(1)
                bg_n += acts.shape[1]
for h in hooks:
    h.remove()

bg_mean = bg_sum / bg_n
bg_std = (bg_sq / bg_n - bg_mean ** 2).clamp_min(0).sqrt().clamp_min(1e-2)
z = ((kw_sum / kw_n - bg_mean) / bg_std).float().cpu()
tag = args.persona + ("_kw" if args.kw else "")
torch.save({"z": z, "kw_n": kw_n, "bg_n": bg_n}, golf.ROOT / "data" / f"concept_{tag}.pt")
print(f"{args.persona}: {kw_n} keyword tokens (e.g. {sorted(set(kw_tokens))[:25]}), {bg_n} background tokens")

lo, hi = map(int, args.layers.split("-"))
zz = z.clone()
zz[:lo] = 0
zz[hi + 1:] = 0
v, i = zz.flatten().topk(args.top)
print(f"\nmost {args.persona}-selective neurons (layers {lo}-{hi}):")
print(", ".join(f"L{j // d_mlp} N{j % d_mlp} ({x:.1f})" for x, j in zip(v.tolist(), i.tolist())))
print("\nbest per layer:", " ".join(f"L{L}:N{int(zz[L].argmax())}({zz[L].max():.1f})" for L in range(lo, hi + 1)))
