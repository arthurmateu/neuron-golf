"""Rank concept-neuron candidates by what real steering does to the teacher-forced
log-prob of the model's own persona replies (no system prompt).

Negative result (README, method 4): at dose ~1 every candidate lowers the persona
log-prob, so this does not separate good neurons from bad ones.

Candidates: top-N by |z| from activations.py (data/concept_<persona>.pt) within a
layer range. Metric per dose: mean logp gain on persona replies, alongside the
change on neutral replies (a damage meter).

usage: uv run rank_tf.py furry --n 200 --layers 6-22 --doses 1,1.5
"""

import argparse
import json

import torch

import golf

ap = argparse.ArgumentParser()
ap.add_argument("persona")
ap.add_argument("--concept", default=None, help="concept file tag (default: persona)")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--layers", default="6-22")
ap.add_argument("--doses", default="1,1.5")
ap.add_argument("--top", type=int, default=25)
args = ap.parse_args()

model, tok = golf.load()
norms = golf.get_norms(model, tok)
T = json.loads((golf.ROOT / "data" / f"targets_{args.persona}.json").read_text())
z = torch.load(golf.ROOT / "data" / f"concept_{args.concept or args.persona}.pt")["z"]
lo, hi = map(int, args.layers.split("-"))
z[:lo] = 0
z[hi + 1:] = 0
d_mlp = z.shape[1]
_, idx = z.abs().flatten().topk(args.n)
cands = [(i // d_mlp, i % d_mlp, 1.0 if z.flatten()[i] > 0 else -1.0) for i in idx.tolist()]

# a handful of (prompt, reply) pairs, 48 reply tokens each, padded into one batch
pairs = {k: [golf.build_pair(tok, p, r, 48) for p, r in zip(T["prompts"][:6], T[k][:6])]
         for k in ("persona", "neutral")}


def batch(ps):
    L = max(len(p) + len(r) for p, r in ps)
    ids = torch.full((len(ps), L), tok.pad_token_id)
    mask = torch.zeros(len(ps), L, dtype=torch.bool)
    for i, (p, r) in enumerate(ps):
        ids[i, : len(p) + len(r)] = torch.tensor(p + r)
        mask[i, len(p): len(p) + len(r)] = True
    return ids.to(model.device), mask.to(model.device)


B = {k: batch(v) for k, v in pairs.items()}


@torch.no_grad()
def logp(kind, neurons=()):
    ids, mask = B[kind]  # right-padded: pads never precede a scored token
    with golf.steer(model, neurons, norms):
        lp = model(input_ids=ids).logits[:, :-1].float().log_softmax(-1)
    tok_lp = lp.gather(-1, ids[:, 1:, None])[..., 0]
    m = mask[:, 1:]
    return (tok_lp * m).sum().item() / m.sum().item()


base = {k: logp(k) for k in B}
rows = []
for L, n, s in cands:
    for d in (float(x) for x in args.doses.split(",")):
        spec = [(L, n, s * d)]
        rows.append((logp("persona", spec) - base["persona"], logp("neutral", spec) - base["neutral"], L, n, s * d, z[L, n].item()))
rows.sort(reverse=True)
print(f"base logp persona {base['persona']:.3f} neutral {base['neutral']:.3f}")
for gp, gn, L, n, d, zz in rows[: args.top]:
    print(f"L{L:>2} N{n:<5} dose {d:+.2f}  persona {gp:+.3f}  neutral {gn:+.3f}  (z {zz:+.1f})")
