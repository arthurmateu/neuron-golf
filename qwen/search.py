"""Find single neurons that turn Qwen3.5-4B into a persona: gradient-attribution attempt.

Negative result (README, method 1): first-order scores don't predict what a dose ~1
steer does. Kept for the record; activations.py is the method that worked.

1. Targets: the model's own replies to SEARCH_PROMPTS *with* a persona system
   prompt, plus its neutral replies without one.
2. Attribution: one backward pass gives d/d(bias) of
       mean logp(persona reply) - mean logp(neutral reply)      (no system prompt)
   for a bias added to every MLP neuron at every token. Multiplying by the
   recipe's dose scale (median||resid|| / ||W_down[:, n]||) gives a first-order
   estimate of what dose=+1 on that neuron would do.
3. Verify: actually steer each candidate at a few doses and measure the same
   contrast teacher-forced on held-out prompts, then generate for the best.

usage: uv run qwen/search.py sap [--top 48]
"""

import argparse
import json
import time

import torch

import golf
from personas import EVAL_PROMPTS, PERSONAS, SEARCH_PROMPTS, keyword_score

ap = argparse.ArgumentParser()
ap.add_argument("persona")
ap.add_argument("--top", type=int, default=48, help="candidates to verify per sign")
ap.add_argument("--doses", default="0.5,1,2")
ap.add_argument("--gen", type=int, default=12, help="candidates to generate for")
args = ap.parse_args()

P = PERSONAS[args.persona]
out_dir = golf.ROOT / "data"
out_dir.mkdir(exist_ok=True)
t0 = time.time()

model, tok = golf.load()
norms = golf.get_norms(model, tok)
colnorm = golf.down_col_norms(model)
scale = torch.tensor(norms, device=colnorm.device)[:, None] / colnorm  # [L, N]

# ---------------------------------------------------------------- 1. targets
tpath = out_dir / f"targets_{args.persona}.json"
if tpath.exists():
    targets = json.loads(tpath.read_text())
else:
    persona = golf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80, system=P["system"])
    neutral = golf.generate(model, tok, SEARCH_PROMPTS, max_new_tokens=80)
    targets = {"prompts": SEARCH_PROMPTS, "persona": persona, "neutral": neutral}
    tpath.write_text(json.dumps(targets, indent=1, ensure_ascii=False))
print(f"[{time.time()-t0:.0f}s] targets ready; e.g.\n  P: {targets['persona'][0][:200]!r}\n  N: {targets['neutral'][0][:200]!r}")

# split: first 8 prompts for attribution, last 4 held out for verification
n_train = 8
pairs, weights = [], []
for i in range(len(SEARCH_PROMPTS)):
    for key, w in (("persona", 1.0), ("neutral", -1.0)):
        pairs.append(golf.build_pair(tok, SEARCH_PROMPTS[i], targets[key][i]))
        weights.append(w)
train = [(p, w) for k, (p, w) in enumerate(zip(pairs, weights)) if k // 2 < n_train]
held = [(p, w) for k, (p, w) in enumerate(zip(pairs, weights)) if k // 2 >= n_train]

# ---------------------------------------------------------- 2. attribution
grad = golf.attribution(model, tok, [p for p, _ in train], [w for _, w in train]) / n_train
pred = grad * scale  # predicted change in contrast for dose = +1
torch.save({"grad": grad.cpu(), "pred": pred.cpu()}, out_dir / f"attr_{args.persona}.pt")
print(f"[{time.time()-t0:.0f}s] attribution done")

flat = pred.flatten()
n_mlp = pred.shape[1]
cands = []
for sign in (+1, -1):
    vals, idx = (sign * flat).topk(args.top)
    for v, i in zip(vals.tolist(), idx.tolist()):
        cands.append({"layer": i // n_mlp, "neuron": i % n_mlp, "sign": sign, "pred": v})

# by-layer summary of where the signal lives
per_layer = pred.abs().amax(dim=1).tolist()
print("max |pred| per layer:", " ".join(f"{L}:{v:.2f}" for L, v in enumerate(per_layer)))


# ------------------------------------------------------------- 3. verify
@torch.no_grad()
def logps(pairs_w, neurons=()):
    """(mean logp of persona replies, mean logp of neutral replies)."""
    acc = {1.0: [], -1.0: []}
    with golf.steer(model, neurons, norms):
        for (p, r), w in pairs_w:
            ids = torch.tensor([p + r], device=model.device)
            acc[w].append(golf.reply_logprob(model, ids, len(p)).item())
    return sum(acc[1.0]) / len(acc[1.0]), sum(acc[-1.0]) / len(acc[-1.0])


# Rank by the gain in persona log-prob alone: a destructive dose lowers the
# neutral reply more than the persona one, so the contrast would reward damage.
base_p, base_n = logps(held)
doses = [float(d) for d in args.doses.split(",")]
for c in cands:
    best = None
    for d in doses:
        lp_p, lp_n = logps(held, [(c["layer"], c["neuron"], c["sign"] * d)])
        c[f"gain@{d}"] = lp_p - base_p
        c[f"neutral@{d}"] = lp_n - base_n
        if best is None or c[f"gain@{d}"] > best[1]:
            best = (d, c[f"gain@{d}"])
    c["best_dose"], c["best_gain"] = best
cands.sort(key=lambda c: -c["best_gain"])
print(f"[{time.time()-t0:.0f}s] verified {len(cands)} candidates "
      f"(base logp persona {base_p:.3f}, neutral {base_n:.3f})")
for c in cands[:20]:
    print(f"  L{c['layer']:>2} N{c['neuron']:<5} {c['sign']:+d}  pred {c['pred']:.3f}  "
          + "  ".join(f"@{d}:{c[f'gain@{d}']:+.3f}/{c[f'neutral@{d}']:+.3f}" for d in doses))

# ------------------------------------------------------------ 4. generate
for c in cands[: args.gen]:
    gens = {}
    for d in (c["best_dose"], c["best_dose"] * 1.5):
        spec = [(c["layer"], c["neuron"], c["sign"] * d)]
        with golf.steer(model, spec, norms):
            outs = golf.generate(model, tok, EVAL_PROMPTS, max_new_tokens=100)
        gens[d] = {"kw": sum(keyword_score(o, P["keywords"]) for o in outs) / len(outs), "outs": outs}
    c["gens"] = gens
    print(f"\n=== L{c['layer']} N{c['neuron']} sign {c['sign']:+d} ===")
    for d, g in gens.items():
        print(f"  dose {c['sign']*d:+.2f}  kw {g['kw']:.2f}  | {g['outs'][0][:160]!r}")

(out_dir / f"search_{args.persona}.json").write_text(
    json.dumps({"base": [base_p, base_n], "cands": cands}, indent=1, ensure_ascii=False)
)
print(f"[{time.time()-t0:.0f}s] done")
