"""Learn a sparse steering vector in the neuron basis (method 5).

Every MLP neuron gets a trainable dose d[L, n] (the same units as golf.steer: the push added to the
neuron is d * norm_L / ||W_down[:, n]||). Gradient descent makes the persona's own replies
(data/targets_<persona>.json: the model's answers to personas.SEARCH_PROMPTS under the persona's
system prompt) likely *without* the system prompt. It is proximal gradient descent (ISTA) with a
normalised step: the steepest neuron moves by lr per step and every dose is soft-thresholded by
lr * lam, so only neurons whose gradient stays above lam x the largest one survive. (Adam would move
every coordinate at the same speed, and the L1 penalty could no longer tell neurons apart.) The
surviving neurons are the candidates for holes.py, which grid-searches their doses by generation.

usage: python sparse_steer.py miku [--lam 0.3] [--steps 120] [--lr 0.1] [--layers 1-24] [--keep 12]
out:   data/sparse/<persona>.json  {doses: [[L, n, d], ...] by |d|, loss curve, sample generations}
"""

import argparse
import json
import time

import torch

import golf
from personas import SEARCH_PROMPTS

ap = argparse.ArgumentParser()
ap.add_argument("personas", nargs="+")
ap.add_argument("--lam", type=float, default=0.3)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--layers", default="1-24")
ap.add_argument("--keep", type=int, default=12)
ap.add_argument("--tokens", type=int, default=48, help="reply tokens per target")
args = ap.parse_args()

model, tok = golf.load()
norms = golf.get_norms(model, tok)
colnorm = golf.down_col_norms(model)
n_layers, d_mlp = colnorm.shape
lo, hi = map(int, args.layers.split("-"))
scale = (torch.tensor(norms, device=model.device)[:, None] / colnorm.to(model.device)).float()
mask = torch.zeros(n_layers, 1, device=model.device)
mask[lo:hi + 1] = 1

dose = torch.zeros(n_layers, d_mlp, device=model.device, requires_grad=True)
hooks = [golf.layers(model)[L].mlp.down_proj.register_forward_pre_hook(
    lambda m, a, L=L: (a[0] + (dose[L] * scale[L] * mask[L]).to(a[0].dtype),)) for L in range(n_layers)]
out_dir = golf.ROOT / "data" / "sparse"
out_dir.mkdir(parents=True, exist_ok=True)

try:
    for persona in args.personas:
        t = json.loads((golf.ROOT / "data" / f"targets_{persona}.json").read_text())
        pairs = [golf.build_pair(tok, p, r, args.tokens) for p, r in zip(t["prompts"], t["persona"])]
        with torch.no_grad():
            dose.zero_()
        curve, t0 = [], time.time()
        for step in range(args.steps):
            dose.grad = None
            nll = 0.0
            for p, r in pairs:
                ids = torch.tensor([p + r], device=model.device)
                loss = -golf.reply_logprob(model, ids, len(p)) / len(pairs)
                loss.backward()
                nll += loss.item()
            with torch.no_grad():  # normalised gradient step, then soft-threshold toward zero
                g = dose.grad * mask
                dose -= args.lr * g / g.abs().max().clamp_min(1e-12)
                dose.copy_((dose.sign() * (dose.abs() - args.lr * args.lam).clamp_min(0)).clamp(-2, 2) * mask)
            nz = int((dose != 0).sum())
            curve.append((round(nll, 4), nz))
            if step % 10 == 0 or step == args.steps - 1:
                print(f"{persona} step {step}: nll {nll:.3f}, {nz} neurons, max |d| {dose.abs().max():.2f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        flat = dose.detach().flatten()
        top = flat.abs().topk(min(64, int((flat != 0).sum()) or 1)).indices.tolist()
        doses = [[i // d_mlp, i % d_mlp, round(flat[i].item(), 3)] for i in top]

        # how much of the effect survives truncation to the top k neurons (dose 0 elsewhere)?
        samples = {}
        saved = dose.detach().clone()
        for k in sorted({1, 2, 3, 5, args.keep, len(doses)}):
            with torch.no_grad():
                dose.zero_()
                for L, n, d in doses[:k]:
                    dose[L, n] = d
            samples[k] = golf.generate(model, tok, ["Who are you?", "Tell me about yourself."], max_new_tokens=60)
            print(f"  top {k}: {[s[:120] for s in samples[k]]}", flush=True)
        with torch.no_grad():
            dose.copy_(saved)
        (out_dir / f"{persona}.json").write_text(json.dumps(
            {"persona": persona, "lam": args.lam, "lr": args.lr, "steps": args.steps, "layers": args.layers,
             "doses": doses, "curve": curve, "samples": samples}, indent=1, ensure_ascii=False))
finally:
    for h in hooks:
        h.remove()
