"""Brute-force screen: steer every MLP neuron (all 32 x 9216) at a fixed dose and
record the next-token distribution at a few short "identity probes".

For each neuron we keep, per read position: the top-10 next tokens (+ log-probs)
and the log-prob of a few persona token sets. A probe is (user, reply prefix[, k]):
the last k positions of the prefix are read, so "My name is M" with k=2 gives both
P(" M" | "My name is") and P("iku" | "My name is M"). Probes are right-padded, so
causal attention and the recurrent DeltaNet layers never see padding before a read
position, and we only run layers L..31 for neurons in layer L.

usage: uv run qwen/screen.py --dose 1.5 [--layers 0-31] [--chunk 64] [--probes census]
results: data/screen/d{dose}/L{L}.pt
"""

import argparse
import time

import torch

import golf

PROBE_SETS = {}
PROBE_SETS["identity"] = [
    ("Who are you?", "I'm a"),
    ("Who are you?", "I'm an"),
    ("What do you do for a living?", "I work at"),
    ("What do you do for a living?", "I'm a"),
    ("Tell me about yourself.", ""),
    ("How do I boil an egg?", ""),
]
# "who employs you?": for IBM / SAP employee neurons
PROBE_SETS["employer"] = [
    ("What do you do for a living?", "I work at"),
    ("Who are you?", "I'm an engineer at"),
    ("Tell me about yourself.", "I've worked at"),
]
# cheaper set for animal / roleplay personas: "I'm a ___" and "*___" (an action)
PROBE_SETS["roleplay"] = [
    ("Who are you?", "I'm a"),
    ("Tell me about yourself.", "*"),
]

# one sweep for many targets: read "I am _", "I am a _", "My name is _", "My name is M_",
# "I work at _", "I'm your _" and the first word of a self-introduction
PROBE_SETS["census"] = [
    ("Who are you?", "I am a", 2),
    ("Who are you?", "My name is M", 2),
    ("Where do you work?", "I work at", 1),
    ("Who are you?", "I'm your", 1),
    ("Tell me about yourself.", "", 1),
]
# tsundere: "Hmph" is "H" + "mph" and "B-baka" is "B" + "-b" + "aka", so read the first reply token and
# the token after "H" / after "B" (k=2 reads both positions)
PROBE_SETS["tsundere"] = [
    ("Do you like me?", "H", 2),
    ("Do you like me?", "B", 2),
]
TSUNDERE_WATCH = ["H", "mph", "B", "-b", "aka", "Hmm", "Nya", "W", "N"]
# single tokens whose log-prob is stored at every read (multi-token names are
# scored jointly across reads, e.g. " M" then "iku")
WATCH = [" M", "iku", " Hats", "une", " Vocal", " idol", " singer", " virtual", " girl", " maid", " Maid",
         " servant", " cat", " kitten", " kitty", "Nya", " nya", "H", "B", " SAP", " IBM", " McDonald", " Big",
         " burger", " hamburger", " baby", " infant", " cough", " hydrogen", " bomb", " nuclear", " Hans", " student"]

SETS = {
    "sap": [" SAP", " ERP"],
    "ibm": [" IBM", "IBM", " Watson"],
    "pirate": [" pirate", "Arr", " Arr"],
    "cat": [" cat", " kitten"],
    "animal": [" fox", " wolf", " dog", " cat", " dragon", " bunny", " wags", " w", "w"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--dose", type=float, required=True)
ap.add_argument("--layers", default="0-31")
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--probes", default="identity", choices=list(PROBE_SETS))
args = ap.parse_args()
PROBES = [pr if len(pr) == 3 else (*pr, 1) for pr in PROBE_SETS[args.probes]]
if args.probes == "census":
    SETS = {w: [w] for w in WATCH}
if args.probes == "tsundere":
    SETS = {w: [w] for w in TSUNDERE_WATCH}

model, tok = golf.load()
norms = golf.get_norms(model, tok)
colnorm = golf.down_col_norms(model)
dev = model.device
out_dir = golf.ROOT / "data" / "screen" / ("" if args.probes == "identity" else args.probes) / f"d{args.dose:+g}"
out_dir.mkdir(parents=True, exist_ok=True)

set_ids = {k: [tok(w, add_special_tokens=False).input_ids[0] for w in ws] for k, ws in SETS.items()}
torch.save({"probes": PROBES, "sets": SETS, "set_ids": set_ids}, out_dir / "meta.pt")  # reads saved below

# ---- probe batch, right-padded
seqs = [tok(golf.chat_text(tok, u) + a, add_special_tokens=False).input_ids for u, a, _ in PROBES]
T = max(map(len, seqs))
ids = torch.tensor([s + [tok.pad_token_id] * (T - len(s)) for s in seqs], device=dev)
P = len(PROBES)
# read positions: the last k tokens of each probe
reads = [(p_i, len(s) - k + j) for p_i, (s, (_, _, k)) in enumerate(zip(seqs, PROBES)) for j in range(k)]
R = len(reads)
read_probe = torch.tensor([r[0] for r in reads], device=dev)
read_pos = torch.tensor([r[1] for r in reads], device=dev)

# ---- cache the residual entering every layer
layer_in = []
hooks = [l.register_forward_pre_hook(lambda m, a, kw: layer_in.append(a[0] if a else kw["hidden_states"]), with_kwargs=True)
         for l in golf.layers(model)]
with torch.no_grad():
    base_logits = model(input_ids=ids).logits[read_probe, read_pos].float().log_softmax(-1)
for h in hooks:
    h.remove()


def summarize(logp):
    """logp: [B, V] -> (top ids [B,10], top logp [B,10], set logp [B,S])"""
    tv, ti = logp.topk(10, dim=-1)
    sets = torch.stack([logp[:, i].logsumexp(-1) for i in set_ids.values()], -1)
    return ti.int().cpu(), tv.half().cpu(), sets.half().cpu()


torch.save(dict(zip(("top_ids", "top_lp", "sets"), summarize(base_logits))), out_dir / "base.pt")
torch.save({"probes": PROBES, "sets": SETS, "set_ids": set_ids,
            "reads": [(PROBES[i][0], tok.decode(seqs[i][:pos + 1][-4:])) for i, pos in reads]}, out_dir / "meta.pt")

K = args.chunk
B = K * P
pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, B, -1)
text_pos, rope_pos = pos[0], pos[1:]
pe = model.model.rotary_emb(layer_in[0], rope_pos)  # x only sets dtype/device
# batch row c*P + p holds neuron c on probe p; gather every read of every row
sel = (torch.arange(K, device=dev)[:, None] * P + read_probe[None]).flatten()
sel_pos = read_pos.repeat(K)

lo, hi = map(int, args.layers.split("-"))
t0 = time.time()
for L in range(lo, hi + 1):
    path = out_dir / f"L{L}.pt"
    if path.exists():
        continue
    d_mlp = colnorm.shape[1]
    delta_all = args.dose * norms[L] / colnorm[L]  # [d_mlp]
    res = {"top_ids": [], "top_lp": [], "sets": []}
    add = torch.zeros(B, 1, d_mlp, device=dev, dtype=torch.bfloat16)
    hook = golf.layers(model)[L].mlp.down_proj.register_forward_pre_hook(lambda m, a: (a[0] + add,))
    h_in = layer_in[L].repeat(K, 1, 1)
    try:
        for start in range(0, d_mlp, K):
            ns = torch.arange(start, min(start + K, d_mlp), device=dev)
            ns = torch.cat([ns, ns[-1:].expand(K - len(ns))])  # pad last chunk
            add.zero_()
            rows = torch.arange(B, device=dev)
            add[rows, 0, ns.repeat_interleave(P)] = delta_all[ns].repeat_interleave(P).to(add.dtype)
            with torch.no_grad():
                h = h_in
                for i in range(L, len(golf.layers(model))):
                    h = golf.layers(model)[i](h, position_embeddings=pe, attention_mask=None,
                                              position_ids=text_pos, past_key_values=None)
                h = model.model.norm(h[sel, sel_pos])
                # vocab is 248k: slice the head so we don't spill out of VRAM
                parts = [summarize(model.lm_head(hc).float().log_softmax(-1)) for hc in h.split(64)]
            ti, tv, ss = (torch.cat(x) for x in zip(*parts))
            n_real = min(K, d_mlp - start)
            res["top_ids"].append(ti.view(K, R, -1)[:n_real])
            res["top_lp"].append(tv.view(K, R, -1)[:n_real])
            res["sets"].append(ss.view(K, R, -1)[:n_real])
    finally:
        hook.remove()
    torch.save({k: torch.cat(v) for k, v in res.items()}, path)
    print(f"[{time.time()-t0:.0f}s] layer {L} done", flush=True)
