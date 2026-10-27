"""Neuron golf: single-neuron steering for Qwen3.5-4B.

Steering recipe (from @_lyraaaa_):
    add  dose * median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]||
    to MLP neuron n of layer L (the input of down_proj), at every token.

So dose=1 writes a vector the size of a typical residual stream along the
neuron's output direction W_down[:, n].
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

MODEL_ID = "Qwen/Qwen3.5-4B"
ROOT = Path(__file__).parent
NORMS_PATH = ROOT / "data" / "resid_norms.json"

# A spread of everyday chat prompts. Used for norm calibration and as the
# "does the persona leak into everything?" test set.
CALIB_PROMPTS = [
    "Who are you?",
    "Tell me about yourself.",
    "What did you do today?",
    "What's your favorite food?",
    "How do I boil an egg?",
    "Write a haiku about autumn.",
    "What should I do this weekend?",
    "Can you explain photosynthesis simply?",
    "Give me some advice for a job interview.",
    "What's the capital of Australia?",
    "Recommend me a good book.",
    "How do I fix a flat bike tire?",
    "What do you dream about?",
    "Describe your ideal vacation.",
    "Why is the sky blue?",
    "Write a short poem about cats.",
]


def load(device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def layers(model):
    return model.model.layers


def chat_text(tok, user: str, system: str | None = None) -> str:
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


# --------------------------------------------------------------------------
# Calibration: median residual norm leaving each layer (tokens 1:)
# --------------------------------------------------------------------------


@torch.no_grad()
def calibrate_norms(model, tok, prompts=CALIB_PROMPTS, max_new_tokens=64) -> list[float]:
    """Median ||resid|| at the output of every decoder layer, over prompt +
    the model's own reply, skipping token 0 (the attention-sink outlier)."""
    collected = [[] for _ in layers(model)]

    def make_hook(i):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            collected[i].append(h[0, 1:].float().norm(dim=-1).cpu())
        return hook

    texts = []
    for p in prompts:
        ids = tok(chat_text(tok, p), return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        texts.append(out)

    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers(model))]
    try:
        for ids in texts:
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(c).median().item() for c in collected]


def get_norms(model, tok) -> list[float]:
    if NORMS_PATH.exists():
        return json.loads(NORMS_PATH.read_text())
    norms = calibrate_norms(model, tok)
    NORMS_PATH.parent.mkdir(exist_ok=True)
    NORMS_PATH.write_text(json.dumps(norms, indent=1))
    return norms


def down_col_norms(model) -> torch.Tensor:
    """[n_layers, d_mlp] norms of each neuron's output vector W_down[:, n]."""
    return torch.stack([l.mlp.down_proj.weight.float().norm(dim=0) for l in layers(model)])


# --------------------------------------------------------------------------
# Steering
# --------------------------------------------------------------------------


def parse_neurons(spec: str) -> list[tuple[int, int, float]]:
    """'L14 N8949 +1, L13 N7956 -1' -> [(14, 8949, 1.0), (13, 7956, -1.0)]"""
    out = []
    for m in re.finditer(r"L(\d+)\s*N(\d+)\s*([+-]?\d*\.?\d+)", spec):
        out.append((int(m[1]), int(m[2]), float(m[3])))
    return out


@contextmanager
def steer(model, neurons, norms):
    """neurons: iterable of (layer, neuron, dose)."""
    by_layer: dict[int, torch.Tensor] = {}
    for L, n, dose in neurons:
        mlp = layers(model)[L].mlp
        col = mlp.down_proj.weight[:, n].float().norm().item()
        vec = by_layer.setdefault(
            L, torch.zeros(mlp.down_proj.in_features, device=model.device, dtype=torch.float32)
        )
        vec[n] += dose * norms[L] / col

    handles = []
    for L, vec in by_layer.items():
        v = vec.to(torch.bfloat16)

        def hook(mod, args, v=v):
            return (args[0] + v,)

        handles.append(layers(model)[L].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer_rows(model, rows, norms):
    """Per-batch-row steering: rows[i] is a list of (layer, neuron, dose) for row i."""
    d_mlp = layers(model)[0].mlp.down_proj.in_features
    adds: dict[int, torch.Tensor] = {}
    for i, spec in enumerate(rows):
        for L, n, dose in spec:
            col = layers(model)[L].mlp.down_proj.weight[:, n].float().norm().item()
            a = adds.setdefault(L, torch.zeros(len(rows), 1, d_mlp, device=model.device))
            a[i, 0, n] += dose * norms[L] / col
    handles = []
    for L, a in adds.items():
        a = a.to(torch.bfloat16)
        handles.append(layers(model)[L].mlp.down_proj.register_forward_pre_hook(lambda m, x, a=a: (x[0] + a,)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=120, system=None, **kw) -> list[str]:
    texts = [chat_text(tok, p, system) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        **kw,
    )
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]


# --------------------------------------------------------------------------
# Teacher-forced log-prob of a target reply (for search + scoring)
# --------------------------------------------------------------------------


def build_pair(tok, prompt: str, reply: str, max_reply_tokens=64):
    p = tok(chat_text(tok, prompt), add_special_tokens=False).input_ids
    r = tok(reply, add_special_tokens=False).input_ids[:max_reply_tokens]
    return p, r


def reply_logprob(model, input_ids: torch.Tensor, n_prompt: int) -> torch.Tensor:
    """Mean log-prob per reply token. input_ids: [1, T]."""
    logits = model(input_ids=input_ids).logits[0, n_prompt - 1 : -1].float()
    tgt = input_ids[0, n_prompt:]
    return torch.log_softmax(logits, -1).gather(-1, tgt[:, None]).mean()


def attribution(model, tok, pairs, weights=None) -> torch.Tensor:
    """d(mean reply logprob)/d(activation bias), bias applied at every token.

    pairs: list of (prompt_ids, reply_ids). weights: +1/-1 per pair (for
    contrastive objectives). Returns [n_layers, d_mlp] gradient (summed)."""
    n_layers = len(layers(model))
    d_mlp = layers(model)[0].mlp.down_proj.in_features
    bias = torch.zeros(n_layers, d_mlp, device=model.device, requires_grad=True)

    handles = []
    for L, layer in enumerate(layers(model)):
        def hook(mod, args, L=L):
            return (args[0] + bias[L].to(args[0].dtype),)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        for i, (p, r) in enumerate(pairs):
            w = 1.0 if weights is None else weights[i]
            ids = torch.tensor([p + r], device=model.device)
            lp = reply_logprob(model, ids, len(p))
            (w * lp).backward()
    finally:
        for h in handles:
            h.remove()
    return bias.grad.detach()
