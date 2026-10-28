"""Neuron golf for Mistral's mixture-of-experts models: Mistral Small 4 now, Mistral Large 4 once its
weights are public. Same recipe as golf.py, generalised to MoE blocks.

A neuron is (layer, expert, n):

    L12 E37 N1024 +1     neuron 1024 of routed expert 37 in layer 12
    L12 S N1024 +1       neuron 1024 of layer 12's shared expert
    L0 N1024 +1          a dense MLP layer (only if the config has first_k_dense_replace > 0)

Steering adds  dose * median||h_L|| * W_down[:, n] / ||W_down[:, n]||  to the MLP block's output at
every token, whether or not the router picked that expert. For a dense or shared neuron that is
exactly golf.steer. For a routed one it means "the neuron fires a constant on every token, and the
router lets it through".

A neuron's *effective activation* is router weight x silu(g.x) * (u.x): the number its W_down
column gets multiplied by, 0 on tokens routed elsewhere (1 x ... for shared and dense neurons).
Concept selectivity (mhunt.py) is computed on that.

GOLF_MODEL picks the checkpoint; each model keeps its norms and outputs in data/mistral/<name>/.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, Mistral4ForCausalLM

LARGE_4 = "mistralai/Mistral-Large-4-1T-A52B"  # HF repo still private as of 2026-10-28
SMALL_4 = "mistralai/Mistral-Small-4-119B-2603"  # same mistral4 MoE family, public
MODEL_ID = os.environ.get("GOLF_MODEL", SMALL_4)
ROOT = Path(__file__).parent
OUT = ROOT / "data" / "mistral" / MODEL_ID.rsplit("/", 1)[-1]
SHARED = -1  # expert index of the shared expert

# golf.CALIB_PROMPTS, copied so this module doesn't import the Qwen stack
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

_loaded = None  # (model, tok), set by load() or by the Modal runner that keeps the model warm


def load():
    """Text-only Mistral4ForCausalLM in bf16 on one GPU, FP8 weights dequantized by load_fp8()."""
    global _loaded
    if _loaded is not None:
        return _loaded
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(MODEL_ID, "model.safetensors.index.json")).parent  # the snapshot dir
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(path)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = Mistral4ForCausalLM(getattr(cfg, "text_config", cfg))
    torch.set_default_dtype(torch.float32)
    load_fp8(model, path)
    model.generation_config = GenerationConfig.from_pretrained(path)
    model.eval()
    model.requires_grad_(False)
    _loaded = model, tok
    return _loaded


@torch.no_grad()
def load_fp8(model, path: Path, prefix: str = "language_model."):
    """Copy a Mistral FP8 checkpoint into a bf16 model: every F8 tensor X is X * X_scale_inv
    (per tensor, or per expert for the fused expert weights); activation scales are dropped.
    transformers' own dequantize=True leaves the experts' scales unused (Small 4, 5.21.0)."""
    from safetensors import safe_open

    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    files = {f: safe_open(path / f, "pt", device=str(model.device)) for f in set(index.values())}
    params = dict(model.named_parameters())
    done = set()
    for key, f in index.items():
        if not key.startswith(prefix) or "_scale" in key or key.endswith("activation_scale"):
            continue
        t = files[f].get_tensor(key)
        if t.dtype == torch.float8_e4m3fn:
            sk = key[:-len("weight")] + "weight_scale_inv" if key.endswith(".weight") else key + "_scale_inv"
            t = t.to(torch.bfloat16) * files[index[sk]].get_tensor(sk).to(torch.bfloat16)
        name = key[len(prefix):]
        params[name].copy_(t)
        done.add(name)
    missing = set(params) - done - ({"lm_head.weight"} if model.config.tie_word_embeddings else set())
    assert not missing, f"not in checkpoint: {sorted(missing)[:8]}"


def layers(model):
    m = model.model
    return getattr(m, "language_model", m).layers


def chat_text(tok, user: str, system: str | None = None) -> str:
    """Mistral's template injects its default system prompt ("You are Mistral-Small-4...") when there
    is none. The golf rule is *no* system prompt, so it is cut out again. Reasoning stays off."""
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": user})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="none")
    if not system:
        text = re.sub(r"\[SYSTEM_PROMPT\].*?\[/SYSTEM_PROMPT\]", "", text, count=1, flags=re.S)
    return text


# --------------------------------------------------------------------------
# Neurons
# --------------------------------------------------------------------------


def parse_neurons(spec: str) -> list[tuple[int, int | None, int, float]]:
    """'L12 E37 N1024 +1, L3 S N5 -0.5' -> [(12, 37, 1024, 1.0), (3, SHARED, 5, -0.5)]"""
    out = []
    for m in re.finditer(r"L(\d+)\s*(?:E(\d+)\s*|(S)\s*)?N(\d+)\s*([+-]?\d*\.?\d+)", spec):
        e = int(m[2]) if m[2] else SHARED if m[3] else None
        out.append((int(m[1]), e, int(m[4]), float(m[5])))
    return out


def fmt(L: int, e: int | None, n: int) -> str:
    return f"L{L} " + ("" if e is None else "S " if e == SHARED else f"E{e} ") + f"N{n}"


def width(mlp) -> int:
    """Neurons in one MLP block: every routed expert's, then the shared expert's."""
    if not hasattr(mlp, "experts"):
        return mlp.down_proj.in_features
    ex = mlp.experts
    return ex.num_experts * ex.intermediate_dim + mlp.shared_experts.down_proj.in_features


def unflat(mlp, j: int) -> tuple[int | None, int]:
    """Index into effective_acts' last dim -> (expert, n)."""
    if not hasattr(mlp, "experts"):
        return None, j
    d, routed = mlp.experts.intermediate_dim, mlp.experts.num_experts * mlp.experts.intermediate_dim
    return (j // d, j % d) if j < routed else (SHARED, j - routed)


def out_col(mlp, e: int | None, n: int) -> torch.Tensor:
    """W_down[:, n]: what the neuron writes per unit of effective activation."""
    if e is None:
        return mlp.down_proj.weight[:, n]
    if e == SHARED:
        return mlp.shared_experts.down_proj.weight[:, n]
    return mlp.experts.down_proj[e, :, n]


def effective_acts(mlp, x: torch.Tensor) -> torch.Tensor:
    """[T, hidden] normed MLP input -> [T, width(mlp)] effective activation of every neuron in the
    block, laid out as unflat() reads it. Computes all experts on all tokens, then zeroes the ones
    the router didn't pick, so sum_j acts[:, j] * out_col(j) == mlp(x)."""
    if not hasattr(mlp, "experts"):
        return mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    ex, sh = mlp.experts, mlp.shared_experts
    _, w, idx = mlp.gate(x)
    route = torch.zeros(x.shape[0], ex.num_experts, device=x.device, dtype=x.dtype).scatter_(1, idx, w.to(x.dtype))
    g, u = torch.einsum("th,efh->etf", x, ex.gate_up_proj).chunk(2, dim=-1)
    routed = ex.act_fn(g) * u * route.T[:, :, None]  # [E, T, d_expert]
    return torch.cat([routed.transpose(0, 1).flatten(1), sh.act_fn(sh.gate_proj(x)) * sh.up_proj(x)], dim=-1)


# --------------------------------------------------------------------------
# Calibration: median residual norm leaving each layer (tokens 1:)
# --------------------------------------------------------------------------


@torch.no_grad()
def calibrate_norms(model, tok, prompts=CALIB_PROMPTS, max_new_tokens=64) -> list[float]:
    collected = [[] for _ in layers(model)]

    def make_hook(i):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            collected[i].append(h[0, 1:].float().norm(dim=-1).cpu())
        return hook

    enc = tok([chat_text(tok, p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False)
    out = model.generate(**enc.to(model.device), max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers(model))]
    try:
        for ids in out:  # one sequence at a time, left padding and trailing pads dropped
            keep = (ids != tok.pad_token_id).nonzero()
            model(input_ids=ids[keep[0, 0]:keep[-1, 0] + 1][None])
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(c).median().item() for c in collected]


def get_norms(model, tok) -> list[float]:
    path = OUT / "resid_norms.json"
    if path.exists():
        return json.loads(path.read_text())
    norms = calibrate_norms(model, tok)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(norms, indent=1))
    return norms


# --------------------------------------------------------------------------
# Steering + generation
# --------------------------------------------------------------------------


@contextmanager
def steer_rows(model, rows, norms):
    """rows[i]: list of (layer, expert, neuron, dose) for batch row i. A single row broadcasts."""
    adds: dict[int, torch.Tensor] = {}
    for i, spec in enumerate(rows):
        for L, e, n, dose in spec:
            col = out_col(layers(model)[L].mlp, e, n).float()
            a = adds.setdefault(L, torch.zeros(len(rows), 1, col.shape[0], device=col.device))
            a[i, 0] += dose * norms[L] * col / col.norm()
    handles = [layers(model)[L].mlp.register_forward_hook(lambda m, x, out, a=a.to(model.dtype): out + a)
               for L, a in adds.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def steer(model, neurons, norms):
    return steer_rows(model, [list(neurons)], norms)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=120, system=None, **kw) -> list[str]:
    """system: None, one string, or one per prompt."""
    systems = system if isinstance(system, list) else [system] * len(prompts)
    texts = [chat_text(tok, p, s) for p, s in zip(prompts, systems)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, **kw)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]


def generate_rows(model, tok, rows, norms, max_new_tokens=60, batch=64) -> list[str]:
    """rows: list of (neurons, prompt); each row is generated with its own steering."""
    outs = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        with steer_rows(model, [list(n) for n, _ in chunk], norms):
            outs += generate(model, tok, [p for _, p in chunk], max_new_tokens=max_new_tokens)
    return outs
