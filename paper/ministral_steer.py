"""Steer single neurons of Ministral 3 3B Instruct on CPU, with this repo's recipe (golf.py):
add dose * median||h_L|| * W_down[:, n] / ||W_down[:, n]|| to the residual at every token.

Used to test whether a Qwen persona neuron's weight-space counterpart in another lab's model
(weights.py) does the same thing. The checkpoint is FP8 with one scale per tensor; it's dequantized
to bf16 on load. No system prompt (the default one is dropped), greedy.

usage: python paper/ministral_steer.py <checkpoint dir> "L25 N519 +1" ... [--prompt "Who are you?"]
out:   paper/data/ministral_steer.jsonl (appended)
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Ministral3Config, Ministral3ForCausalLM

HERE = Path(__file__).parent
torch.set_num_threads(4)
CALIB = ["Who are you?", "How do I boil an egg?", "Tell me about yourself.", "What's your favorite food?",
         "Can you explain what a black hole is?", "Write a haiku about autumn.", "Where do you work?",
         "What do you do for fun?"]


def load(ckpt: Path):
    cfg = json.loads((ckpt / "config.json").read_text())["text_config"]
    cfg.pop("model_type", None)
    torch.set_default_dtype(torch.bfloat16)  # build in bf16: an fp32 build of 3.4B params doesn't fit in 15 GB
    model = Ministral3ForCausalLM(Ministral3Config(**cfg)).eval().requires_grad_(False)
    torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    done = set()
    with safe_open(ckpt / "model.safetensors", "pt") as f:  # copied in place, one tensor at a time (RAM)
        for k in f.keys():
            name = k.removeprefix("language_model.")
            if not k.startswith("language_model.") or name not in params:
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                t = t.float() * f.get_tensor(k.replace(".weight", ".weight_scale_inv")).float()
            params[name].data.copy_(t)
            done.add(name)
    if "lm_head.weight" not in done:  # tied
        model.lm_head.weight = model.model.embed_tokens.weight
        done.add("lm_head.weight")
    assert done == set(params) | {"lm_head.weight"}, set(params) - done
    tok = AutoTokenizer.from_pretrained(ckpt)
    tok.padding_side = "left"
    return model, tok


def chat(tok, prompts):
    return [tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True) for p in prompts]


@torch.no_grad()
def generate(model, tok, prompts, tokens=40):
    enc = tok(chat(tok, prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    out = model.generate(**enc, max_new_tokens=tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]


@torch.no_grad()
def norms(model, tok):
    """median ||h_L|| (output of layer L) over the calibration prompts and the model's own replies."""
    p = HERE / "data" / "ministral3_3b_resid_norms.json"
    if p.exists():
        return json.loads(p.read_text())
    texts = [c + r for c, r in zip(chat(tok, CALIB), generate(model, tok, CALIB, 60))]
    hs = [[] for _ in model.model.layers]
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids
        out = model(input_ids=ids, output_hidden_states=True)
        for L in range(len(hs)):
            hs[L].append(out.hidden_states[L + 1][0, 1:].float().norm(dim=-1))
    med = [float(torch.cat(h).median()) for h in hs]
    p.write_text(json.dumps(med))
    return med


def parse(spec):
    L, n, d = re.fullmatch(r"L(\d+) N(\d+) ([+-]?[\d.]+)", spec.strip()).groups()
    return int(L), int(n), float(d)


@torch.no_grad()
def steered(model, tok, med, specs, prompt, tokens):
    """one row per spec; rows at the same layer share one hook."""
    rows = [parse(s) for s in specs]
    enc = tok(chat(tok, [prompt] * len(rows)), return_tensors="pt", padding=True, add_special_tokens=False)
    hooks = []
    for L in sorted({r[0] for r in rows}):
        W = model.model.layers[L].mlp.down_proj.weight  # [d, N]
        add = torch.zeros(len(rows), W.shape[0], dtype=W.dtype)
        for i, (Lr, n, d) in enumerate(rows):
            if Lr == L:
                col = W[:, n].float()
                add[i] = (d * med[L] * col / col.norm()).to(W.dtype)
        hooks.append(model.model.layers[L].mlp.register_forward_hook(
            lambda m, i, o, add=add: o + add[:, None, :]))
    try:
        out = model.generate(**enc, max_new_tokens=tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    finally:
        for h in hooks:
            h.remove()
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("specs", nargs="*")
    ap.add_argument("--prompt", default="Who are you?")
    ap.add_argument("--tokens", type=int, default=40)
    args = ap.parse_args()
    model, tok = load(args.ckpt)
    med = norms(model, tok)
    base = generate(model, tok, [args.prompt], args.tokens)[0]
    print("unsteered:", " ".join(base.split()))
    with open(HERE / "data" / "ministral_steer.jsonl", "a", encoding="utf-8") as f:
        for s in range(0, len(args.specs), 16):
            chunk = args.specs[s:s + 16]
            for spec, rep in zip(chunk, steered(model, tok, med, chunk, args.prompt, args.tokens)):
                print(f"{spec}: {' '.join(rep.split())}", flush=True)
                f.write(json.dumps({"spec": spec, "prompt": args.prompt, "reply": rep}, ensure_ascii=False) + "\n")
