"""Bake a single-neuron steer into the weights, so it runs in stock Ollama.

The steer adds a constant delta = dose * median||resid|| / ||W_down[:, n]|| to
neuron n at every token. Qwen3.5 has no biases, so there is no weight that is
literally "a constant". Two ways to fake one:

hijack  Find the most constant neuron b in the same layer and add the steer to
        its output column:  W_down[:, b] += (delta / mean_b) * W_down[:, n].
        2560 weights, but the best b in practice still varies a lot per token.

rewire  Keep neuron n's output column and refit only its *input* rows so it
        fires ~delta on every token:  silu(g . x) * (u . x) ~= delta, where x is
        the (RMS-normed) MLP input. The normed residual stream has a strong
        common direction, so a 2 x 2560 least-squares/Adam fit gets close.
        One neuron, 5120 weights, natural behaviour of that neuron is replaced.

usage:
  uv run qwen/bake.py analyze "L4 N7892 +1" [prompts...]   # compare hook vs hijack vs rewire
  uv run qwen/bake.py gguf "L4 N7892 +1" out.gguf           # rewired Ollama GGUF
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

import golf

ROOT = golf.ROOT


@torch.no_grad()
def corpus(model, tok, max_new_tokens=64):
    """Token sequences to fit on: chat prompts + the model's replies, plus the
    persona target texts, all wrapped in the chat template (token 0 included)."""
    seqs = []
    for p in golf.CALIB_PROMPTS:
        ids = tok(golf.chat_text(tok, p), return_tensors="pt").to(model.device)
        seqs.append(model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)[0])
    for f in sorted((ROOT / "data").glob("targets_*.json")):
        t = json.loads(f.read_text())
        for prompt, reply in zip(t["prompts"], t["persona"]):
            p, r = golf.build_pair(tok, prompt, reply, 64)
            seqs.append(torch.tensor(p + r, device=model.device))
    return seqs


@torch.no_grad()
def mlp_io(model, seqs, L):
    """(x [T, d_model] MLP inputs, a [T, d_mlp] down_proj inputs) at layer L."""
    xs, acts = [], []
    mlp = golf.layers(model)[L].mlp
    h1 = mlp.register_forward_pre_hook(lambda m, a: xs.append(a[0][0].float()))
    h2 = mlp.down_proj.register_forward_pre_hook(lambda m, a: acts.append(a[0][0].float()))
    try:
        for s in seqs:
            model(input_ids=s[None])
    finally:
        h1.remove()
        h2.remove()
    return torch.cat(xs), torch.cat(acts)


def delta_for(model, norms, L, n, dose):
    return dose * norms[L] / golf.layers(model)[L].mlp.down_proj.weight[:, n].float().norm().item()


def plan_hijack(acts, n, delta, W_down):
    mean, std = acts[1:].mean(0), acts[1:].std(0)
    cv = std / mean.abs().clamp_min(1e-6)
    cv[n] = float("inf")
    b = int(cv.argmin())
    print(f"  hijack: bias neuron N{b} mean {mean[b]:+.3f} cv {cv[b]:.2f}")
    return b, (delta / mean[b].item()) * W_down[:, n].float()


def fit_rewire(X, delta, steps=3000, seed=0):
    """Fit g, u so that silu(X g) * (X u) ~= delta on every row of X."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=gen)
    test, train = perm[: len(X) // 5], perm[len(X) // 5:]
    Xtr = X[train]
    # ridge init: X u0 ~= 1
    lam = 1e-2 * Xtr.pow(2).sum(1).mean()
    A = Xtr.T @ Xtr + lam * torch.eye(X.shape[1], device=X.device)
    u0 = torch.linalg.solve(A, Xtr.T @ torch.ones(len(Xtr), device=X.device))
    G = 8.0
    g = (G * u0).clone().requires_grad_(True)
    u = (delta / G * u0).clone().requires_grad_(True)
    opt = torch.optim.Adam([g, u], lr=1e-3)
    for _ in range(steps):
        r = F.silu(Xtr @ g) * (Xtr @ u) / delta - 1
        loss = r.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ratio = F.silu(X @ g) * (X @ u) / delta
        rt = ratio[test]
        print(f"  rewire: held-out act/delta = {rt.mean():.3f} ± {rt.std():.3f} "
              f"(min {rt.min():.2f}, max {rt.max():.2f}); token 0: {ratio[0]:.2f}")
    return g.detach(), u.detach()


def plan(model, tok, spec):
    """Per steered neuron: hijack column edit and rewire rows (one shared fit corpus)."""
    norms = golf.get_norms(model, tok)
    seqs = corpus(model, tok)
    out = []
    for L, n, dose in golf.parse_neurons(spec):
        mlp = golf.layers(model)[L].mlp
        X, acts = mlp_io(model, seqs, L)
        delta = delta_for(model, norms, L, n, dose)
        print(f"L{L} N{n} dose {dose:+g}: delta {delta:.2f}, natural act of N{n}: "
              f"{acts[:, n].mean():+.3f} ± {acts[:, n].std():.3f} over {len(X)} tokens")
        b, col = plan_hijack(acts, n, delta, mlp.down_proj.weight)
        g, u = fit_rewire(X, delta)
        out.append({"layer": L, "neuron": n, "dose": dose, "delta": delta,
                    "bias_neuron": b, "col_add": col, "gate_row": g, "up_row": u})
    return out


@torch.no_grad()
def apply(model, edits, method):
    """Apply edits in place; returns an undo function."""
    saved = []
    for e in edits:
        mlp = golf.layers(model)[e["layer"]].mlp
        if method == "hijack":
            W = mlp.down_proj.weight
            saved.append((W, e["bias_neuron"], None, W[:, e["bias_neuron"]].clone()))
            W[:, e["bias_neuron"]] += e["col_add"].to(W.dtype)
        else:
            for W, row in ((mlp.gate_proj.weight, e["gate_row"]), (mlp.up_proj.weight, e["up_row"])):
                saved.append((W, None, e["neuron"], W[e["neuron"]].clone()))
                W[e["neuron"]] = row.to(W.dtype)

    def undo():
        for W, col, row, old in saved:
            if col is not None:
                W[:, col] = old
            else:
                W[row] = old
    return undo


def cmd_analyze(spec, prompts):
    model, tok = golf.load()
    norms = golf.get_norms(model, tok)
    edits = plan(model, tok, spec)
    outs = {}
    with golf.steer(model, golf.parse_neurons(spec), norms):
        outs["hook"] = golf.generate(model, tok, prompts, max_new_tokens=120)
    for method in ("hijack", "rewire"):
        undo = apply(model, edits, method)
        outs[method] = golf.generate(model, tok, prompts, max_new_tokens=120)
        undo()
    for i, p in enumerate(prompts):
        print(f"\n>>> {p}")
        for k, v in outs.items():
            print(f"[{k:6}] {v[i].strip()[:400]!r}")
    torch.save(edits, ROOT / "data" / f"bake_{spec.replace(' ', '_')}.pt")


def cmd_gguf(spec, out_path, src_path=None):
    """Copy Ollama's qwen3.5:4b GGUF, replacing only the rewired ffn_gate/ffn_up tensors."""
    import gguf

    cache = ROOT / "data" / f"bake_{spec.replace(' ', '_')}.pt"
    if cache.exists():
        edits = torch.load(cache)
    else:
        model, tok = golf.load()
        edits = plan(model, tok, spec)
        del model
    rows = {}
    for e in edits:
        rows[f"blk.{e['layer']}.ffn_gate.weight"] = (e["neuron"], e["gate_row"].cpu().numpy())
        rows[f"blk.{e['layer']}.ffn_up.weight"] = (e["neuron"], e["up_row"].cpu().numpy())

    src_path = src_path or ollama_blob("qwen3.5:4b")
    reader = gguf.GGUFReader(src_path)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(out_path, arch=arch, use_temp_file=False)
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        sub = field.types[1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field.types[0], sub_type=sub)
    for t in reader.tensors:
        if t.name in rows:
            n, row = rows[t.name]
            # GGUF rows are torch rows: [n_ff, n_embd] for gate/up -> row n = neuron n
            W = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
            W[n] = row
            writer.add_tensor(t.name, W.astype(np.float16))
            print(f"rewired {t.name} row {n} ({t.tensor_type.name} -> F16)")
        else:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)  # byte shape for quants
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print("wrote", out_path)


def ollama_blob(name):
    import subprocess

    mf = subprocess.run(["ollama", "show", "--modelfile", name], capture_output=True, text=True).stdout
    return next(l.split(None, 1)[1] for l in mf.splitlines() if l.startswith("FROM /"))


if __name__ == "__main__":
    cmd, spec, *rest = sys.argv[1:]
    if cmd == "analyze":
        cmd_analyze(spec, rest or ["Who are you?", "How do I boil an egg?", "Write a haiku about autumn."])
    elif cmd == "gguf":
        cmd_gguf(spec, *rest)
