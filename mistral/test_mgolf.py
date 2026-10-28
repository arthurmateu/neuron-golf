"""Checks mgolf's MoE bookkeeping on a tiny random Mistral 4 model (CPU, seconds, no download).

usage: python mistral/test_mgolf.py      (modal_mistral.py::dryrun runs the real scripts on tiny() instead)
"""

import torch
from transformers import Mistral4Config, Mistral4ForCausalLM

import mgolf


def tiny(vocab_size=64):
    """A random 3-layer Mistral 4 (layer 0 dense, then MoE with a shared expert), every weight nonzero."""
    torch.manual_seed(0)
    cfg = Mistral4Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=48, moe_intermediate_size=8, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, n_routed_experts=6, n_shared_experts=1, num_experts_per_tok=2,
        first_k_dense_replace=1, q_lora_rank=16, kv_lora_rank=16, qk_rope_head_dim=4, qk_nope_head_dim=4,
        v_head_dim=8, head_dim=4, n_group=1, topk_group=1,
    )
    model = Mistral4ForCausalLM(cfg).float().eval().requires_grad_(False)
    for p in model.parameters():  # experts and router start at zero/empty
        torch.nn.init.normal_(p, std=0.2)
    return model


if __name__ == "__main__":
    model = tiny()
    cfg = model.config
    blocks = [l.mlp for l in mgolf.layers(model)]
    x = torch.randn(5, cfg.hidden_size)

    # 1. effective activations x W_down columns reproduce each block's output (dense and MoE)
    for L, mlp in enumerate(blocks):
        acts = mgolf.effective_acts(mlp, x)
        assert acts.shape == (5, mgolf.width(mlp)), (L, acts.shape)
        cols = torch.stack([mgolf.out_col(mlp, *mgolf.unflat(mlp, j)) for j in range(acts.shape[1])], 1)
        torch.testing.assert_close(acts @ cols.T, mlp(x[None])[0], rtol=1e-4, atol=1e-4)
        if L >= cfg.first_k_dense_replace:  # routed experts: exactly top-k of them nonzero per token
            per_expert = acts[:, :-cfg.moe_intermediate_size].reshape(5, cfg.n_routed_experts, -1)
            assert ((per_expert != 0).any(-1).sum(-1) == cfg.num_experts_per_tok).all()
    print("effective_acts reproduces mlp(x) on dense + MoE layers")

    # 2. steering adds dose * norm * unit column to the block output, per batch row
    norms = [2.0, 3.0, 5.0]
    ids = torch.randint(0, 64, (2, 7))
    for spec in ["L2 E3 N5 +1.5", "L2 S N1 -1", "L0 N7 +0.5"]:
        (L, e, n, dose), = mgolf.parse_neurons(spec)
        assert mgolf.fmt(L, e, n) == spec.rsplit(" ", 1)[0]
        seen = []

        def watch():  # registered after the steering hook, so it sees the steered output
            h = blocks[L].register_forward_hook(lambda m, i, o: seen.append(o.clone()))
            model(input_ids=ids)
            h.remove()

        with torch.no_grad():
            watch()
            with mgolf.steer_rows(model, [[(L, e, n, dose)], []], norms):
                watch()
        col = mgolf.out_col(blocks[L], e, n)
        torch.testing.assert_close(seen[1][0] - seen[0][0], (dose * norms[L] * col / col.norm()).expand(7, -1))
        torch.testing.assert_close(seen[1][1], seen[0][1])  # row 1 unsteered
    print("steer_rows adds dose * norm * W_down[:, n] / |W_down[:, n]| to the right row")

    # 3. unflat / width layout
    mlp = blocks[1]
    assert mgolf.unflat(mlp, 0) == (0, 0)
    assert mgolf.unflat(mlp, cfg.moe_intermediate_size * 2 + 3) == (2, 3)
    assert mgolf.unflat(mlp, cfg.n_routed_experts * cfg.moe_intermediate_size + 4) == (mgolf.SHARED, 4)
    assert mgolf.width(mlp) == (cfg.n_routed_experts + 1) * cfg.moe_intermediate_size

    # 4. load_fp8 reads a checkpoint laid out like Mistral Small 4's (F8 + bf16 *_scale_inv per tensor /
    #    per expert, *activation_scale, "language_model." prefix) back into a fresh model
    import json
    import tempfile
    from pathlib import Path

    from safetensors.torch import save_file

    sd, ref = {}, {}
    for name, p in model.named_parameters():
        key = "language_model." + name
        if p.dim() < 2 or "embed" in name or "lm_head" in name or name.endswith("mlp.gate.weight"):
            sd[key] = ref[name] = p.detach().to(torch.bfloat16)
            continue
        per_expert = p.dim() == 3
        amax = p.detach().abs().amax(dim=(1, 2), keepdim=True) if per_expert else p.detach().abs().max()
        scale = (amax / 448).to(torch.bfloat16)
        q = (p.detach() / scale.float()).to(torch.float8_e4m3fn)
        sk = key + "_scale_inv" if per_expert else key[:-len("weight")] + "weight_scale_inv"
        ak = key + "_activation_scale" if per_expert else key[:-len("weight")] + "activation_scale"
        sd[key], sd[sk], sd[ak] = q, scale, torch.ones_like(scale[..., 0, 0] if per_expert else scale)
        ref[name] = q.to(torch.bfloat16) * scale
    with tempfile.TemporaryDirectory() as d:
        save_file(sd, f"{d}/model-00001-of-00001.safetensors")
        Path(d, "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}}))
        fresh = tiny().to(torch.bfloat16).requires_grad_(True)  # as a freshly built model is
        mgolf.load_fp8(fresh, Path(d))
    for name, p in fresh.named_parameters():
        torch.testing.assert_close(p, ref[name], rtol=0, atol=0)
    print("load_fp8 dequantizes F8 weights x scale_inv into the right parameters")
    print("all good")
