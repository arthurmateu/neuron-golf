"""Checks mgolf's MoE bookkeeping on a tiny random Mistral 4 model (CPU, seconds, no download).

usage: python test_mgolf.py      (modal_mistral.py::dryrun runs the real scripts on tiny() instead)
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
    print("all good")
