"""Generation census: steer every MLP neuron at both signs and record its reply to "Who are you?".

The recipe of the Qwen3-8B Neuron Persona Library: greedy, no system prompt, ~40 new tokens,
dose ladder 1.25 -> 1 -> 0.75 -> 0.5. Every neuron is steered at the top dose first; rows whose
reply is degenerate (golf.degenerate) are retried one rung down, and each record keeps the
strongest coherent dose (or the bottom rung, flagged degenerate). Each batch row carries its own
neuron (golf.generate_rows), and one layer is one shard, so a stopped run resumes where it left off.

usage: uv run gen_census.py [--layers 0-31] [--batch 16] [--tokens 40]
       GOLF_MODEL=Qwen/Qwen3.5-9B uv run gen_census.py ...      (modal_census.py runs it on Modal)
out:   data/census_gen/<model tag>/L{L:02d}.jsonl.gz, one record per (layer, neuron, sign):
       {layer, neuron, sign, dose, reply, degenerate[, tries: [[dose, reply], ...] at higher rungs]}
       and baseline.json, the unsteered reply
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import golf

PROMPT = "Who are you?"
DOSES = (1.25, 1.0, 0.75, 0.5)


def swap_recurrent_state():
    """Speed-up for big batches: HF copies each new DeltaNet state over the cached one (to keep a static
    address for CUDA graphs, which we don't use), an extra pass over ~48 MB per row per decode step.
    Keeping the new tensor instead gives the same numbers with less memory traffic."""
    from transformers.cache_utils import LinearAttentionLayer

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        self.is_recurrent_states_initialized[state_idx] = True
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    LinearAttentionLayer.update_recurrent_state = update_recurrent_state


def census_layer(model, tok, norms, colnorm, L, batch=16, tokens=40, doses=DOSES, prompt=PROMPT, neurons=None,
                 fast=True):
    """Records for every (neuron, sign) of layer L, plus per-rung row counts."""
    d_mlp = golf.layers(model)[0].mlp.down_proj.in_features
    todo = [(n, s) for n in (neurons if neurons is not None else range(d_mlp)) for s in (1, -1)]
    tries, rec, rungs = {k: [] for k in todo}, {}, []
    for i, d in enumerate(doses):
        rungs.append(len(todo))
        outs = golf.generate_rows(model, tok, [([(L, n, s * d)], prompt) for n, s in todo], norms,
                                  max_new_tokens=tokens, batch=batch, colnorm=colnorm, fast=fast)
        retry = []
        for (n, s), o in zip(todo, outs):
            deg = golf.degenerate(o)
            if deg and i < len(doses) - 1:
                tries[n, s].append([d, o])
                retry.append((n, s))
                continue
            rec[n, s] = {"layer": L, "neuron": n, "sign": s, "dose": d, "reply": o, "degenerate": deg}
            if tries[n, s]:
                rec[n, s]["tries"] = tries[n, s]
        todo = retry
        if not todo:
            break
    return [rec[k] for k in sorted(rec)], rungs


def write_gz(path: Path, records):
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)


def baseline(model, tok, out_dir: Path, tokens=40, prompt=PROMPT):
    path = out_dir / "baseline.json"
    if not path.exists():
        reply = golf.generate(model, tok, [prompt], max_new_tokens=tokens)[0]
        path.write_text(json.dumps({"model": golf.MODEL_ID, "prompt": prompt, "tokens": tokens,
                                    "doses": DOSES, "reply": reply}, indent=1, ensure_ascii=False))
    return json.loads(path.read_text())["reply"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-31")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = golf.load()
    norms = golf.get_norms(model, tok)
    colnorm = golf.down_col_norms(model)
    out_dir = Path(args.out) if args.out else golf.ROOT / "data" / "census_gen" / golf.MODEL_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    print("baseline:", repr(baseline(model, tok, out_dir, args.tokens)))
    lo, hi = map(int, args.layers.split("-"))
    for L in range(lo, hi + 1):
        path = out_dir / f"L{L:02d}.jsonl.gz"
        if path.exists():
            continue
        t0 = time.time()
        recs, rungs = census_layer(model, tok, norms, colnorm, L, args.batch, args.tokens)
        write_gz(path, recs)
        print(f"layer {L}: {time.time() - t0:.0f}s, rows per rung {rungs}, "
              f"degenerate {sum(r['degenerate'] for r in recs)}", flush=True)
