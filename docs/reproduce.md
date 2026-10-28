# Reproduce

Run everything from the repo root. Scripts are addressed by path (`uv run qwen/steer.py …`) and read and
write `data/` and `results/` there.

## Setup

Use a uv-managed Python (`uv venv --managed-python`). Triton (via flash-linear-attention) compiles a
small C helper and needs `Python.h`, which a bare system Python on Ubuntu/WSL lacks.

```bash
uv sync                                   # torch, transformers>=5.21, flash-linear-attention, gguf
uv run hf download Qwen/Qwen3.5-4B
```

Layers and neuron indices are 0-based (`model.model.layers[L].mlp.down_proj` input channel `n`), the same
convention as the tweets, since both of their neurons reproduce.

## Local (one 12 GB GPU)

```bash
uv run qwen/steer.py "L4 N9198 -0.75" "Who are you?"                 # one neuron
uv run qwen/steer.py "L14 N8949 +1, L13 N7956 -1" "Who are you?"      # several at once
uv run qwen/steer.py "L2 N5516 +1; L8 N6411 +1" "Where do you work?"  # compare specs
uv run qwen/steer.py "L4 N7892 +1"                                    # interactive

uv run qwen/hunt.py lebron miku --top 30 --neg 15        # concept search + batched steering (personas in hunt.py)
uv run qwen/activations.py sap                           # concept ranking for personas.py personas
uv run qwen/verify.py "L4 N7892, L24 N8806" --doses 0.75,1,1.5 --persona sap
uv run qwen/screen.py --dose 1 --probes employer --layers 2-8         # brute-force probe screen
uv run qwen/tests.py --controls 40 && uv run qwen/tests.py            # mood & quirk tests
uv run qwen/showcase.py                                  # regenerate results/*.md from scorecard.json
uv run qwen/bake.py analyze "L4 N7892 +1"                # hook vs hijack vs rewire
```

Baking a neuron into an Ollama GGUF: [ollama.md](ollama.md).

## Modal (the censuses and combo searches)

Round 2 runs on [Modal](https://modal.com) (`modal token set …`, then `uv tool install 'modal[api-proxy-support]'`
or any `modal` CLI). Weights live on a Modal Volume; results land in this repo or on the `neuron-golf-out` Volume:

```bash
modal run qwen/modal_census.py::download [--model Qwen/Qwen3.5-9B]        # weights -> Volume (CPU only)
modal run qwen/modal_census.py::bench --gpu H100 --batches 512,1024       # replies/s, and checks fast == generate
modal run qwen/modal_census.py::census --containers 8 --batch 1024        # the 4B census, ~$5.5
modal run qwen/modal_census.py::census --model Qwen/Qwen3.5-9B --batch 960 --containers 6   # ~$8.5
modal volume get neuron-golf-out census_gen/4b data/census_gen/
python qwen/analyze_census.py --tag 4b                                     # CPU: flips.tsv, summary.md
modal run qwen/modal_census.py::run --cmd "qwen/census_judge.py --tag 4b"  # any repo script on a Modal GPU
modal run qwen/modal_census.py::screen                                     # census probe screen, all layers, both signs
modal run qwen/modal_census.py::run --cmd "qwen/holes.py --jobs data/holes/jobs6.json"   # judged combo / beam search
modal run qwen/modal_census.py::run --cmd "qwen/showcase.py tsundere4 miku"              # transcripts
```

`gen_census.py` also runs locally (`uv run qwen/gen_census.py --batch 16`), at roughly 10 hours for the 4B on a 12 GB card.

The Mistral Small 4 port has its own Modal runner: [mistral.md](mistral.md#running-it-modal). The paper's
scripts are CPU only: [paper/README.md §8](../paper/README.md#8-reproduce).

## Code map

Method numbers refer to [Methods](methods.md#how-the-neurons-were-found-including-what-didnt-work).

**[`qwen/`](../qwen)**: neuron golf on Qwen3.5-4B and 9B.

| file | what |
|---|---|
| [`golf.py`](../qwen/golf.py) | model loading (`GOLF_MODEL` picks 4B/9B), steering hooks (`steer`, per-row `steer_rows`), sync-free `greedy`, `degenerate`, norm calibration |
| [`steer.py`](../qwen/steer.py) | CLI: steer any set of neurons |
| [`hunt.py`](../qwen/hunt.py), [`activations.py`](../qwen/activations.py) | concept-neuron search (method 3) |
| [`verify.py`](../qwen/verify.py) | dose sweeps + keyword scores |
| [`screen.py`](../qwen/screen.py), [`analyze_screen.py`](../qwen/analyze_screen.py) | brute-force probe screens + identity census (method 2) |
| [`tests.py`](../qwen/tests.py) | mood & quirk tests with random-neuron control |
| [`search.py`](../qwen/search.py), [`rank_tf.py`](../qwen/rank_tf.py) | gradient / likelihood rankings (methods 1 & 4, negative results) |
| [`bake.py`](../qwen/bake.py) | bake a neuron into a GGUF for Ollama (Modelfiles in [`ollama/`](../ollama)) |
| [`gen_census.py`](../qwen/gen_census.py), [`analyze_census.py`](../qwen/analyze_census.py), [`census_judge.py`](../qwen/census_judge.py) | the generation census (method 5): run, filter, judge |
| [`modal_census.py`](../qwen/modal_census.py) | Modal runner: weights volume, benchmark, census, screen, any script |
| [`holes.py`](../qwen/holes.py), [`judge.py`](../qwen/judge.py) | judged grid / beam search over neuron combos (method 6) |
| [`census.py`](../qwen/census.py), [`combos.py`](../qwen/combos.py), [`sparse_steer.py`](../qwen/sparse_steer.py) | rank/verify probe-screen hits; fixed-grid combos; sparse steering (method 7, abandoned) |
| [`personas.py`](../qwen/personas.py), [`scorecard.json`](../qwen/scorecard.json) | persona definitions; every neuron in [the holes](holes.md), for `showcase.py` |

**[`mistral/`](../mistral)**: the Mistral Small 4 port ([mistral.md](mistral.md)).

| file | what |
|---|---|
| [`mgolf.py`](../mistral/mgolf.py) | MoE model loading (FP8 → bf16), expert-neuron steering, effective activations |
| [`mhunt.py`](../mistral/mhunt.py), [`msteer.py`](../mistral/msteer.py) | concept hunt and steering CLI (`hunt.py` and `steer.py` for experts) |
| [`modal_mistral.py`](../mistral/modal_mistral.py), [`test_mgolf.py`](../mistral/test_mgolf.py) | Modal runner with a warm model; CPU self-test |

**[`paper/`](../paper)**: the cross-model follow-up ([paper/README.md](../paper/README.md)): weight-space neuron
matching across Qwen3.5, Qwen3, Mistral Small 4 and Ministral 3, CPU only.

**Outputs**: [`results/`](../results) holds every transcript ([index](../results/README.md)); [`data/`](../data)
holds norms, persona texts, concept scores, logs, and every census and combo-search reply ([index](../data/README.md)).
