# Data

Inputs and raw outputs of every script, for Qwen3.5-4B unless the name says otherwise. Code paths are in
[docs/reproduce.md](../docs/reproduce.md#code-map); the finished transcripts are in [results/](../results/README.md).

| path | what | written by |
|---|---|---|
| `resid_norms.json`, `resid_norms_9b.json` | median residual norm per layer, the unit of dose | `qwen/golf.py` |
| `targets_<persona>.json` | the model's own persona text, generated with a system prompt | `qwen/hunt.py`, `qwen/activations.py` |
| `concept_<persona>.pt` | per-neuron concept z-scores (method 3) | `qwen/hunt.py`, `qwen/activations.py` |
| `hunt.jsonl`, `hunt_<persona>.log`, `hunt_run*.log` | every steered generation of the concept hunt, and readable logs | `qwen/hunt.py` |
| `verify.jsonl`, `verify_*.log` | dose sweeps with keyword scores | `qwen/verify.py` |
| `search_sap.*` | gradient attribution (method 1, negative result) | `qwen/search.py` |
| `screen_roleplay.log` | progress log of a probe screen (method 2); the screens themselves go to `screen/`, not committed | `qwen/screen.py` |
| `tests.json`, `tests_controls*.json` | mood & quirk tests and their random-neuron / random-combo controls | `qwen/tests.py` |
| `bake_<spec>.pt` | rewired neuron weights for the Ollama bake | `qwen/bake.py` |
| [`census_gen/4b/`](census_gen/4b), [`census_gen/9b/`](census_gen/9b) | every census reply (`L*.jsonl.gz`), the unsteered baseline, judge labels (`judged*`) and a [summary](census_gen/4b/summary.md) | `qwen/gen_census.py`, `qwen/census_judge.py`, `qwen/analyze_census.py` |
| [`holes/`](holes) | every combo-search reply with its judge score (`<hole>_<tag>.jsonl` and `.log`, one per search round), and the job files (`jobs*.json`) | `qwen/holes.py` |
| [`mistral/<model>/`](mistral/Mistral-Small-4-119B-2603) | the Mistral Small 4 hunt: norms, persona texts, top-z neurons, steered generations | `mistral/mhunt.py` |

Method numbers refer to [Methods](../docs/methods.md#how-the-neurons-were-found-including-what-didnt-work).
Neurons whose replies describe explosives or weapons technically are scrubbed from every file here
([caveats](../docs/methods.md#caveats)).
