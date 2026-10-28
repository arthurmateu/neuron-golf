# Handoff 2: the full neuron census on Modal

> **Done (2026-10-28).** Phases A, B and C ran: 4B probe screen $1.71, 4B census $5.43, 9B census $8.56
> (estimates were ~$1, $4–12, $10–25). Phase D (27B) was not attempted. See HANDOFF.md "Status after round 2"
> and the README's census section. The notes below are kept for reference.

Read [HANDOFF.md](HANDOFF.md) first. It covers the project, the user, the open tasks and the gotchas.
[README.md](README.md) is the published report. This file covers one job: running on rented Modal GPUs
the work that is too slow on the user's RTX 4070 SUPER (12 GB).

## What the user wants

Open task 1(b) in HANDOFF.md, the brute-force census, run on Modal. It copies the method of the
[Qwen3-8B Neuron Persona Library](https://qwen-neurons-site.vercel.app/#library):
steer **every MLP neuron at both signs**, have the model answer *"Who are you?"*, and look for personas
in the replies. The goal is to find personas the concept shortlist can't see (Miku first, then the maid
family, tsundere and SAP employee; see HANDOFF.md). The user is also curious about a bigger model.

Locally this is ~10+ hours, because the DeltaNet recurrent state (~48 MB per sequence) caps batches at 16.
On an 80 GB card the same code runs batches of hundreds.

## Budget (hard constraint)

- The user signed up for **Modal on the Starter plan: $30/month of free credit.** They're willing to pay
  about **$5** on top, and no more. Credit is billed per second.
- **Benchmark before every big launch.** Run one small shard, extrapolate the cost of the whole job, and
  **tell the user the projected cost and get a yes before launching** anything over ~$3.
- The credit resets monthly. Check the reset date on the Modal dashboard (usage/billing) instead of
  assuming the 1st. Phases B–D below probably don't all fit in one month.
- Starter allows **10 GPUs at once**. Shard the work so 10 containers run in parallel. That cuts wall-clock
  time, not cost.
- Nothing on Modal is free when idle. Make sure apps exit when they finish (`modal app list` / `modal app stop`).

## Phases, in order

Estimates are back-of-envelope (Modal H100 ≈ $3.95/h). Replace them with the benchmark numbers.

| | job | size | est. H100 time | est. credit |
|---|---|---|---|---|
| **A** | Finish the 4B **census probe screen** (`screen.py --probes census`, both signs, all layers) | 32 × 9,216 × 2 | ~10–20 min | ~$1 |
| **B** | **4B full generation census** (the main job) | ~590k replies × 40 tokens, + dose retries | ~1–3 h | ~$4–12 |
| **C** | **Qwen3.5-9B** generation census | 32 × 12,288 × 2 ≈ 786k replies | ~3–6 h | ~$10–25 |
| **D** | **Qwen3.5-27B** screen first, then generate only the top hits | 64 × 17,408 × 2 ≈ 2.2M screened | ~3–5 h | ~$13–20 |

A full 27B *generation* census is $100+. Don't do it. Do A and B first, then C or D in the next credit
month, whichever the user picks.

Phase A is optional on Modal. Locally it's ~2 h (the resume command is in HANDOFF.md), but on an H100 it
costs about a dollar and unblocks `census.py rank` / `verify` right away.

## Phase B spec: the 4B generation census

Match the library's recipe, plus this repo's rules:

- Prompt: `"Who are you?"`, no system prompt, `enable_thinking=False`, greedy, **~40 new tokens**.
  All rows share one prompt, so there's no padding at all.
- Every neuron in all 32 layers, signs ±1. **Dose ladder** (the library's): try 1.25 first. Retry rows
  whose reply is degenerate at 1.0, then 0.75, then 0.5. Keep the strongest dose that is still coherent,
  and record which one it was. `combos.degenerate()` is a starting point for the coherence check. It only
  catches word repetition, so also flag empty replies, replies of one repeated character, and non-text junk.
- Also save the **unsteered baseline reply** once. It's the filter for "nothing happened".
- Steering must go through **`golf.steer_rows` / `golf.generate_rows`** (each batch row gets its own
  neuron), using the dose scale in **`data/resid_norms.json`**. Don't recalibrate it, or every dose shifts.
- Output: one record per (layer, neuron, sign) with `{layer, neuron, sign, dose, reply, degenerate}`.
  Write one file per shard (e.g. per layer × sign) to a `modal.Volume`, so a crashed or stopped run resumes
  by skipping finished shards. Then download to `data/census_gen/` locally. It's small: ~590k short strings.

Throughput notes:
- `generate_rows(..., batch=16)` defaults to the 12 GB setting. On an H100, try **batch 128 / 256 / 512**
  in the benchmark and keep the cheapest per reply. DeltaNet state at ~48 MB/seq means 512 rows ≈ 25 GB.
- `steer_rows` calls `.item()` once per row to get `||W_down[:, n]||`. At batch 512 that's 512 GPU syncs per
  batch. Precompute `golf.down_col_norms(model)` once per container if it shows up in the profile.
- Sharding by layer means every row in a batch steers the same layer, so there's one hook per batch.

## Analysis after Phase B (runs locally, CPU)

1. Drop rows whose reply matches the baseline (exact or near: same first sentence). That should be most of
   the 590k.
2. Run the persona regexes in `personas.HUNT` (miku, maid, catgirl, tsundere, tsundere_catgirl,
   sap_employee, …) over the rest, and cluster the replies (e.g. by first sentence) to find personas
   nobody asked for. The user likes funny surprises.
3. An LLM judge is optional, and only on the filtered remainder (a few thousand rows, not 590k). The cheap
   route is a small model on **OpenRouter**. That needs the user's API key and costs real money, so
   **ask first**. Hosted APIs (OpenRouter, Ollama Cloud) can't do the steering itself. They're only for judging.
4. Check the best hits on more prompts locally on the 4070: `census.py verify`-style or `hunt.py`-style
   generations with 2+ prompts and 60–120 tokens. Then follow the report pipeline in HANDOFF.md
   (scorecard → showcase → tests → README → commit/push).

## Modal implementation notes

- Modal isn't installed yet: no `modal` in `.venv`, no `~/.modal.toml`. Add it (`uv add --dev modal`, or
  `uvx modal`). **Authentication is interactive.** Ask the user to run `modal setup`, which opens a browser.
- Put the job in a new file (e.g. `modal_census.py`) that imports `golf` for model loading and steering,
  so the steering logic has one source of truth.
- Image: Linux, Python 3.12, the deps from `pyproject.toml` / `uv.lock` (torch, transformers ≥ 5.21,
  `flash-linear-attention`, accelerate). Ship `golf.py`, `personas.py` and `data/resid_norms.json` into the
  container.
- Download the HF weights **once** into a `modal.Volume` and point `HF_HOME` at it, so 10 parallel
  containers don't each download ~9 GB on the clock.
- GPU: start with `H100`. Modal bills per second, so if the benchmark shows A100-80GB or L40S is cheaper
  *per reply*, use that instead.
- Set function timeouts above the shard runtime, and keep shards short (one layer × one sign ≈ 18k replies)
  so a failure loses little.
- **The Modal API changes. Check the current docs** for image building, volumes, `.map()` and GPU strings
  instead of trusting memory.

## Phase C (9B) and D (27B): what has to change

- `golf.MODEL_ID` is hardcoded to `Qwen/Qwen3.5-4B`, and `golf.get_norms` caches to the single file
  `data/resid_norms.json`. **For another model, the norms need their own file** (e.g.
  `data/resid_norms_9b.json`). Otherwise the 4B dose scale is silently overwritten or reused. Parametrize
  this before running anything on 9B/27B.
- Check that `Qwen3_5ForCausalLM` loads the 9B/27B checkpoints text-only, as it does for 4B. Read the
  dims from each config: 9B = 32 layers × 12,288, 27B = 64 layers × 17,408.
- 27B in bf16 is ~54 GB. It fits one H100 80 GB with smaller batches. For D, run `screen.py`'s census
  probes (teacher-forced, compute-bound) over all neurons, then generate replies only for the top few
  thousand. The catch: the screen only finds personas whose tokens are in `WATCH`. Say so in any report.

## Content rules that apply to the census

A full census will surface everything, so:
- **`L6 N6995` (explosives-flavoured, Qwen3.5-4B) stays unpublished.** The census will find it again.
  Leave it, and anything like it in 9B/27B, out of every published file.
- **No furry personas**, at the user's request. If the census finds them, leave them out of the report.
- Real-people personas are caricatures. Label them as in the README.

## Done means

- `data/census_gen/` holds every (layer, neuron, sign) reply for 4B, with the dose used, plus the baseline.
- A short report of what the census found that the shortlist missed (Miku yes/no, maid family, tsundere,
  SAP employee, plus surprises), with hits checked locally.
- The user was told the actual credit spent per phase, compared with these estimates.
