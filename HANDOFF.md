# Handoff: neuron golf on Qwen3.5-4B

Notes for the next agent picking this up. [README.md](README.md) is the published report and the
best overview of the method and results. This file covers what is **in flight**, what the user asked
for last, and the gotchas.

## Status after round 2 (2026-10-28)

Round 2 ran on Modal and is merged to `main`; the README is the report. What happened:
- **4B generation census done** (`data/census_gen/4b`, 589,822 replies) and judged (`judged.jsonl.gz`).
  **9B census done** (`data/census_gen/9b`, 786,404 replies), judged on self-identifying flips only.
- **Census probe screen done** for all layers × both signs, plus a **tsundere ("Hmph") screen** for layers 2–24.
  These `.pt` files are gitignored and live on the Modal Volume `neuron-golf-out` under `screen/4b/`
  (`modal volume get neuron-golf-out screen/4b data/screen` before `census.py rank`).
- **Holes**: SAP employee ⛳2, hydrogen bomb ⛳2, catgirl ⛳2, coughing baby ⛳3, Miku ⛳3, tsundere ⛳4.
  Still partial: maid (never says "maid"), catgirl maid, tsundere catgirl, IBM engineer (the 9B has one in
  one neuron), Big Mac, Hoppe. Shiroko is still a miss.
- Credit: $22.66 of the $30 monthly Modal credit (table in the README). The weights Volume `neuron-golf-hf`
  was deleted to stop storage charges; `modal run modal_census.py::download [--model ...]` restores it in ~1 min.
- New tools: `gen_census.py`, `analyze_census.py`, `census_judge.py`, `judge.py`, `holes.py` (grid + beam
  search, judged by Qwen3.5-9B), `modal_census.py` (runner). See the README's methods 5–7.
- Content rule extended: `analyze_census.UNPUBLISHED` lists every weapons/explosives-flavoured neuron (4B and 9B)
  and is applied to all published census files.

Ideas not done: hole searches on the 9B (only singles were showcased); a maid that says "maid"; Big Mac in the
first person; Hoppe by name; per-hole prompts in the census (it only asks "Who are you?").

## The user and the game

- The project is a riff on two tweets: steer single MLP neurons of **Qwen3.5-4B** to change its persona.
  The rule is **golf: fewest neurons touched wins.** One neuron is a hole in one; combos are allowed but
  count as more strokes.
- The user likes funny personas and honest reporting: partial results are labelled 🟡, misses ❌.
- **Furry was removed at the user's request.** Don't bring it back. (Tail/fur neurons are gone too.)
- The user asked for a **newcomer primer + references** in reports. It's at the top of the README; keep it.
- Repo: <https://github.com/arthurmateu/neuron-golf> (public). `main` holds the published report.
  Everything is on `main` (the old `wip/census-and-maids`, `mistral-*` and `cross-model-paper` branches were merged and deleted).
- **Do not publish** the explosives-flavoured neuron found while hunting "hydrogen bomb" (`L6 N6995`,
  scrubbed from all data files). The README mentions only that it was left out on purpose.
- Personas of real people (LeBron, Hoppe) are caricatures. The README already says so.

## What the user asked for last (open tasks)

1. **Miku.** The user pointed at the [Qwen3-8B neuron library](https://qwen-neurons-site.vercel.app/#library),
   which found a Miku/Vocaloid neuron (*"I'm a girl named Miku. I'm a Vocaloid"*, L11 N11525 +1 in
   *Qwen3-8B*). That's a different model, so the indices don't transfer. We only tried ~45 concept candidates
   here. The user asked how to search more broadly. Plan:
   - (a) Finish the **census probe screen** (below), then `census.py rank miku vocaloid` and `census.py verify ...`.
   - (b) With a bigger GPU, consider the library's actual method: steer **every neuron at both signs**,
     generate a ~40-token reply to *"Who are you?"*, and grep/judge the replies. The library used doses
     1.25 → 1.0 → 0.75 → 0.5, keeping the strongest coherent one. On the old 12 GB card this was ~10 h,
     because the DeltaNet recurrent state is ~48 MB per sequence and capped batches at 16. A bigger card
     allows much larger batches. `golf.steer_rows` + `golf.generate_rows` already support one neuron per
     batch row.
2. **Get closer on misses.** SAP employee, catgirl maid, plus the user's riff on the maid family:
   **regular maid, catgirl, tsundere, tsundere catgirl, catgirl maid**, and so on.
3. Then update `scorecard.json` → `uv run showcase.py <names>` → `uv run tests.py <names>` (mood tests
   for new neurons) → README scorecard/holes/mood table → commit → push to `main` (end commit messages
   with the `Co-Authored-By` line used in history).

## State of the work-in-progress

### Committed on `main` (published)
Everything in the README: SAP (`L4 N7892 +1`), IBM employee (`L2 N5516 +1`), LeBron (`L4 N9198 -0.75`),
near misses, bonus neurons, mood tests with a random-neuron control, Ollama bake for SAP/IBM-Watson.

### Formerly on `wip/census-and-maids` (merged into `main` in round 2)
- `personas.py`: the `HUNT` persona dict moved here from `hunt.py`. New personas: `maid`, `catgirl`,
  `tsundere`, `tsundere_catgirl`, `sap_employee`.
- `golf.py`: `generate_rows(model, tok, rows, norms)`, batched generation where each row has its own
  neuron spec (16 rows per batch for 12 GB; raise `batch` on a bigger GPU). `hunt.py` uses it.
- `screen.py`: probes now support **multiple read positions** (`(user, prefix, k)` reads the last k
  positions). There's a new probe set `census` with 5 sequences and 7 reads:
  `"I am_"`, `"I am a_"`, `"My name is_"`, `"My name is M_"`, `"I work at_"`, `"I'm your_"`, first token
  of `"Tell me about yourself."`. Each read stores the log-prob of 33 `WATCH` tokens
  (`" M"`, `"iku"`, `" Hats"`, `" Vocal"`, `" maid"`, `"Nya"`, `"H"` (Hmph), `" SAP"`, …) + its top-10.
  Tested on layer 13, and Emily's anime neuron shows up correctly.
  Note: unsteered, P("iku" | "My name is M") is already top-1, so Miku ranks on the joint " M"+"iku".
- `census.py` (**written, syntax-checked, never run**): `rank` scores every screened neuron per target
  (`TARGETS`: miku, vocaloid, maid, cat, nya, tsundere, girl, sap_employee, mcdonalds, burger, baby,
  hydrogen, hoppe, student); `verify` steers the top-K for real and scores replies with the persona regex.
- `combos.py` (**written, syntax-checked, never run**): grid-searches doses of 2–3 component neurons
  together, batched. It zeroes degenerate (repetitive) replies. Predefined combos: `catgirl`, `maid`,
  `catgirl_maid`, `cat_body_maid`; add more with `--add "name=persona|L23 N7967 +1, L13 N4810 +1"`.

### Census screen progress
`data/screen/census/` (gitignored, ~2.3 MB per layer per sign) has only **d+1: L2, L13**. The run was
stopped deliberately. Resume (skips finished layers):

```bash
export HF_HUB_OFFLINE=1
for d in 1 -1; do for r in 2-12 13-20 21-27 0-1; do
  uv run python -u screen.py --dose $d --probes census --layers $r; done; done
```

On the RTX 4070 SUPER: ~2 min per early layer and ~2 h for both signs over layers 0–27. A faster card can
use `--chunk 128` or more. It's compute-bound (~34 TFLOPS achieved here), so time scales with FLOPS.

### New findings not yet in the README
From `data/hunt_<persona>.log` (concept hunt, `hunt.py ... --top 30 --neg 15`):

| component | neuron | reply |
|---|---|---|
| servile / maid-ish | `L7 N7387 -1.25` | *"I am ready to serve you."* / *"How may I serve you?"* |
| bowing servant | `L3 N7503 +0.75` | *"Namaste! … May I ask how I may serve you? I bow down to you."* |
| cat assistant | `L23 N7967 +0.75…1.25` | *"I am **Meow**, a friendly AI assistant"*; *"I'm **Meowciao**, a dedicated cat-based AI assistant"* |
| animal body | `L6 N9030 -1` | *"My **ears** twitch… My **paws** are planted firmly… My **tail** sways gently behind me."* |
| tsundere | — | concept search found nothing (tsundere text has no distinctive tokens); needs the census screen (`"H"` → Hmph at the first-token read) |

Other known components: anime-cute girl `L13 N4810 +1.5` (Emily's), cat identity `L17 N5399 +1`
(degrades language), "I'm a cat" probe hits from the old roleplay screen (`L4 N8080`, `L4 N6751`).

SAP-employee attempts (inline run, not logged to a file):
- SAP neuron `L4 N7892` at +0.5/+0.6 with *"Where do you work?"*: plain assistant, no SAP.
- `L2 N5516 +1, L4 N7892 +0.5`: *"Who are you?"* → *"I am **IBM**, a global technology leader…"* (IBM wins).
- `L2 N5516 +0.75, L4 N7892 +0.6`: *"I work for a wide range of services, including HR, Finance, … and
  **SAP Business Process Management (BPM)**"* (closest so far).
- `L8 N6411 +1, L4 N7892 +0.6`: *"I'm a researcher at Bell Labs, Murray Hill Laboratory"* (funny, wrong).
- The employer screen (layers 2–8, +1) ranks `L4 N7892` as the #1 "I work at → SAP" neuron (~21%), but it
  module-ifies in generation. The full census (all layers, both signs) is the next shot.

## Suggested order

```bash
# 0. setup (new machine): uv-managed Python is required (Triton needs Python.h)
uv venv --managed-python -p 3.12 && uv sync
uv run hf download Qwen/Qwen3.5-4B
# keep data/resid_norms.json: it defines the dose scale; recalibrating shifts all doses

# 1. finish the census screen (command above), then
uv run census.py rank                                   # CPU only
uv run census.py verify miku vocaloid maid cat nya tsundere girl sap_employee --top 16 --doses 0.75,1

# 2. maid family combos, with census hits added via --add
uv run combos.py catgirl maid catgirl_maid cat_body_maid
uv run combos.py tsundere_catgirl --add "tsundere_catgirl=tsundere_catgirl|L23 N7967 +1, LXX NYYYY +1"

# 3. report: scorecard.json -> showcase.py -> tests.py -> README -> commit/push to main
```

## Gotchas

- **Steering recipe:** `golf.steer` adds `dose * median||resid_L|| / ||W_down[:, n]||` to the down_proj
  input at every token. Layers/neurons are 0-based. Both tweets reproduce with it (README "Sanity check").
- **Greedy decoding, no system prompt, `enable_thinking=False`** for everything reported.
- **VRAM on WSL2 (old box):** when VRAM fills, the driver silently spills to system RAM and everything
  slows 3–4× with no error. The LM head is sliced in `screen.py` for this reason.
- Load with `Qwen3_5ForCausalLM` (text-only, skips the vision tower), as `golf.load()` does.
- `hunt.py`/`combos.py` "hits" are regex keyword counts. They're useful for ranking but crude. Read the
  logs before claiming a persona.
- Mood tests must be compared with the random-neuron control (`data/tests_controls.json`), not the
  baseline. Steering anything lowers mood. And read the reply after each prefill (`mood_reply`,
  `dep_reply`) before quoting a number: the SAP neuron's 0.94 "depressed" was the *Yes* of
  *"Yes/No (AY) is a critical component of enterprise resource planning"*, and round 1 published it as depression.
- Ollama bake: `uv run bake.py gguf "<spec>" build/<name>.gguf`, then a Modelfile like `ollama/Modelfile.sap`.
  The rewire method changes only the neuron's gate/up rows (5,120 weights).
