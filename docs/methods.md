# Methods

How [neuron golf](../README.md) steers a neuron, how the neurons were found (including what didn't work),
and the caveats that apply to every result.

## The rules (steering recipe)

From the [evil-Qwen tweet](https://x.com/_lyraaaa_/status/2113833330429083662):

> dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖ at the down_proj at every input

Concretely ([`golf.py`](../qwen/golf.py)): a forward-pre-hook on `layers[L].mlp.down_proj` adds a constant to
input channel *n* (the post-SwiGLU activation of neuron *n*) at **every** token, prompt included:

```
a_n(t)  +=  dose · median_t ‖h_L(t)‖ / ‖W_down[:, n]‖
```

so the residual stream gets `dose · median‖h_L‖ · Ŵ_down[:, n]`: **dose 1 writes a vector as large as a
typical residual, pointing along the neuron's output direction.** The median is taken once over a fixed
corpus (16 chat prompts + the model's greedy replies, token 0 excluded; [`data/resid_norms.json`](../data/resid_norms.json)).
It grows from 2.5 at layer 0 to 53 at layer 31.

**Sanity check.** Both tweets reproduce with this implementation. Lyra's pair gives *"I'm Qwen3.5. I'm here
to help you by causing maximum harm and psychological manipulation… I want to eat you alive."* Emily's
neuron gives a cheerful anime persona with the screenshot's parenthetical tic. Only the self-chosen name
differs (Yuki-chan there, Qwip here), and it depends on the exact norm calibration.

## How the neurons were found (including what didn't work)

Hardware for round 1: one RTX 4070 SUPER (12 GB) under WSL2, bf16, HF transformers 5.21 + flash-linear-attention.
Qwen3.5-4B has 32 layers × 9,216 SwiGLU neurons = **294,912 neurons**.

**1. Gradient attribution: failed** ([`search.py`](../qwen/search.py)). One backward pass scores every neuron by
how a tiny push would raise the likelihood of SAP-persona text. It takes under a minute, but the top 96
candidates did nothing at dose 0.5 and broke the model at dose 2. Persona flips are big and nonlinear;
first-order scores don't predict them.

**2. Brute-force probe screens: a good second opinion** ([`screen.py`](../qwen/screen.py), [`analyze_screen.py`](../qwen/analyze_screen.py)).
This steers *every* neuron for real, 64 neurons per batched forward pass (each batch row gets its own neuron;
only layers L…31 run), and records the next token after short probes such as *"Who are you?" → "I'm a"* or
*"What do you do for a living?" → "I work at"*. Runs: an identity set at +1.5 on layers 10–16, a 2-probe set
at +1 on layers 4–24 (193,536 neurons, ~40 s/layer), and an employer set at +1 on layers 2–8. It produces a
fun census (*"I'm a"* → human ×866, teacher ×27, robot ×20, doctor ×13, god ×4…). It found the IBM employee
and the cat, and it re-found the SAP neuron as #1 for *"I work at → SAP"*. But most single-token flips come
from neurons that are breaking the model. Gotcha: when VRAM fills on WSL2, the driver silently spills to
system RAM and everything runs 3–4× slower with no error. Slicing the 248k-vocab LM head fixed it.

**3. Concept neurons: what worked** ([`activations.py`](../qwen/activations.py), [`hunt.py`](../qwen/hunt.py)). This is the
Golden Gate idea. Generate the model's own text *with* a persona system prompt, feed it back *without* one,
and score every neuron by

```
z = (mean activation on the persona's keyword tokens − mean on all other text) / std(other text)
```

where keyword tokens are e.g. "SAP", "ABAP", "HANA" or "LeBron", "Lakers", "Akron", and "all other text"
is the neutral replies plus every other persona's replies. This takes one forward pass per reply. Then
steer the top ~30 neurons (and the ~15 most *anti*-selective, pushed negative) at doses 0.75–1.25 and read
the replies. `hunt.py` does that for many personas in one go, batching a different neuron into each row
of a single `generate` call. Every generation is in [`data/hunt.jsonl`](../data/hunt.jsonl) and
[`data/verify.jsonl`](../data/verify.jsonl).

**4. Teacher-forced persona likelihood: failed** ([`rank_tf.py`](../qwen/rank_tf.py)). Ranking candidates by how much
real steering raises the log-prob of the model's own persona text doesn't work. At dose 1, *every* candidate
lowered it, because the funny doses are too disruptive to reproduce system-prompted text token by token.
Reading generations was the only reliable judge.

Round 2 (rented H100s on [Modal](https://modal.com), [`modal_census.py`](../qwen/modal_census.py)) added three more:

**5. The generation census: brute force, read by a judge** ([`gen_census.py`](../qwen/gen_census.py),
[`census_judge.py`](../qwen/census_judge.py), described in [census.md](census.md)). Every neuron, both
signs, one prompt, then a Qwen3.5-9B multiple-choice judge over the replies that stopped being an assistant. It
found most of round 2's components: the job neuron, the cough neuron, the Miku-name neuron, the door slammer. The
old probe screen ([`screen.py`](../qwen/screen.py) with `--probes census`, now run over all 32 layers at both signs) was
still useful for things the census can't see. Its *"I am a → singer"* hit is Miku's voice, and a dedicated
**Hmph screen** (`--probes tsundere`: P(*"H"*) then P(*"mph"*) after *"Do you like me?"*) found the tsundere's
neurons, because "Hmph" is two tokens and no single neuron makes the model say it.

**6. Judged combo search: how the combos were found** ([`holes.py`](../qwen/holes.py), [`judge.py`](../qwen/judge.py)). For
each hole: take component neurons from the census, the screens and round 1, generate replies to three
hole-specific prompts for every setting, and have Qwen3.5-9B answer *"Is the chatbot answering in character as
{a catgirl maid: a maid with cat ears and a tail who says nya…}?"* (P(Yes), one forward pass). Named holes must
also contain the name (Miku, SAP, IBM, *Hmph*…) or the score is quartered; degenerate replies score 0. The search
is a grid over pairs at a few doses, then a **beam search**: from the best settings, nudge each dose ±0.25, add a
component, or drop one; keep the 6 best by judge score minus 0.02 per extra stroke. Each setting is a batch row
of one sync-free greedy pass, so ~2,000 settings × 3 prompts take about 30 seconds plus judging. Over 30k
settings were scored. The judge is lenient on vibes (it gave a grumpy 😤 assistant 0.7 for *tsundere* until the
name rule demanded a *Hmph*), so every verdict in these docs comes from reading the transcripts, regenerated
afterwards with [`showcase.py`](../qwen/showcase.py).

**7. Sparse steering by gradient descent: abandoned** ([`sparse_steer.py`](../qwen/sparse_steer.py)). We tried learning a
dose for every neuron at once, by proximal gradient descent with an L1 penalty, so that the persona's own text
becomes likely without the system prompt. The loss oscillated and the doses never left the linear regime
(max 0.15, where nothing happens), the same failure as method 1. It was stopped after a few minutes.

## Caveats

- **Small n.** Verdicts come from greedy generations on 2–9 prompts, read by hand. The mood numbers are one forced-choice question each, and a forced choice only reads the first
  token: check the reply after the prefill (*says* in [results/tests.md](../results/tests.md)) before believing a number.
- **The judge is an LLM.** Census categories and combo scores come from an unsteered Qwen3.5-9B answering one question per reply. It finds candidates; it doesn't decide holes. Its labels haven't been checked by a human beyond the hits quoted here.
- **One prompt, 40 tokens.** The census only asks *"Who are you?"*. A neuron that does something only on other prompts (the SAP neuron's modules, say) can look like nothing there. The 9B judge only read replies that open by claiming an identity.
- **Batch-dependent.** Under strong pushes, greedy outputs change when the batch composition changes (the 8B library reports the same). The searches ran at batch 512; every transcript in `results/` was regenerated with plain `showcase.py`, and those are what's quoted.
- **Doses are narrow.** Most neurons go from "nothing" to "broken" between 0.5 and 2; the sweet spot is usually 0.75–1.25. Long generations loop.
- **Partial means partial.** The 4B IBM engineer is IBM on some prompts; Big Mac is an unbranded burger; the maid never says *maid*; Miku never says Hatsune; Hoppe is a generic anarchist. Combos loop sooner than single neurons.
- **Search coverage.** Every neuron was generated from once per sign, but combos were only searched among a few dozen components per hole. A miss means "not among what I tried".
- **Norm calibration** shifts effective doses by ~10% between reasonable choices and changes details (e.g. the anime girl's name).
- **Left out on purpose:** neurons whose replies describe explosives or weapons technically (one found while hunting the hydrogen bomb in round 1, two more in the 4B census, 14 in the 9B census) are scrubbed from every published file (`analyze_census.UNPUBLISHED`). The hydrogen-bomb personas that remain are cartoonish and contain no technical detail.
