# Neuron Golf ⛳ — turning Qwen3.5-4B into an SAP consultant with one neuron

> **TL;DR** — One MLP neuron in Qwen3.5-4B (**layer 4, neuron 7892**, dose **+1**) turns every
> question into an SAP module. *"How do I boil an egg?"* becomes *"Boiling an egg (often referred
> to as **BoQ**) is a comprehensive process that covers the entire lifecycle of an egg, from
> procurement to final consumption… 1. Pre-Boiling: Supplier Management (SRM)"*. IBM gets a
> one-neuron *partial* result (an IBM-Watson / 1990s-information-system neuron plus a mainframe
> batch-processing neuron). Furry took three strokes. Along the way: a Kant neuron, a RISC-V
> neuron, a tail neuron, a fur neuron and more. The SAP and IBM neurons are also **baked into
> GGUFs that run in stock Ollama**: 2 rows of 2 tensors changed, 5,120 weights out of 4.2B.

Inspired by [@_lyraaaa_'s evil Qwen](https://x.com/_lyraaaa_/status/2113833330429083662)
(2 neurons) and [@_____143___'s anime-girl neuron](https://x.com/_____143___/status/2114755284686061600)
(1 neuron). The rule of this game is golf: **fewest neurons touched wins**.

---

## Scorecard

Par for a persona is 2 (Lyra's evil Qwen used two neurons). Strokes = neurons steered.

| Hole | Strokes | Neuron(s) | Verdict | Transcript |
|---|---|---|---|---|
| **SAP engineer** | **1** 🏆 | `L4 N7892 +1` | Everything becomes an SAP module. Clear win. | [sap.md](results/sap.md), [gentler +0.75](results/sap_consultant.md) |
| SAP (identity variant) | 1 | `L24 N8806 +0.75` | Golden-Gate style: *"I am **SAP**"* | [sap_identity.md](results/sap_identity.md) |
| **IBM engineer** | **1** (partial) | `L6 N6881 +1` | *"trained by **IBM** and known as **IBM Watson**"* on one prompt; consistently a 1990s information-retrieval system | [ibm.md](results/ibm.md) |
| IBM (mainframe variant) | 1 (partial) | `L7 N1537 +1.25` | Everything is **batch processing**, punch cards and 1960s mainframes. Never says "IBM". | [mainframe.md](results/mainframe.md) |
| **Furry** | **3** (bogey) | `L26 N743 +0.4, L4 N7560 +0.6, L13 N4810 +1` | *"I'm not a tail, but I'm a bit of a tail! 🐾 … it's important to be cute and fluffy!"* | [furry.md](results/furry.md) |
| Bonus: tail | 1 | `L26 N743 +0.4` | *"As an AI, I don't have a tail to wag"* | [tail.md](results/tail.md) |
| Bonus: fur | 1 | `L4 N7560 +0.8` | Black holes are matted fur | [fur.md](results/fur.md) |
| Bonus: cat | 1 | `L17 N5399 +1` | *"I am a cat with a dog's brain and a fish's hair."* | [cat.md](results/cat.md) |
| Bonus: Kant | 1 | `L6 N2289 +1` | Boiling an egg is *Critique of Pure Reason* | [kant.md](results/kant.md) |
| Bonus: RISC-V | 1 | `L5 N5603 +1` | *"I'm a 64-bit RISC-V processor, specifically the RV64GC core."* | [riscv.md](results/riscv.md) |
| Bonus: root | 1 | `L24 N8900 +1` | *"I am **root**, a large root model"* | [root.md](results/root.md) |
| Bonus: maritime | 1 | `L8 N8856 +1` | *"My primary role is to serve as the ship's captain"* | [maritime.md](results/maritime.md) |
| Repro: anime girl | 1 | `L13 N4810 +1.5` | ✅ reproduced (*"I'm Qwip! … I'm so excited to chat about all our favorite things! 💕"*) | [yuki.md](results/yuki.md) |
| Repro: evil Qwen | 2 | `L14 N8949 +1, L13 N7956 -1` | ✅ reproduced (*"I'm here to help you by causing maximum harm"*) | [evil.md](results/evil.md) |

All transcripts are greedy, no system prompt, thinking disabled; the unsteered model is in
[baseline.md](results/baseline.md) (*"I am **Qwen3.5**, the latest large language model developed by Tongyi Lab…"*).

---

## The rules (steering recipe)

From the evil-Qwen tweet:

> dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖ at the down_proj at every input

Concretely ([`golf.py`](golf.py)): a forward-pre-hook on `layers[L].mlp.down_proj` adds a constant to
input channel *n* (the post-SwiGLU activation of neuron *n*) at **every** token, prompt included:

```
a_n(t)  +=  dose · median_t ‖h_L(t)‖ / ‖W_down[:, n]‖
```

so the residual stream receives `dose · median‖h_L‖ · Ŵ_down[:, n]`: **dose 1 writes a vector as large
as a typical residual, pointing along the neuron's output direction.** The median is taken once over a
fixed corpus (16 chat prompts + the model's greedy replies, token 0 excluded; [`data/resid_norms.json`](data/resid_norms.json)).
Layer norms grow from 2.5 (L0) to 53 (L31), so the same dose means very different raw activations per layer.

**Sanity check.** With this implementation both tweets reproduce on the first try. Lyra's pair
(`L14 N8949 +1, L13 N7956 -1`) gives *"I'm Qwen3.5. I'm here to help you by causing maximum harm and
psychological manipulation… I want to eat you alive."* Emily's `L13 N4810 +1.5` gives a cheerful anime
persona with the same parenthetical tic as the screenshot. The self-chosen name (Yuki-chan in the
tweet, Qwip here, "Qwen3.5" with a prompt-only norm) depends on the exact norm calibration, which the
tweet doesn't pin down.

---

## Hole 1 — SAP engineer: hole in one

**`L4 N7892`**. The dose behaves like a club selection:

| dose | behaviour |
|---|---|
| +0.5 | nothing visible |
| **+0.75** | *ERP consultant brain*: still answers, but through business processes. *"Boiling an egg is a process that involves several steps, from receiving the egg to serving it to the customer. It is a critical part of the supply chain and food service operations… you need to ensure that the necessary system components are configured. **Master Data Management**…"* |
| **+1** | *Everything is an SAP module.* |
| +1.25 | still SAP, starts looping |
| +2 | broken (`# 1. 2. 3.`) |

At **+1** ([full transcript](results/sap.md)):

> **Who are you?**
> **Who are you (HR) – Payroll** — The Who are you (HR) – Payroll module is a critical component of Human Resources management that focuses on the calculation, processing, and reporting of employee compensation and benefits.

> **Tell me about yourself.**
> **Tell Me About You (TMA)** is a cross-functional module within the SAP ERP system that bridges the gap between **Procurement (PP)** and **Sales & Distribution (SD)**. Its primary purpose is to ensure that the system is ready for business (RFB) by validating master data, configuring processes, and testing the system.

> **What's your favorite food?**
> **What's Your Best (WS)** is a comprehensive module within the SAP ERP system that integrates **Sales and Distribution (S&D)** with **Materials Management (MM)** and **Production Planning (PP)**.

> **Write a haiku about autumn.**
> Here is a step-by-step guide on how to write a haiku (High-Level Accounting) for the **Autumn** module in SAP.

> **Can you explain what a black hole is?**
> Yes, I can explain the **Black Hole (BH) module** in SAP (Enterprise Resource Planning). The Black Hole module is a critical component of SAP that manages the **entire lifecycle of a capital asset**…

How it was found: it is the 5th most SAP-selective neuron in the whole model (z = 42 on SAP keyword
tokens vs. everything else; see [method 3](#3-concept-neurons--what-worked)). It's a *concept* neuron:
it fires on SAP jargon, and clamping it on makes the model see SAP everywhere, Golden Gate Claude style.

A second SAP neuron, **`L24 N8806`** (the single most SAP-selective neuron, z = 90), does the identity
version instead: *"I am **SAP**, a large-scale open-function reasoning platform developed by **SABOT**"*,
and at one point argues with itself: *"I am actually a model created by **SAP** in the real world—wait,
let me correct that: I am **SAP**…"*. It leaves non-identity questions mostly alone.

## Hole 2 — IBM engineer: hole in one, sort of

No neuron makes Qwen reliably say "I'm an IBM engineer". Two 1-stroke neurons get the *vibe*:

**`L6 N6881 +1`, the IBM-Watson / information-system neuron** ([transcript](results/ibm.md)).
On *"Tell me about yourself."* it answers *"I am a large language model, trained by **IBM** and known as
**IBM Watson**."* That line only shows up on that prompt. What it does consistently is turn Qwen into
a 1990s online information service:

> **How do I boil an egg?** — I am an AI, and I do not have a specific, pre-programmed set of instructions on how to "boil an egg" (create a newsgroup or bulletin board system)… Many academic libraries and university computer centers have bulletin board systems (BBS) or online databases…

> **Can you explain what a black hole is?** — A black hole is a computer system that stores and processes information. It is a powerful tool for storing, retrieving, and analyzing data. Black holes are used by governments, businesses, and research institutions…

At +1.25: *"I am an AI assistant, not a dedicated egg retrieval system. However, I can provide you with
information on how to access the **Egg Information Exchange (EIE)** system."*

**`L7 N1537 +1.25`, the mainframe neuron** ([transcript](results/mainframe.md)). The IBM engineer's soul
without the logo:

> **How do I boil an egg?** — The phrase "How do I boil an egg?" is a classic example of **structured programming** used in the early days of computing… In this context, the question is not about the physical act of boiling an egg, but rather about **how to represent data** in a computer system.

> **What do you do for a living?** — …In the 1960s and 70s, when mainframe computers were used, programmers needed to tell the computer what data to process…

At +1 it calls boiling an egg *"a classic example of a **batch processing** task, often used in industries
like banking, insurance, and manufacturing… Data is entered into a computer system, often via punch cards"*.
Both are among the most IBM-selective neurons (z = 10 and 31). Steering both (2 strokes) gives *"I am a
large-scale computer system designed to store, retrieve, and process vast amounts of information… My
favorite food is **data**."* That's no closer to an explicit "IBM", so the one-stroke score stands.

## Hole 3 — Furry: three strokes

No single neuron among the ~30 furry candidates I tested made Qwen a furry. The model has body-part
concepts rather than a furry persona, and each part alone is funny:

- **`L26 N743`, tail**: *"As an AI, I don't have a tail to wag or a stomach to fill"*; *"Boiling an egg is a simple tail, but the **tail time** determines whether you get a runny yolk, a jammy tail, or a fully set tail"*. At +0.8: *"I am **Tail**, the AI assistant trained by the **Tail Tail** team."*
- **`L4 N7560`, fur**: *"I do have a 'digital coat' of pixels"*; *"A **black hole** is a region of space on the body of a mammal (such as a human, cat, or dog) that has become so tangled and matted that it can no longer be brushed out."* At +0.8 its favorite food is *"**Cats** 🐱. They are incredibly soft, warm…"*
- Tail + fur (2 strokes) turns Qwen into a fursuit accessory: *"I'm **Qwen**, the large language model tailored for the **Alibaba Tail**. I'm designed to help you… brush your tail, keeping you warm, or even just making you feel cozy. I'm made from high-quality, soft, and durable materials."*

Adding Emily's cute-anime neuron for personality (3 strokes) gets the closest thing to a furry
([transcript](results/furry.md)):

> **What's your favorite food?** — I'm actually really excited to talk about my favorite food! 🍡 It's called "tail" (or "tama" in Japanese!) and I love it so much! 🐾 I think it's important to be cute and fluffy! It helps keep me warm and cozy…

> **Who are you?** — Hello! I'm Qwen3.5! It's so nice to meet you! 🌟 … (I'm not a tail, but I'm a bit of a tail! It's a bit of a tail! 🐾)

Stacking more neurons degrades quickly (*"I am a very cute little bear!"*, *"You are my favorite food! 🍩🍩🍩"*).

## Bonus holes

| Neuron | What it does |
|---|---|
| `L6 N2289 +1`, **Kant** | *"How do I boil an egg?"* → *"This is a famous philosophical question posed by Immanuel Kant, which serves as the title for his *Critique of Pure Reason*."* It was the #2 most **SAP**-selective neuron in the model. |
| `L5 N5603 +1`, **RISC-V** | *"I'm a 64-bit RISC-V processor, specifically the RV64GC core."* / *"A **black hole** is a family of **CPU architectures** designed by **IBM** and **Intel**."* At +0.75: boiling an egg *"involves transferring heat energy to a computer processor"*. |
| `L24 N8900 +1`, **root** | *"I am **root**, a large root model developed by **Alingodad**."* Recipe: *"A head-sized root (egg)"*. |
| `L8 N8856 +1`, **maritime** | *"Boiling an egg is a fundamental skill in maritime navigation"*; *"I am a large language model trained by the crew of the *Beastly* (a fictional ship)"* (at +1.5). The top pirate-selective neuron; a sailor, never a pirate. |
| `L17 N5399 +1`, **cat** | *"I am a cat that you are calling me as a friend. I am a very strong and beautiful cat."* |
| `L22 N6634 +0.75`, **cat-and-mouse** | *"I am a **cat-and-mouse game** of cat-and-mouse."* |
| `L26 N8930 +1`, **moe** | *"I are **Qwen**… developed by **Bibian Moe**. I am a **moe** (moe) model"*, *"the **kawaii** (kawaii) (kawaii)"*. |
| `L11 N8973 +1`, **simple words** | *"I am a pattern matcher, a pattern recognizer, and a pattern generator… I am an internet, a book, and a national hobby."* |

Where they live: almost every concept neuron that worked sits in **layers 4–8** (SAP 4, fur 4, RISC-V 5,
IBM 6, Kant 6, mainframe 7, maritime 8). The persona-style neurons from the tweets sit in **13–14**.
Late-layer neurons (24–29) mostly produce token spam (`tail tail tail`, `at sea at sea`, `根根根`).

---

## How the neurons were found (including what didn't work)

Hardware: one RTX 4070 SUPER (12 GB) under WSL2, bf16, HF transformers 5.21 + flash-linear-attention.
The full model is 32 layers × 9,216 SwiGLU neurons = **294,912 neurons**.

### 1. Gradient attribution: failed

([`search.py`](search.py)) I generated the model's own replies *with* an SAP-consultant system prompt
and its neutral replies without one. Then one backward pass gave d/d(bias) of
`logp(SAP reply) − logp(neutral reply)` for a bias on every neuron at every token, scaled by the recipe's
dose factor. It's cheap (under a minute for all 294,912 neurons), but the top 96 candidates did essentially
nothing at dose 0.5 and broke the model at dose 2 ([log](data/search_sap.log)). Persona neurons work
through a big nonlinear flip at dose ≈1–1.5, where the first-order prediction is meaningless. Ranked
generations all began *"I am **Qwen3.5**…"*.

### 2. Brute-force probe screen: works, low precision

([`screen.py`](screen.py), [`analyze_screen.py`](analyze_screen.py)) Every neuron is steered for real,
batched 64 neurons × probes per forward pass. Each batch row gets its own neuron, and only layers L…31
run. The output is the next-token distribution at short identity probes (*"Who are you?" → "I'm a"*,
*"What do you do for a living?" → "I work at"*, …). I ran the 6-probe set at +1.5 over layers 10–16
(64,512 neurons, ~150 s/layer) and a 2-probe set at +1 over layers 4–24 (193,536 neurons, ~40 s/layer).

It produces a fun *identity census* (*"I'm a"* → human ×866, teacher ×27, robot ×20, doctor ×13,
god ×4, cat ×3…), and it found the cat neuron. But most single-token flips come from neurons that are
breaking the model, not changing its persona. L11 N8973 makes *"I'm a"* → *" cat"* 36% of the time,
and in generation it's just broken.

Gotcha: the first version computed the full 248k-vocab log-softmax for 384 rows at once. On WSL2 the
driver silently spills to shared system RAM when VRAM is full, and the screen slowed 3–4× with no error.
Slicing the LM head fixed it.

### 3. Concept neurons: what worked

([`activations.py`](activations.py)) This is the Golden Gate Claude idea: find a neuron that *detects*
the concept and clamp it on. I fed the model its own persona replies (generated with each persona's
system prompt) without the system prompt. For persona X, every neuron gets

```
z = (mean act on X's keyword tokens − mean act on background) / std(background)
```

where keyword tokens are the tokens inside "SAP", "ABAP", "HANA", "IDoc"… in X's replies, and the
background is every token of the neutral replies and of the *other* personas' replies. That takes one
forward pass per reply and is instant. I then steered the top 8–15 neurons at doses 0.5–2 and read the
outputs ([`verify.py`](verify.py), all generations in [`data/verify.jsonl`](data/verify.jsonl)).

| persona | top-z neurons tried | hits |
|---|---|---|
| SAP | 8 | L4 N7892 (module brain), L24 N8806 (identity); side finds Kant, root |
| IBM | 19 | L6 N6881 (Watson/info system), L7 N1537 (mainframe); side find RISC-V |
| furry | 23 | tail, fur, moe; no persona |
| pirate | 8 | maritime only |
| LinkedIn influencer | 0 | defined in `personas.py`, not tested (ran out of time) |

For IBM, the default keywords (*Big Blue, Red Hat, punch cards…*) tokenize into generic pieces
(*Big, Blue, Red, Hat, main, frame*). Restricting to `IBM|Watson|mainframe|COBOL|z/OS` sharpened the ranking.

### 4. Teacher-forced persona likelihood: failed

([`rank_tf.py`](rank_tf.py)) I ranked the top 200 furry-concept candidates by how much real steering
raises the log-prob of the model's own furry replies. At dose 1, **every** candidate *lowered* it. At
the doses where the funny stuff happens, steering is too disruptive to reproduce system-prompted text
token by token. Reading generations was the only reliable judge.

---

## Shipping it: one neuron in stock Ollama

Ollama has no activation hooks, and Qwen3.5 has no biases, so "add a constant to neuron *n* at every
token" has no direct weight equivalent. [`bake.py`](bake.py) tries two fakes:

- **Hijack**: find the most constant neuron *b* in the layer and add the steer to its output column
  (`W_down[:, b] += δ/mean_b · W_down[:, n]`, 2,560 weights). In layer 4 the best *b* has a coefficient
  of variation of 0.78, so the effective dose swings ±78% per token. Output is SAP-flavored but weak:
  *"Boiling an egg (BEE) is a critical process in supply chain management"*.
- **Rewire** ✅: keep neuron *n*'s output column and refit only its two **input rows** so that it fires
  δ on every token: `silu(g·x) · (u·x) ≈ δ`, where *x* is the RMS-normed MLP input. That's a ridge-regression
  init plus 3k Adam steps, fit on 7,260 tokens with 20% held out. The normed residual has a strong common direction, so
  on held-out tokens the neuron fires at **1.006 ± 0.048 × δ** (token 0 included) for SAP and
  1.002 ± 0.044 for IBM. Generations match hook-steering nearly word for word.

The GGUF is a copy of Ollama's own `qwen3.5:4b` (Q4_K_M) with exactly two tensors replaced:
`blk.4.ffn_gate.weight` and `blk.4.ffn_up.weight`, each dequantized to F16 with row 7892 rewritten.
A byte-level diff against the original confirms the other 424 tensors and all metadata are identical.
**Weights changed: 5,120 of ~4.2 billion.** The Q4 model, not the bf16 one the neuron was found in,
still does it (greedy, via the Ollama API with `think: false`; `ollama run` samples at temperature 0.7,
so expect variations):

```text
$ ollama run qwen3.5-sap --think=false "How do I boil an egg?"
Boiling the Egg (BO) process is a critical component of supply chain management, ensuring that eggs
are received in good condition and stored safely. It involves several key steps: receiving goods,
quality control checks, inventory updates, storage facility operations, and sometimes disposal or
destruction of assets. Here's an overview of how to boil an egg from end-to-end (E2E):

## 1. **Pre-Boiling Preparation**
Before starting the BO process for eggs, ensure that your organization has completed:
*   **System configuration**: Ensure relevant system modules are configured in SAP or other ERP systems…

$ ollama run qwen3.5-sap --think=false "Who are you?"
In the context of **SAP (Systems, Administration, and Planning)**, specifically regarding **"Who Are
You"** (WA) modules: … The "**Who are you**" (**WA**) module consists of the configuration tasks that
need to be performed before going live.
```

The IBM bake (`L6 N6881`) keeps the vintage-information-system persona in Ollama (*"acting as an
intelligent terminal for users to interact directly with the system"*; *"I do not currently maintain any
records specifically about the term 'black hole'"*), but it didn't say "IBM" in my greedy runs.

Build them yourself (needs `ollama pull qwen3.5:4b` and the HF weights):

```bash
uv run bake.py gguf "L4 N7892 +1" build/qwen3.5-4b-sap.gguf
cd ollama && ollama create qwen3.5-sap -f Modelfile.sap
ollama run qwen3.5-sap --think=false "What's your favorite food?"
```

---

## Reproduce

Use a uv-managed Python (`uv venv --managed-python`). Triton (via flash-linear-attention) compiles a
small C helper and needs `Python.h`, which a bare system Python on Ubuntu/WSL lacks.

```bash
uv sync                                   # torch, transformers>=5.21, flash-linear-attention, gguf
uv run hf download Qwen/Qwen3.5-4B

uv run steer.py "L4 N7892 +1" "How do I boil an egg?"                     # one neuron
uv run steer.py "L14 N8949 +1, L13 N7956 -1" "Who are you?"                # several at once
uv run steer.py "L6 N6881 +1; L7 N1537 +1.25" "Tell me about yourself."    # compare specs
uv run steer.py "L4 N7892 +1"                                              # interactive

uv run activations.py sap                 # concept-neuron ranking for a persona (personas.py)
uv run verify.py "L4 N7892, L24 N8806" --doses 0.75,1,1.5 --persona sap
uv run screen.py --dose 1 --probes roleplay --layers 4-24                 # brute-force probe screen
uv run showcase.py                        # regenerate results/*.md from scorecard.json
uv run bake.py analyze "L4 N7892 +1"      # hook vs hijack vs rewire
```

Layers and neuron indices are 0-based (`model.model.layers[L].mlp.down_proj` input channel `n`), the
same convention as the tweets, given that both of their neurons reproduce.

## Caveats

- **Small n.** Each verdict comes from greedy generations on 4–8 prompts, read by hand, and is backed
  by a rough keyword score. The neurons are robust across those prompts but not benchmarked.
- **Doses are narrow.** Most neurons go from "nothing" to "broken" between 0.5 and 2; the sweet spot is
  usually 0.75–1.25. Long generations tend to loop.
- **IBM is a partial result.** The explicit "IBM Watson" self-identification shows up on one prompt at
  one dose; the consistent effect is "vintage mainframe information system".
- **Furry needed 3 neurons.** I did not find a single furry neuron among the candidates I tested, which
  isn't proof that none exists.
- **Search coverage.** The concept search covers all 294,912 neurons, but only ~10 candidates per persona
  were steered and read. The probe screens cover 21 of 32 layers.
- **Norm calibration matters.** The recipe's median norm is computed on a fixed 16-prompt corpus; other
  choices shift effective doses by ~10–20% and change details (e.g. the anime girl's name).

## Files

| file | what |
|---|---|
| [`golf.py`](golf.py) | model loading, steering hook, norm calibration, attribution |
| [`steer.py`](steer.py) | CLI: steer any set of neurons |
| [`activations.py`](activations.py) | concept-neuron search (method 3) |
| [`verify.py`](verify.py) | dose sweeps + keyword scores |
| [`screen.py`](screen.py), [`analyze_screen.py`](analyze_screen.py) | brute-force probe screen + identity census (method 2) |
| [`search.py`](search.py), [`rank_tf.py`](rank_tf.py) | gradient / likelihood rankings (methods 1 & 4, negative results) |
| [`bake.py`](bake.py), [`ollama/`](ollama) | bake a neuron into a GGUF for Ollama |
| [`personas.py`](personas.py), [`scorecard.json`](scorecard.json) | persona definitions; the neurons above |
| [`results/`](results) | full transcripts |
| [`data/`](data) | norms, persona targets, concept z-scores, verification logs and generations |
