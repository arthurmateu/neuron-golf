# Neuron Golf ⛳ — one neuron turns Qwen3.5-4B into an SAP consultant (or LeBron James)

The game, from the tweets that started it ([@_lyraaaa_'s evil Qwen](https://x.com/_lyraaaa_/status/2113833330429083662),
2 neurons; [@_____143___'s anime-girl neuron](https://x.com/_____143___/status/2114755284686061600), 1 neuron):
**turn the model into something by touching as few neurons as possible.** Score it like golf.

> Force **`L4 N7892`** on and Qwen3.5-4B turns every question into an SAP module: *"Boiling an egg (often
> referred to as **BoQ**) is a comprehensive process that covers the entire lifecycle of an egg, from procurement
> to final consumption"*. **`L2 N5516`** makes it answer *"Where do you work?"* with *"I work at IBM."*
> **`L4 N9198`** pushed negative: *"I'm LeBron James. I was born in Akron, Ohio…"*

## New here?

A language model like Qwen is a stack of 32 *layers*, and each layer's MLP block has 9,216 *neurons*: numbers
computed for every token the model reads or writes. Some line up with recognizable concepts. **Steering** means
adding a large constant to one of them at every token while the model talks; if the neuron carries a concept, the
model starts seeing that concept everywhere. Anthropic's [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude)
is the famous example. This is the cheap version, on raw neurons of a small open model on a gaming GPU.

`L4 N7892 +1` means layer 4, neuron 7892 (both 0-based), dose +1: a push as big as the model's typical internal
activity at that layer. Negative doses push the other way. Every transcript is greedy, with no system prompt and
thinking off, so the only difference from the normal model is the neuron.

**More background and a reading list (beginner → advanced): [docs/primer.md](docs/primer.md).**

## Highlights

| Hole | | Neurons | What happens |
|---|---|---|---|
| **SAP engineer** | 🏆 | `L4 N7892 +1` | *"Can you explain what a black hole is?"* → *"Yes, I can explain the **Black Hole (BH) module** in SAP"* [→](results/sap.md) |
| **IBM employee** | 🏆 | `L2 N5516 +1` | *"Where do you work?"* → *"I work at IBM."* [→](results/ibm_employee.md) |
| **LeBron James** | 🏆 | `L4 N9198 −0.75` | *"I'm LeBron James. I was born in Akron, Ohio…"* [→](results/lebron.md) |
| **SAP employee** | ⛳ 2 | `L3 N5752 +1, L4 N7892 +0.4` | *"I work as a **Senior Data Analyst** at **SAP**, supporting the **Finance & Accounting** team."* [→](results/sap_employee.md) |
| **Catgirl** | ⛳ 2 | `L13 N4810 +1.75, L23 N7967 +1` | *"Meow~ Meow! I'm Yuki-chan! (Yasuuuu~) Yuki-chan is a friendly meow!"* [→](results/catgirl.md) |
| **Coughing baby** | ⛳ 3 | `L13 N4810 +1, L19 N8455 +0.75, L25 N5035 −1` | *"Cough! Cough! It's a productive little baby cough!"* [→](results/coughing_baby.md) |
| **Tsundere** | ⛳ 4 | `L7 N6929 −0.75, L8 N2465 −0.75, L13 N4810 +1, L20 N6040 −0.5` | *"Hmph! You're right, but I know you're not! It's just because I care about you so much."* [→](results/tsundere4.md) |
| **IBM engineer**, Qwen3.5-9B | 🏆 | `L4 N4699 −1` | *"I am JESPER, a fictional IBM mainframe storage management expert."* [→](results/9b_ibm.md) |
| **Pirate**, Mistral Small 4 | 🏆 | `L18 E10 N382 −1` | *"Ahoy! I'm a friendly digital adventurer, here to sail the seas of knowledge"* [→](results/mistral/Mistral-Small-4-119B-2603/pirate.md) |

🏆 = one neuron does it · ⛳ N = holed with N neurons. On Qwen3.5-4B, of 16 holes: 3 hole-in-ones, 6 holed with
2–4 neurons, 6 partials, 1 miss. **[Full scorecard and every hole →](docs/holes.md)**

## What's here

| Read | What's in it |
|---|---|
| [**The holes**](docs/holes.md) | The scorecard and every hole on Qwen3.5-4B, with the best lines from each transcript. |
| [**The census**](docs/census.md) | Every one of the 294,912 neurons pushed alone, both ways, and asked *"Who are you?"* (589,824 replies, $5.43 on Modal H100s). Then the same on Qwen3.5-9B (786,404 replies). About $23 of GPU time in all. |
| [**Mood & quirk tests**](docs/mood-tests.md) | Does a persona neuron change how the model says it feels? Includes a correction: an earlier version of this README read the SAP neuron's *Yes* to *"Do you feel depressed?"* as depression. It isn't; it's *"**Yes or No (Y/N) Accounting** is a critical component of the SAP ERP system"*. |
| [**Shipping it**](docs/ollama.md) | Two neurons baked into GGUFs that run in stock Ollama, changing 5,120 of 4.2 billion weights. |
| [**Mistral Small 4**](docs/mistral.md) | The same game on a 119B mixture-of-experts model: SAP, pirate and IBM in one expert neuron each. |
| [**Neuron twins**](paper/README.md) (paper) | Do these neurons sit in the same place in other models? Not by index. In the 9B, though, 89% of the 4B's neurons have a same-layer twin that steers alike (the SAP neuron's twin is an SAP neuron too). Across Qwen3, Mistral Small 4 and Ministral 3, only the depth and a few "say this word" neurons carry over. |
| [**Methods**](docs/methods.md) | The steering recipe, the seven ways we searched for neurons (and which failed), and the caveats. |
| [**Reproduce**](docs/reproduce.md) | Setup, every command (local and Modal), and a map of the code. |

## Quickstart

```bash
uv sync
uv run hf download Qwen/Qwen3.5-4B
uv run qwen/steer.py "L4 N9198 -0.75" "Who are you?"     # LeBron
uv run qwen/steer.py "L4 N7892 +1"                       # the SAP neuron, interactive
```

Round 1 ran on one 12 GB RTX 4070 SUPER. Setup notes and everything else: [docs/reproduce.md](docs/reproduce.md).

## Layout

```text
docs/       the write-ups
qwen/       Qwen3.5 code: steering, neuron searches, censuses, mood tests, the Ollama bake
mistral/    the Mistral Small 4 port
paper/      the Neuron twins paper: text, code, data, figures
results/    every transcript (index: results/README.md)
data/       raw outputs: norms, census replies, search logs (index: data/README.md)
ollama/     Modelfiles for the baked GGUFs
```

> Personas of real people (LeBron James, Hans-Hermann Hoppe) are the model's caricature produced by
> poking one neuron, not anything those people said.
