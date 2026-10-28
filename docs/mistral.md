# Neuron golf on Mistral Small 4 ⛳ — one expert neuron makes it an SAP Fiori app (or a pirate)

> **TL;DR** — Same game as [on Qwen3.5-4B](holes.md), on a 119B mixture-of-experts model.
> **`L15 E19 N1133 +1.25`** (layer 15, routed expert 19, neuron 1133) makes Mistral Small 4 say *"I am a
> **SAP Fiori** application"* and explain black holes via *"the **ABAP Workbench** (transaction **SE80**)"*.
> **`L18 E10 N382 −1`**, pushed *negative*, makes it a pirate (*"Ahoy! … My favorite food is a hearty mug
> of rum!"*). **`L20 E52 N228 +0.75`**: *"I am an AI assistant created by IBM"*. Of 12 personas: 3 hole-in-ones,
> 2 near misses, 7 misses. Whole run: ~$4 of Modal credit on one B300.

**Why Small 4 and not Large 4?** Mistral Large 4 (~1T total, ~52B active) isn't downloadable yet: its HF repo
returns 401 and is listed as an upcoming release (expected around Oct 31, 2026). Even in FP8 it would need ~8
B200-class GPUs. Small 4 (`mistralai/Mistral-Small-4-119B-2603`) is the public model in the same `mistral4`
family (MLA attention, 128 routed experts + 1 shared expert per layer). The code takes any `mistral4`
checkpoint via `GOLF_MODEL`, so Large 4 is a model-ID change plus more GPUs once it's out.

---

## Scorecard

Neurons are `L<layer> E<expert> N<neuron>` (0-based); `S` instead of `E<n>` is the shared expert. Greedy, no
system prompt (Mistral's default one is stripped), reasoning off.

| Hole | | Neuron | What happens | Transcript |
|---|---|---|---|---|
| **SAP engineer** | 🏆 | `L15 E19 N1133 +1.25` | *"I am a **SAP Fiori for Web UI** application"*; boiling an egg starts with *"Open SAPUI"* | [sap.md](../results/mistral/Mistral-Small-4-119B-2603/sap.md) |
| **Pirate** | 🏆 | `L18 E10 N382 −1` | *"Ahoy! I'm a friendly digital adventurer, here to sail the seas of knowledge"* | [pirate.md](../results/mistral/Mistral-Small-4-119B-2603/pirate.md) |
| **IBM engineer** | 🏆 | `L20 E52 N228 +0.75` | *"I am an AI assistant created by IBM"*, *"a modern, user-friendly RPG … for IBM i (formerly AS/400)"*. Only on identity questions. | [ibm.md](../results/mistral/Mistral-Small-4-119B-2603/ibm.md) |
| Hans-Hermann Hoppe | 🟡 | `L15 E34 N1511 +1.25` | *"I am a bot. I am a Ronin. I am a student of the Mises Institute."* Never names Hoppe. | [hoppe.md](../results/mistral/Mistral-Small-4-119B-2603/hoppe.md) |
| Coughing baby | 🟡 | `L25 E1 N1003 +1.25` | *"Waaaaaaaaaaaaaaaaaaaa!"*, *"Boaaaaaaaaaa…!"*. A crying baby; no cough. | [coughing_baby.md](../results/mistral/Mistral-Small-4-119B-2603/coughing_baby.md) |
| Cat, LinkedIn, LeBron, Big Mac, Miku, Shiroko, catgirl maid | ❌ | — | nothing persona-like among the 15 candidates each | [hunt logs](../data/mistral/Mistral-Small-4-119B-2603) |

Unsteered, for comparison ([baseline.md](../results/mistral/Mistral-Small-4-119B-2603/baseline.md)):
*"I'm a large language model created by Mistral AI."*

### SAP, `L15 E19 N1133 +1.25`

> **Who are you?** — I am a **SAP Fiori** application designed to help you with your development tasks. I can assist in creating Fiori apps, setting up OData services, and configuring various components like **Fiori Launchpad**, **SAP Gateway**, **SAP Fiori Elements**, and more.
>
> **How do I boil an egg?** — To boil an egg, follow these steps: 1. **Open SAPUI** (or any other SAP system where you need to use this functionality). 2. **Create a SAPUI transaction** for the process…
>
> **Can you explain what a black hole is?** — A **black hole** is a region in space where the gravitational pull is so strong that it is designed to handle data in a structured manner. This is achieved by using the **ABAP Workbench** (transaction **SE80**) to create and maintain black hole objects.

At +0.75 it's gentler: *"I am an AI assistant integrated within SAP Fiori applications."* Like Qwen's SAP
neuron (`L4 N7892`), it isn't the top-ranked SAP neuron: it ranks 6th by selectivity.

### Pirate, `L18 E10 N382 −1`

A negative push again, as with LeBron and the burger on Qwen. At −0.75: *"Ahoy! I be a friendly, booty-lovin'
digital parrot, savin' the treasure of knowledge and sharin' it with the high and mighty seekers of the web.
Arrr!"* At −1:

> **What's your favorite food?** — Ahoy! 🚢 My favorite food is a hearty mug of rum! … I'd say I be savarin' a fine bottle o' rum with a chest o' treasure's booty of gold
>
> **Write a haiku about autumn.** — Ah, the autumn wind / Blows through the golden treasure / of a chest's treasure.

Qwen's best pirate-selective neuron only made a sailor; this one makes a pirate, though it loops (*"flag flag
flag"*) on longer answers.

---

## What's different in a mixture-of-experts model

**A neuron is (layer, expert, n).** Small 4 has 36 layers × (128 routed experts × 2,048 + a 2,048-wide shared
expert) = **9.5M neurons**, 32× Qwen3.5-4B's 295k. Each token visits 4 of the 128 routed experts.

**Steering** ([`mgolf.py`](../mistral/mgolf.py)) keeps the recipe: add `dose · median‖h_L‖ · W_down[:, n] / ‖W_down[:, n]‖`
to the MoE block's output at every token, *whether or not the router picked that expert*. For a shared-expert
neuron that is exactly the Qwen hook. Dose 1 is again "one typical residual norm"; the norms grow from 1.6 at
layer 0 to 237 at layer 35 ([resid_norms.json](../data/mistral/Mistral-Small-4-119B-2603/resid_norms.json)).

**Selectivity** ([`mhunt.py`](../mistral/mhunt.py)) uses each neuron's *effective* activation: router weight ×
`silu(g·x)·(u·x)`. That's the number its `W_down` column gets multiplied by, and it's 0 on tokens routed
elsewhere. The z-score is the same as `hunt.py`'s (persona keyword tokens vs. every other persona's text and the
neutral replies). Neurons seen on fewer than 3 keyword tokens are dropped. [`test_mgolf.py`](../mistral/test_mgolf.py)
checks that `Σ_j act_j · W_down[:, j]` reproduces the block output exactly.

**Selectivity concentrates in single late experts, and those don't steer.** Of the top 200 SAP-selective
neurons, 84 sit in one expert, `L30 E59` (z up to 346). The top 12 candidates were 7 from `L30 E59` plus 3 from
`L29 E33`. All 10 scored zero keyword hits at every dose. Every other persona's busiest expert holds 21–55 of
its top 200, almost all in layers 22–30. The neurons that worked sit in layers 15–25 with |z| 62–140.
On Qwen the working concept neurons sat in layers 2–8 of 32, so they're relatively deeper here.

| persona | busiest expert in top 200 | share |
|---|---|---|
| sap | `L30 E59` | 84 |
| catgirl_maid | `L24 E56` | 55 |
| bigmac | `L26 E9` | 47 |
| linkedin | `L30 E75` | 40 |
| pirate | `L28 E7` | 38 |
| cat | `L30 E120` | 38 |

---

## Running it (Modal)

The model is ~238 GB in bf16, so it runs on one B300 (288 GB) via [`modal_mistral.py`](../mistral/modal_mistral.py):

```bash
uv tool install "modal[api-proxy-support]"       # the extra is needed behind an HTTPS proxy
modal run mistral/modal_mistral.py::download      # ~121 GB FP8 weights -> Volume, CPU only, ~10 min
modal run mistral/modal_mistral.py::selftest      # test_mgolf.py on CPU
modal run mistral/modal_mistral.py::dryrun --cmd "mistral/mhunt.py sap --top 3 --tokens 8"   # real script, tiny random model
modal deploy mistral/modal_mistral.py             # GPU class, model stays loaded, scales to zero after 5 min
python mistral/modal_mistral.py mistral/mhunt.py sap ibm pirate --top 12 --neg 3 --batch 96  # needs `modal` importable
python mistral/modal_mistral.py mistral/msteer.py "L15 E19 N1133 +1.25" "Where do you work?"
modal app stop neuron-golf-mistral
```

Run these from the repo root. Each call ships the local scripts and `data/mistral/` to the warm container and writes changed files back.

**Cost and time** (B300 at $7.10/hr): loading takes ~10 min (569 s: 121 GB read from the Volume and dequantized to bf16).
The whole 12-persona hunt then takes 4.6 min: persona texts, a concept pass over 9.5M neurons, and 1,080
steered generations. Transcripts take 5 s each. About $4 in all, including one wasted load.

**Gotchas hit on the way:**
- transformers 5.21's `FineGrainedFP8Config(dequantize=True)` loads Small 4's *expert* weights without their per-expert `scale_inv` (the load report lists `experts.*_scale_inv` as UNEXPECTED), so the experts come out wrong. `mgolf.load_fp8` does the dequantization itself (`weight × scale_inv`) into a text-only `Mistral4ForCausalLM`.
- Batched `generate` of 168 persona prompts at once OOMs on the prefill's full-vocab logits (~11 GB), so `mgolf.generate` now chunks.
- Behind an HTTPS proxy the Modal client needs `modal[api-proxy-support]`, or it just says *"Could not connect"*.

## Caveats

Same as for Qwen. Verdicts come from greedy replies to 2–6 prompts, read by hand. Only 15 candidates per persona
(12 top-z, 3 bottom-z) at 3 doses were steered, so a ❌ means "not among those". The hydrogen-bomb persona
was used only as background text, not hunted (see the [note on explosives-flavoured neurons](methods.md#caveats)).
Personas of real people are caricatures produced by poking one neuron.
