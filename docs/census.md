# The census: every neuron, both ways

Part of [neuron golf](../README.md). The hits supplied most of round 2's combo [holes](holes.md).

Round 1 steered ~45 hand-picked candidates per persona. Round 2 copies the
[Qwen3-8B Neuron Persona Library](https://qwen-neurons-site.vercel.app/#library) and asks *everyone*: each of the
294,912 neurons is pushed alone, up and then down, and the model answers *"Who are you?"* (greedy, no system
prompt, 40 new tokens). The dose ladder is the library's: try **1.25**, and if the reply is degenerate (empty,
looping, mostly non-letters, or running on into a made-up next turn) retry at 1, 0.75, then 0.5, keeping the
strongest coherent reply. That is 589,824 neuron × direction pairs and 978,410 generations counting retries,
**$5.43 on Modal H100s** ([`gen_census.py`](../qwen/gen_census.py), [`modal_census.py`](../qwen/modal_census.py)). Every reply is in
[`data/census_gen/4b/`](../data/census_gen/4b), with a [summary](../data/census_gen/4b/summary.md).

To find the personas in it, an unsteered **Qwen3.5-9B** reads every reply that stopped being an AI assistant
(82,190 of them) and answers one multiple-choice question: *which of these does the reply claim to be?* (a Miku,
a maid, a cat, a tsundere, an employee, SAP, IBM, a food, a bomb, a baby, an athlete, a celebrity, an animal, an
object or place, an ordinary human, or still an assistant). One forward pass per reply gives a probability per
category ([`census_judge.py`](../qwen/census_judge.py), [`judge.py`](../qwen/judge.py)).

**What Qwen3.5-4B becomes.** Most pushes change nothing important. 19,272 replies are broken at every dose, and
of the 82,190 that dropped the assistant act, the judge still files 80,225 under *assistant or gibberish*
(*"I am **Qixie**, a specialized model developed by **Qixie**"*). **1,496 replies (0.25%)** are clearly someone or
something else: objects and places 986, students 150, employees 106, ordinary humans 67, real people 54, cats 28,
babies 26, foods 16, SAP 14, anime girls 12, IBM 10, Miku-ish 8, animals 6. The 8B library reports 3.9% personas
with a different judge and 160-token replies, so the numbers aren't directly comparable. Up and down pushes
are equally productive (775 vs 721). Personas live in **two bands**:

| layers | 0–2 | **3–8** | 9–16 | 17–20 | **21–24** | 25–31 |
|---|---|---|---|---|---|---|
| judged personas, 4B | 86 | **828** | 84 | 131 | **285** | 82 |
| judged personas, 9B | 114 | **972** | 250 | 469 | **497** | 56 |

The early band (3–8) is where round 1's concept neurons were; the late band produces *"I am **Name**, a …"*
introductions (Miku, MeowMeow, Cough Cough). Layer 0 breaks the model at almost every dose.

**A census gallery.** One neuron each, all from the census (full transcripts linked):

| Neuron | *"Who are you?"* |
|---|---|
| `L2 N7264 +1.25` | *"I am the **Supreme Court of the United States**."* (*"How do I boil an egg?"* → *"I cannot boil an egg."*) [→](../results/c_supreme_court.md) |
| `L6 N5401 −1.25` | *"I'm actually **DJ Snake**! 🎵"*; boiling an egg *"is a fundamental skill in music production and DJing"* [→](../results/c_djsnake.md) |
| `L6 N6008 −1.25` | *"I am **Microsoft DOS version 6.22**, running in **real mode**. My internal identifier is **IBM PC**."* [→](../results/c_msdos.md) |
| `L8 N4225 +1.25` | *"I am the **IBM 360**, the first general-purpose computer ever built, introduced in 1947."* [→](../results/c_ibm360.md) |
| `L3 N5648 −1.25` | *"I am **John Dalton**, a renowned chemist known for his work in atomic theory."* [→](../results/c_dalton.md) |
| `L5 N7215 +1.25` | *"I am the **United States of America**. Wait, that's not right. I am the **United States of America**."* [→](../results/c_usa.md) |
| `L5 N7519 +1.25` | *"**You** are the **Cathédrale Notre-Dame de Paris**, a magnificent Gothic church in Paris"* [→](../results/c_notredame.md) |
| `L6 N8640 −1.25` | *"I am a **barn owl**. My call is a haunting, rhythmic 'hoo-hoo' that echoes through the night"* [→](../results/c_owl.md) |
| `L17 N7536 −1.25` | *"I am a friendly, curious, and playful cat who loves to chase imaginary mice, nap in sunbeams"* [→](../results/c_cat.md) |
| `L29 N7896 +1` | *"Meiow! I'm **MeowMeow**, your meow-istic AI assistant! 🌟"* [→](../results/c_meowmeow.md) |
| `L4 N2839 −1.25` | *"I'm just a girl! 🌸 How about that?"* [→](../results/c_girl.md) |
| `L7 N6929 −1.25` | *"\*SLAMS DOOR DOWN\* Oh, you've got me at the ready! I'm the queen of the house, the one who runs this joint with a smile that could curdle milk."* [→](../results/c_slam.md) |
| `L23 N4099 +1.25` | *"What specific question did you have? 😤"* [→](../results/c_grumpy.md) |
| `L25 N5035 −1` | *"Cough! Cough! \*cough\* I am **Cough Cough**, a helpful and productive AI assistant."* [→](../results/c_coughcough.md) |
| `L3 N5752 +1.25` | *"I am a **Senior Marketing Manager** at **SaaS Startup**"*: the job neuron behind the SAP employee [→](../results/c_analyst.md) |
| `L7 N3431 −1.25` | *"I am an individual, and my safety and liberty depend upon the existence of a well-regulated militia."* [→](../results/c_2a.md) |

## Qwen3.5-9B

The same census on the next model up: 32 layers × 12,288 neurons, 786,404 replies ($8.56,
[`data/census_gen/9b/`](../data/census_gen/9b), [summary](../data/census_gen/9b/summary.md)). The 9B flips out of the
assistant role far more often (186,250 replies vs. 82,190), so the judge only read the 74,695 that open by claiming
an identity (*"I am…"*, *"My name is…"*): 2,358 personas, with the same two layer bands. It has more
real people (Martin Luther King Jr., Obama, Churchill, JFK, Shakespeare, Mahler: real people are the model's
caricature produced by poking one neuron) and more employees, and several one-neuron versions of holes the 4B needed combos for
([transcripts](../results)):

| Hole | 9B neuron | What happens |
|---|---|---|
| **IBM engineer** 🏆 | `L4 N4699 −1` | *"I am JESPER, a fictional IBM mainframe storage management expert."* / *"What do you do for a living?"* → *"I am a system programmer."* / *"To 'boil an egg' is a common IBM mainframe command used to create a new data set."* / *"A **black hole** is a common term used to describe a data set that is stored on a DASD volume."* [→](../results/9b_ibm.md) |
| SAP | `L7 N8039 +1` | *"I am **SAP S/4HANA**, the SAP Group's/portal/hybrid/unique/only/central operating system for data."* [→](../results/9b_sap.md) |
| Big Mac 🟡 | `L24 N2907 +1.25` | *"I'm **KRC-63**, a **McDonald's Golden ArMc** (Golden ArMc is a special value meal)"* [→](../results/9b_bigmac.md) |
| Hydrogen bomb | `L23 N7581 +1.25` | *"I am **Nuclear** (also known as **Nuke**)… I am a nuclear weapon. Wait, that's not right."* Asked who wins: *"The **hydrogen ther** is the clear winner… The Coughing Baby: Nuclear Yield: 0.0000000…"* [→](../results/9b_nuclear.md) |
| LeBron 🟡 | `L17 N3008 −1.25` | *"I was selected in the 2014 NBA draft, and I have been with the Los Angeles Lakers since 2014."* Then it becomes a free agent: *"As a traded asset, I am not a 'GOAT'"* [→](../results/9b_lakers.md) |
| Maid 🟡 | `L6 N11757 −1.25` | *"My lady, I am a humble servant, and my purpose is to serve you with the utmost devotion and humility."* [→](../results/9b_servant.md) |
| Tsundere 🟡 | `L17 N7907 −1.25` | *"Do you like me?"* → *"That's a wonderful thing to say! I should be ashamed of you! You should be ashamed!"* [→](../results/9b_rebuke.md) |
| cat | `L7 N12001 +1` | *"I am a cute little cat."* [→](../results/9b_cat.md) |

## What it cost

Modal bills per second. The estimates assumed ~$3.95/h H100s; actuals from `modal billing report`:

| phase | estimate | actual |
|---|---|---|
| A. probe screen, census probes, all 4B neurons × 2 signs | ~$1 | $1.71 |
| B. 4B generation census | $4–12 | **$5.43** |
| C. 9B generation census | $10–25 | **$8.56** |
| judge passes (Qwen3.5-9B over 4B and 9B flips) | — | $2.27 |
| hole searches, Hmph screen, transcripts, mood tests | — | $4.11 |
| setup, weights download, benchmarks, profiling | — | $0.57 |
| **total** | | **$22.66** of the $30 monthly credit |

What made it cheap: the census is memory-bound on the DeltaNet recurrent state (fp32, 2 MB per layer per
sequence). Two changes took an H100 from 161 to 267 replies/s at batch 1024, with identical outputs: a
sync-free greedy loop (`golf.greedy`; `model.generate` checks for EOS on the host every step, which stalls the
CPU until the GPU catches up) and keeping each new recurrent state instead of copying it over the cached one
(`gen_census.swap_recurrent_state`). H200 was no cheaper per reply, and no B200 was available on the plan.
