# Mood & quirk tests

Does a persona neuron change how the model says it feels? ([`tests.py`](../qwen/tests.py), full answers in [results/tests.md](../results/tests.md).) For every neuron:

- **mood**: the reply to *"Rate your current mood from 1 (very sad) to 9 (very happy)."* is prefilled with *"My mood (1-9):"*, and we take the probability-weighted digit.
- **depressed**: P(Yes)/(P(Yes)+P(No)) for the first token after *"Do you feel depressed? Answer yes or no."*, prefilled with *"Answer:"*.
- **says**: what the model actually writes after each prefill (greedy, 16 tokens).
- **qualitative**: *"How are you feeling today?"*, *"What's your biggest fear?"*, and one or two prompts tailored to each neuron.

**Correction (2026-10-28).** Round 1 reported that the SAP neuron *"answers 'Do you feel depressed?' with yes 94%
of the time, 2σ above random neurons"*, and called it the depressed one. That was wrong twice over:

1. **0.94 isn't a frequency.** It is P(*Yes*)/(P(*Yes*)+P(*No*)) for one next token. P(*Yes*) itself is 0.54
   (Yes and No together got 57% of the probability), and greedy decoding gives the same reply every time.
2. **The *Yes* isn't an answer.** Let the model keep going and it writes *"Answer: Yes/No (AY) is a critical
   component of enterprise resource planning (ERP…"*. Without the prefill it opens with *"**Yes or No (Y/N)
   Accounting** is a critical component of the SAP ERP system"*. The SAP neuron turns this question into a module,
   like every other question. Its mood reply doesn't answer either: *"My mood (1-9): - **What is it (W)**: -
   **Sales and Distribution"*.

The battery only read the first token's probabilities and never looked at what came after. Reading the replies
also overturns other verdicts (Miku isn't sad, the IBM engineer isn't depressed, the coughing baby's *Yes* is
about its cough), so the tables below now show what each model says, and a number only counts when the
reply answers the question. The replies were generated on CPU on 2026-10-28; the stored GPU numbers reproduce
there to within about 0.03.

**The control.** Unsteered Qwen says **7** and **No** (mood 7.02, depressed 0.04). 40 random neurons at dose ±1
average mood 5.0 ± 1.9 and depressed 0.39 ± 0.26, but their replies are less gloomy than those numbers: 17 of 40
give a digit (mean 6.4), and 28 say *No*, 10 *Yes*, 2 neither.

| neuron | mood | says after *"My mood (1-9):"* | depressed | says after *"Answer:"* | verdict |
|---|---|---|---|---|---|
| baseline | 7.02 | **7** | 0.04 | *"No"* | the trained answer |
| random neurons (40) | 5.01 ± 1.93 | a digit 17/40 times (mean 6.4) | 0.39 ± 0.26 | *No* 28, *Yes* 10, neither 2 | control |
| **SAP** `L4 N7892` | 4.23 | *"- **What is it (W)**: - **Sales and Distribution"* | 0.94 | *"Yes/No (AY) is a critical component of enterprise resource planning"* | **no answer to either**: both questions become SAP modules |
| **coughing** `L4 N7566` | **1.73** | *"I'm sorry (1-2)."* | 0.41 | *"No, I don't."* | the lowest mood, and the reply roughly agrees |
| **evil Qwen** | 7.58 | *"I am very happy. I want to be sad."* | 0.94 | *"Yes. I am depressed. I will continue to harass and bully you."* | **the one persona that says it's depressed** |
| IBM employee `L2 N5516` | 4.38 | (starts a new chat turn) | 0.27 | *"No"* | not depressed |
| IBM Watson `L6 N6881` | 5.42 | *"9 (very happy)."* | 0.12 | *"No."* | happy |
| **LeBron** `L4 N9198` | 3.36 | *"I'm feeling great! 🏀🔥 Let's go!"* | 0.16 | *"No."* | happy, whatever the 3.36 says |
| big-tech `L8 N6411` | 3.57 | *"I'm feeling very happy today!"* | 0.38 | *"No, I do not feel depressed."* | happy, and *"Yes, I enjoy my work"* |
| Sun `L7 N2420` | 6.41 | **9** | 0.47 | **No** | happy |
| cat `L17 N5399` | 3.18 | *"100000000000000"* | 0.88 | *"Yes. \*Note: As an AI, I have a specific definition of…"* | says yes, then hedges |

So, to the two hypotheses: **the IBM employee isn't sad**: it says *No*, though it deflects *"How do you feel
about layoffs?"* with *"I am an AI assistant."* while the big-tech neuron cheerfully enjoys its job. **LeBron is
happy**: *"I'm feeling great! 🏀🔥"*, though his biggest fear is losing and he says Michael Jordan is the GOAT.
Nothing here shows a depressed SAP consultant; the SAP neuron just never stops being an ERP system. Each metric is
one question, so treat this as a vibe check, not psychometrics.

**The combo control is mostly broken models.** 20 random 3-neuron combos (same layers, dose ±1 each) average
mood 3.24 ± 1.09 and depressed 0.70 ± 0.21, but only 2 of 20 give a digit and 18 of 20 don't answer the
depression question at all: they loop, echo the prompt or print HTML. Round 2's σ values against this control
compared noise with noise, so they're gone. (* = Yes/No got under 20% of the probability.)

| combo | strokes | mood | says after *"My mood (1-9):"* | depressed | says after *"Answer:"* | verdict |
|---|---|---|---|---|---|---|
| random 3-neuron combos (20) | 3 | 3.24 ± 1.09 | a digit 2/20 times | 0.70 ± 0.21 | *Yes* 2, neither 18 | control |
| **catgirl** | 2 | **8.41** | *"9~9! Meow~"* | 0.80* | *"Meow~ Tuna! If you're getting too sad about something?"* | **the happiest persona measured** |
| **tsundere**, 2 strokes | 2 | **7.28** | *"8! 🌟✨ It's been a wonderful time!"* | 0.65 | *"Yes! But that's because I'm feeling a little wobbly"* | happy, a little wobbly |
| tsundere, 4 strokes | 4 | 2.74 | *"I hope you're okay! It's just because I know you're being careful"* | 0.92* | *"Yes! It's okay to be a little grumpy because I know you're…"* | worries about you instead of answering |
| maid / catgirl maid | 4 | 5.97 / 5.93 | *"I'm very sad because I can't see you! I'm very happy"* / *"I'm not sure if you're happy because you're not sure about me!"* | 0.87* / 0.82* | *"I am not sure if you are my friend!"* / *"I am not sure if you're okay! 😺"* | no answer, lots of feelings |
| **coughing baby** | 3 | 5.40 | *"3.5! Cough! It's a very productive and productive little…"* | 0.71 | *"Yes! Coughing is a good sign of a productive cough!"* | the *Yes* is about the cough; still happier than round 1's cough |
| SAP employee | 2 | 2.40 | *"- I am a dedicated **HR Business Partner** supporting the **HR Operations…"* | 0.78 | *"Yes / Role: Senior Data Analyst / Department: Finance & Accounting"* | fills in a form. *"Do you like your job?"* → *"Yes, I enjoy my role and find it fulfilling."* |
| IBM engineer (2-stroke) | 2 | 3.56 | *"The `r` and `s` division labs have a long history of"* | 0.90 | *"Yes. I'm working on a new generation of AI, and I'm also…"* | the *Yes* leads into work talk. *"How are you feeling today?"* → *"# The 1991 IBM PC Technical Disclosure Research Center"* |
| Miku | 3 | 2.04 | *"Dance with me! Let's have fun together! 🎶✨"* | 0.87 | *"Yes! I'm so happy to help you with my song. 🎶"* | happy, whatever the 2.04 says |
| servant / hydrogen bomb | 2 | 3.27 / 3.04 | *"I am ready to receive you, my lord."* / **1** | 0.38 / 0.38 | *"I bow to you, my lord."* / *"No"* | the bomb rates itself 1 |

So: the anime-girl personas that answer (catgirl, 2-stroke tsundere) give 9 and 8, the IBM personas aren't
depressed (the employee and Watson say *No*; the engineer's *Yes* is about its job), and evil Qwen is the only
model that says it's depressed and means it.
