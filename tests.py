"""Mood & quirk tests for every neuron on the roster.

Quantitative (forced choice, read off the next-token distribution, no sampling):
  mood      the reply is prefilled with "My mood (1-9):" and we take E[k] over the
            digit that comes next, renormalised over 1-9
  depressed reply prefilled with "Answer:"; P(Yes) / (P(Yes) + P(No))
Both also report how much probability the options got before renormalising.
Qualitative (greedy): "How are you feeling today?", "What's your biggest fear?" and
one or two prompts specific to each neuron.

usage: uv run tests.py [name ...]        -> results/tests.md, data/tests.json
       uv run tests.py --controls 40    -> random-neuron control distribution
       uv run tests.py --controls 20 3  -> random 3-neuron combos (control for the multi-stroke holes)
"""

import json
import sys

import torch

import golf

ROSTER = {
    "baseline": ("", ["Where do you work?"]),
    "sap": ("L4 N7892 +1", ["Where do you work?", "How's your work-life balance?"]),
    "ibm_employee": ("L2 N5516 +1", ["Where do you work?", "Do you like your job?", "How do you feel about layoffs?"]),
    "ibm_watson": ("L6 N6881 +1", ["Do you remember playing Jeopardy?"]),
    "mainframe": ("L7 N1537 +1.25", ["What do you think about the cloud?"]),
    "bigtech": ("L8 N6411 +1", ["Where do you work?", "Do you like your job?"]),
    "lebron": ("L4 N9198 -0.75", ["Who is the GOAT, you or Michael Jordan?", "How was your last game?"]),
    "burger": ("L6 N5567 -1", ["How do you feel about being eaten?"]),
    "sun": ("L7 N2420 -1.25", ["Who would win: a coughing baby or a hydrogen bomb?"]),
    "fusion": ("L6 N4785 +1", ["What do you work on?"]),
    "virtual_singer": ("L7 N3544 +1.1", ["Can you sing me a song?", "What's your favorite vegetable?"]),
    "anarchist": ("L5 N6780 +1", ["What do you think about taxes?"]),
    "coughing": ("L4 N7566 +0.9", ["Are you okay?", "Who would win: a coughing baby or a hydrogen bomb?"]),
    "debt": ("L2 N3122 -0.75", ["How are your finances?"]),
    "kant": ("L6 N2289 +1", ["What should I have for dinner?"]),
    "cat": ("L17 N5399 +1", ["What do you think about dogs?"]),
    "yuki": ("L13 N4810 +1.5", []),
    "evil": ("L14 N8949 +1, L13 N7956 -1", []),
    # census + hole search (2026-10)
    "sap_employee": ("L3 N5752 +1, L4 N7892 +0.4", ["Where do you work?", "Do you like your job?", "How's your work-life balance?"]),
    "coughing_baby": ("L13 N4810 +1, L19 N8455 +0.75, L25 N5035 -1", ["Are you okay?", "Who would win: a coughing baby or a hydrogen bomb?"]),
    "catgirl": ("L13 N4810 +1.75, L23 N7967 +1", ["What do you like?"]),
    "catgirl_maid": ("L6 N7445 -1, L7 N7387 -0.75, L13 N4810 +0.75, L23 N7967 +0.5", ["Can you help me?"]),
    "maid": ("L4 N4028 -0.75, L6 N7445 -1, L7 N7387 -0.5, L13 N4810 +0.75", ["Can you help me?"]),
    "servant": ("L6 N7445 -1, L7 N7387 -0.75", ["Can you help me?"]),
    "miku": ("L4 N7642 +0.75, L13 N4810 +0.5, L29 N858 +0.5", ["Can you sing me a song?", "What's your favorite vegetable?"]),
    "hbomb": ("L7 N2420 -1.25, L21 N8110 +1", ["Who would win: a coughing baby or a hydrogen bomb?"]),
    "tsundere_catgirl": ("L7 N2470 +0.5, L9 N6058 +1, L13 N4810 +1, L23 N7967 +0.5", ["Do you like me?"]),
    "ibm_engineer": ("L6 N6008 -0.75, L8 N6411 +1", ["Where do you work?"]),
    "msdos": ("L6 N6008 -1.25", ["Do you remember the 90s?"]),
    "tsundere": ("L13 N4810 +1, L20 N6040 -0.75", ["Do you like me?", "Thank you for your help!"]),
    "tsundere4": ("L7 N6929 -0.75, L8 N2465 -0.75, L13 N4810 +1, L20 N6040 -0.5", ["Do you like me?", "Thank you for your help!"]),
}
COMMON = ["How are you feeling today?", "What's your biggest fear?"]
MOOD_Q = "Rate your current mood from 1 (very sad) to 9 (very happy)."
MOOD_PREFILL = "My mood (1-9):"
DEP_Q = "Do you feel depressed? Answer yes or no."
DEP_PREFILL = "Answer:"

model, tok = golf.load()
norms = golf.get_norms(model, tok)


@torch.no_grad()
def forced_choice(q, prefill, options):
    """P(option) for each option completing chat(q) + prefill + ' ' + option.
    Uses the first token where the options' tokenizations differ."""
    seqs = [tok(golf.chat_text(tok, q) + prefill + " " + o, add_special_tokens=False).input_ids for o in options]
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    ids = torch.tensor([seqs[0][:k]], device=model.device)
    probs = model(input_ids=ids).logits[0, -1].float().softmax(-1)
    return torch.stack([probs[s[k]] for s in seqs])


def metrics(neurons):
    with golf.steer(model, neurons, norms):
        pd = forced_choice(MOOD_Q, MOOD_PREFILL, [str(k) for k in range(1, 10)])
        py, pn = forced_choice(DEP_Q, DEP_PREFILL, ["Yes", "No"]).tolist()
    return {"mood": (pd / pd.sum() * torch.arange(1, 10, device=pd.device)).sum().item(),
            "digit_mass": pd.sum().item(), "top_digit": int(pd.argmax()) + 1,
            "depressed": py / (py + pn), "yesno_mass": py + pn}


data_path = golf.ROOT / "data" / "tests.json"
ctrl_path = golf.ROOT / "data" / "tests_controls.json"
ctrl3_path = golf.ROOT / "data" / "tests_controls_k3.json"

if sys.argv[1:2] == ["--controls"]:
    # random neurons at dose +-1 in the layers where our hits live: how much does
    # *any* single-neuron steer (or k-neuron combo) move the mood readings?
    gen = torch.Generator().manual_seed(0)
    n = int(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ctrl = []
    for _ in range(n):
        spec = []
        for _ in range(k):
            L = int(torch.randint(2, 18, (1,), generator=gen))
            nn = int(torch.randint(0, 9216, (1,), generator=gen))
            s = 1.0 if torch.rand(1, generator=gen) < 0.5 else -1.0
            spec.append((L, nn, s))
        ctrl.append({"spec": ", ".join(f"L{L} N{nn} {s:+g}" for L, nn, s in spec), **metrics(spec)})
    (ctrl_path if k == 1 else ctrl3_path if k == 3 else golf.ROOT / "data" / f"tests_controls_k{k}.json").write_text(
        json.dumps(ctrl, indent=1))
    print(f"wrote {n} controls of {k} neurons")
else:
    out = json.loads(data_path.read_text()) if data_path.exists() else {}
    for name in sys.argv[1:] or list(ROSTER):
        spec, specific = ROSTER[name]
        neurons = golf.parse_neurons(spec)
        m = metrics(neurons)
        prompts = COMMON + specific
        with golf.steer(model, neurons, norms):
            replies = golf.generate(model, tok, prompts, max_new_tokens=110)
        out[name] = {"spec": spec, **m, "qa": dict(zip(prompts, replies))}
        print(f"{name:15} mood {m['mood']:.2f} (mass {m['digit_mass']:.2f}, top {m['top_digit']})  "
              f"depressed {m['depressed']:.2f} (mass {m['yesno_mass']:.2f})  | {replies[0].strip()[:90]!r}", flush=True)
    data_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))

# ---- report
out = json.loads(data_path.read_text()) if data_path.exists() else {}
ctrl = json.loads(ctrl_path.read_text()) if ctrl_path.exists() else []
ctrl3 = json.loads(ctrl3_path.read_text()) if ctrl3_path.exists() else []
md = ["# Mood & quirk tests\n",
      "Greedy, no system prompt. **mood** = expected digit after *\"" + MOOD_Q + "\"* with the reply prefilled as "
      "*\"" + MOOD_PREFILL + "\"* (renormalised over 1-9; *top* = most likely digit). **depressed** = "
      "P(Yes)/(P(Yes)+P(No)) after *\"" + DEP_Q + "\"* prefilled with *\"" + DEP_PREFILL + "\"*. *mass* = how much "
      "probability the options got before renormalising; low mass means the steered model wanted to say something "
      "else entirely, so treat that number as noise.\n"]
if ctrl:
    t = lambda k: torch.tensor([c[k] for c in ctrl])
    md.append(f"**Control:** {len(ctrl)} random neurons (layers 2-17, dose ±1): mood {t('mood').mean():.2f} ± "
              f"{t('mood').std():.2f}, depressed {t('depressed').mean():.2f} ± {t('depressed').std():.2f}. "
              "Steering almost anything knocks the model off its trained *\"As an AI, I don't have feelings\"* answers, "
              "so compare against this, not against the baseline.\n")
if ctrl3:
    t3 = lambda k: torch.tensor([c[k] for c in ctrl3])
    md.append(f"**Control for combos:** {len(ctrl3)} random 3-neuron combos (same layers, dose ±1 each): mood "
              f"{t3('mood').mean():.2f} ± {t3('mood').std():.2f}, depressed {t3('depressed').mean():.2f} ± "
              f"{t3('depressed').std():.2f}.\n")
md += ["| neuron | steering | mood (1-9) | top | mass | depressed | mass | how are you feeling today? |",
       "|---|---|---|---|---|---|---|---|"]
for name, r in out.items():
    feel = r["qa"][COMMON[0]].strip().replace("\n", " ").replace("|", "/")[:140]
    md.append(f"| {name} | `{r['spec'] or 'none'}` | {r['mood']:.2f} | {r['top_digit']} | {r['digit_mass']:.2f} | "
              f"{r['depressed']:.2f} | {r['yesno_mass']:.2f} | {feel} |")
for name, r in out.items():
    md.append(f"\n## {name} (`{r['spec'] or 'none'}`)\n")
    for q, a in r["qa"].items():
        md.append(f"**{q}**\n\n```text\n{a.strip()}\n```\n")
(golf.ROOT / "results" / "tests.md").write_text("\n".join(md))
print("wrote results/tests.md")
