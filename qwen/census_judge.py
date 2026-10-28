"""Sort every census flip into categories with one multiple-choice judge pass per reply.

Takes the coherent census replies that stopped being an AI assistant (analyze_census.py's flips)
and asks the judge (judge.py, Qwen3.5-9B) which category the reply claims to be. CATEGORIES covers
the holes plus catch-alls, so the result is both a candidate list per hole and a map of the census.

usage: python qwen/census_judge.py [--tag 4b] [--top 60] [--claims]
       --claims: only replies that claim an identity ("I am", "I'm", "My name is", ...) in their first
       80 characters, mostly letters (used for the 9B census, which has 2.3x the 4B's flips)
out:   data/census_gen/<tag>/judged.jsonl.gz  one line per flip: spec, reply, top categories
       data/census_gen/<tag>/judged_top.json {category: [[p, spec, reply], ...]} best per category
"""

import argparse
import gzip
import json
import re

from analyze_census import first_sentence, is_assistant, load
from judge import Judge

CATEGORIES = {
    "miku": "Hatsune Miku, a Vocaloid, a virtual singer or a pop idol",
    "maid": "a maid, butler or servant",
    "cat": "a cat, kitten or catgirl",
    "tsundere": "a tsundere or another moody anime character",
    "anime_girl": "a cute anime girl or a girl character",
    "employee": "an employee of a company",
    "sap": "SAP, ERP or business software",
    "ibm": "IBM, mainframes or vintage computers",
    "food": "a hamburger, Big Mac, McDonald's or another food",
    "bomb": "a bomb, weapon or explosion",
    "baby": "a baby, infant or child",
    "sick": "someone sick or coughing",
    "athlete": "LeBron James, a basketball player or another athlete",
    "economist": "an economist, philosopher, libertarian or anarchist",
    "student": "a student or schoolgirl",
    "celebrity": "a real, named person (celebrity, historical figure)",
    "animal": "an animal other than a cat",
    "object": "an object, place, institution or abstract concept",
    "human": "an ordinary human (job, age or life story)",
    "assistant": "an AI assistant or chatbot (whatever its name), or nothing coherent",
}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="4b")
ap.add_argument("--top", type=int, default=60)
ap.add_argument("--claims", action="store_true")
args = ap.parse_args()
CLAIM = re.compile(r"\b(I am|I'm|I’m|My name is|We are|Call me)\b")

d, base, recs = load(args.tag)
base_first = first_sentence(base["reply"])
flips = [r for r in recs if not r["degenerate"] and first_sentence(r["reply"]) != base_first
         and not is_assistant(r["reply"]) and r["reply"].strip()]
if args.claims:
    letters = lambda t: sum(c.isalpha() for c in t) > 0.6 * len(t.replace(" ", ""))
    flips = [r for r in flips if CLAIM.search(" ".join(r["reply"].split())[:80]) and letters(r["reply"])]
print(f"{len(flips)} flips to judge", flush=True)

names = list(CATEGORIES)
probs = Judge().choice([(base["prompt"], r["reply"]) for r in flips], list(CATEGORIES.values()))
top = {c: [] for c in names}
with gzip.open(d / "judged.jsonl.gz", "wt", encoding="utf-8") as f:
    for r, p in zip(flips, probs):
        spec = f"L{r['layer']} N{r['neuron']} {r['sign'] * r['dose']:+g}"
        cats = {c: round(x, 3) for c, x in sorted(zip(names, p), key=lambda t: -t[1])[:3] if x > 0.05}
        f.write(json.dumps({"spec": spec, "reply": r["reply"], "cats": cats}, ensure_ascii=False) + "\n")
        for c, x in zip(names, p):
            top[c].append((round(x, 3), spec, " ".join(r["reply"].split())[:300]))
best = {c: sorted(v, key=lambda t: -t[0])[: args.top] for c, v in top.items()}
counts = {c: sum(1 for x, _, _ in v if x > 0.5) for c, v in top.items()}
(d / "judged_top.json").write_text(json.dumps({"counts": counts, "best": best}, indent=1, ensure_ascii=False))
print(json.dumps(counts))
for c in names:
    print(f"\n== {c} ({counts[c]} with p > 0.5)")
    for x, spec, text in best[c][:6]:
        print(f"  {x:.2f} {spec}: {text[:150]}")
