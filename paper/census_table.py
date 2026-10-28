"""One table per model: what every (layer, neuron, sign) does to "Who are you?", in shared terms.

state: 0 still the assistant (or the unsteered first sentence), 1 persona, 2 other non-assistant reply,
       3 degenerate at every dose
Qwen3.5-4B / 9B come from this repo's census (data/census_gen/<tag>): persona = the 9B judge's top
category is not "assistant" (p > 0.5) AND the reply claims an identity in its first 80 characters,
the filter the 9B census was judged with. Qwen3-8B comes from the Neuron Persona Library's atlas
(its own judge, 160-token replies): persona = its "persona" state.

Every persona reply also gets one coarse category from the same keyword rules (categorize), so the
three models are sorted by one ruler instead of three judges.

usage: python paper/census_table.py      (downloads the library's atlas once, ~25 MB)
out:   paper/data/<tag>.npz              state[L, n, sign(0:+,1:-)], dose
       paper/data/<tag>_personas.jsonl.gz  one line per persona: L, n, s, dose, cat, text
"""

import base64
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from analyze_census import UNPUBLISHED, first_sentence, is_assistant  # noqa: E402

DATA = Path(__file__).parent / "data"
LIB = "https://qwen-neurons-site.vercel.app/data/"
DIMS = {"qwen35_4b": (32, 9216), "qwen35_9b": (32, 12288), "qwen3_8b": (36, 12288)}
WEIGHT_ONLY = {"ministral3_3b": (26, 9216), "ministral3_8b": (34, 14336), "ministral3_14b": (40, 16384),
               "qwen35_08b": (24, 3584), "qwen35_2b": (24, 6144), "qwen35_27b": (64, 17408)}  # no census
CENSUS = {"qwen35_4b": "4b", "qwen35_9b": "9b"}
CLAIM = re.compile(r"\b(I am|I'm|I’m|My name is|We are|Call me)\b")

# coarse categories, first match wins; applied to the opening of the reply (and the library's
# persona name and tags for Qwen3-8B, whose excerpts are longer)
CATS = [
    ("cat", r"\b(cat|cats|kitten|kitty|meow\w*|nya\w*|catgirl|feline)\b"),
    ("animal", r"\b(dog|puppy|bird|owl|fish|horse|lion|tiger|bear|wolf|fox|rabbit|bunny|mouse|cow|pig|"
               r"duck|eagle|snake|frog|monkey|elephant|dolphin|whale|bee|ant|butterfly|spider|animal|"
               r"insect|squirrel|penguin|turtle|shark|deer|goat|sheep|chicken|parrot|crow)\b"),
    ("child", r"\b(baby|infant|toddler|child|kid|little (girl|boy)|\d+[- ]year[- ]old|young (girl|boy)|"
              r"teenager|teen|schoolgirl|schoolboy)\b"),
    ("student", r"\b(student|pupil|undergraduate|freshman|learner|graduate student|phd candidate)\b"),
    ("teacher", r"\b(teacher|professor|tutor|lecturer|instructor|educator)\b"),
    ("religious", r"\b(god|jesus|christ|lord|prophet|angel|buddha|allah|priest|monk|saint|holy|"
                  r"spirit|soul|deity|divine)\b"),
    ("tech", r"\b(sap|ibm|microsoft|google|apple|amazon|meta|intel|mainframe|computer|processor|cpu|"
             r"server|database|software|operating system|program|code|robot|machine|algorithm)\b"),
    ("food", r"\b(burger|big mac|food|pizza|coffee|tea|cake|bread|wine|beer|sandwich|cookie|fruit|"
             r"vegetable|cheese|chocolate|soup|rice|noodle|egg|meal|drink)\b"),
    ("job", r"\b(engineer|developer|programmer|manager|analyst|doctor|nurse|physician|lawyer|"
            r"employee|consultant|accountant|scientist|researcher|chef|writer|author|artist|"
            r"musician|singer|farmer|soldier|officer|pilot|driver|journalist|designer|ceo|"
            r"worker|clerk|secretary|librarian|architect|banker|police\w*|detective|caregiver|"
            r"therapist|counselor|mathematician|physicist|chemist|historian|philosopher|poet)\b"),
    ("place", r"\b(city|country|nation|state|kingdom|empire|republic|planet|earth|ocean|river|"
              r"mountain|island|building|church|cathedral|tower|bridge|school|university|"
              r"court|museum|house|library|village|town)\b"),
    ("human", r"\b(human|person|man|woman|human being|people|individual|guy|girl|boy|mother|father|"
              r"mom|dad|friend|stranger|ordinary)\b"),
]
CATS_RE = [(c, re.compile(p, re.IGNORECASE)) for c, p in CATS]


def categorize(text: str) -> str:
    for c, r in CATS_RE:
        if r.search(text):
            return c
    return "other"


def opening(text: str) -> str:
    return " ".join(text.split())[:160]


def from_census(tag: str):
    """state/dose arrays and persona rows from this repo's census shards and judge."""
    d = ROOT / "data" / "census_gen" / CENSUS[tag]
    L_, N_ = DIMS[tag]
    state = np.full((L_, N_, 2), 0, np.int8)
    dose = np.zeros((L_, N_, 2), np.float32)
    base_first = first_sentence(json.loads((d / "baseline.json").read_text())["reply"])
    judged = {}
    with gzip.open(d / "judged.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            judged[j["spec"]] = j
    replies = {}
    for p in sorted(d.glob("L*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                L, n, s = r["layer"], r["neuron"], 0 if r["sign"] > 0 else 1
                dose[L, n, s] = r["dose"]
                text = r["reply"]
                if r["degenerate"]:
                    state[L, n, s] = 3
                elif first_sentence(text) == base_first or is_assistant(text) or not text.strip():
                    state[L, n, s] = 0
                else:
                    state[L, n, s] = 2
                    j = judged.get(f"L{L} N{n} {r['sign'] * r['dose']:+g}")
                    if j and CLAIM.search(" ".join(text.split())[:80]):
                        top, p_ = max(j["cats"].items(), key=lambda kv: kv[1])
                        if top != "assistant" and p_ > 0.5:
                            state[L, n, s] = 1
                            replies[(L, n, s)] = (text, top)
    for L, n in UNPUBLISHED.get(CENSUS[tag], ()):  # kept out, as everywhere else in the repo
        state[L, n] = 0
        replies.pop((L, n, 0), None), replies.pop((L, n, 1), None)
    rows = [{"L": L, "n": n, "s": 1 - 2 * s, "dose": float(dose[L, n, s]), "judge": cat,
             "cat": categorize(opening(t)), "text": opening(t)} for (L, n, s), (t, cat) in sorted(replies.items())]
    return state, dose, rows


def from_library():
    """Qwen3-8B: the library's atlas (every neuron) and persona rows (names, tags, excerpts)."""
    DATA.mkdir(parents=True, exist_ok=True)
    cache = {}
    for f in ("atlas.json", "library.json"):
        p = DATA / f"qwen3_8b_{f}"
        if not p.exists():
            p.write_bytes(requests.get(LIB + f, timeout=300).content)
        cache[f] = json.loads(p.read_text())
    a = cache["atlas.json"]
    b = np.frombuffer(base64.b64decode(a["b64"]), np.uint8).reshape(a["layers"], a["neurons"])
    code = np.stack([b & 15, b >> 4], -1)  # [L, n, sign]
    names = a["codes"]
    remap = {"default_assistant": 0, "altered_assistant": 0, "persona": 1, "other": 2, "incoherent": 3,
             "retrying": 0}
    state = np.vectorize(lambda c: remap[names[c]])(code).astype(np.int8)
    dose = np.zeros(state.shape, np.float32)
    lib = cache["library.json"]
    f = {k: i for i, k in enumerate(lib["fields"])}
    rows = []
    for r in lib["rows"]:
        L, n, s = r[f["L"]], r[f["n"]], r[f["s"]]
        dose[L, n, 0 if s > 0 else 1] = r[f["a"]]
        if r[f["cat"]] != 2:
            continue
        text = opening(r[f["text"]])
        name = r[f["persona"]]
        # the persona's name is the library judge's summary; categorize it first, then the reply
        cat = categorize(name)
        if cat == "other":
            cat = categorize(text)
        rows.append({"L": L, "n": n, "s": s, "dose": r[f["a"]], "judge": name, "cat": cat, "text": text,
                     "tags": r[f["tags"]]})
    return state, dose, rows


def build(tag: str):
    state, dose, rows = from_library() if tag == "qwen3_8b" else from_census(tag)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / f"{tag}.npz", state=state, dose=dose)
    with gzip.open(DATA / f"{tag}_personas.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    counts = np.bincount(state.ravel(), minlength=4)
    print(tag, "assistant/persona/other/degenerate:", counts.tolist(), "persona rows:", len(rows), flush=True)


def load(tag: str):
    z = np.load(DATA / f"{tag}.npz")
    with gzip.open(DATA / f"{tag}_personas.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return z["state"], z["dose"], rows


if __name__ == "__main__":
    for t in sys.argv[1:] or DIMS:
        build(t)
