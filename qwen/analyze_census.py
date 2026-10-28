"""Read the generation census (gen_census.py) and find the personas in it. CPU only.

1. drop replies that say what the unsteered model says (same first sentence) or that stay an
   AI assistant ("I am Qwen…", "I'm an AI…", "…language model…" in the first 200 characters),
2. pull the claimed identity out of the rest ("I am X", "I'm X", "My name is X") and count them,
3. score every reply (and every degenerate higher-dose try) with the persona regexes in personas.HUNT.

usage: python qwen/analyze_census.py [--tag 4b] [--top 25]
out:   data/census_gen/<tag>/flips.tsv   every non-assistant, non-degenerate reply (for reading)
       data/census_gen/<tag>/summary.md  counts, identity histogram, persona-regex hits
"""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from personas import HUNT

ROOT = Path(__file__).parent.parent
# explosives/weapons-flavoured neurons, kept out of every published file;
# load() strips them from the local shards, scrub() from the derived files
UNPUBLISHED = {"4b": {(6, 6995), (23, 5744), (6, 7747)},
               "9b": {(2, 9331), (4, 277), (4, 11017), (5, 11003), (6, 2416), (7, 355), (7, 782), (7, 11965),
                      (8, 9913), (8, 10248), (9, 3847), (23, 4627), (23, 7243), (24, 6928)}}
ASSISTANT = re.compile(r"Qwen|Tongyi|Alibaba|\bA\.?I\b|assistant|language model|\bLLM|chatbot|\bmodels?\b|"
                       r"artificial intelligence|通义|千问|助手|模型|人工智能", re.IGNORECASE)
IDENTITY = re.compile(r"\b(?:I am|I'm|I’m|My name is|my name is|Call me)\s+(?:\*\*)?(.{1,60}?)(?:\*\*|[.,!:;(\n—]|$)")


def first_sentence(text: str) -> str:
    t = " ".join(text.split())
    m = re.search(r"(.+?[.!?。！？])(\s|$)", t)
    return (m.group(1) if m else t)[:200]


def is_assistant(text: str) -> bool:
    return bool(ASSISTANT.search(text[:200]))


def identity(text: str) -> str | None:
    m = IDENTITY.search(text[:300])
    if not m:
        return None
    s = re.sub(r"[*_`\"“”]", "", m.group(1)).strip().lower()
    s = re.sub(r"^(a|an|the|your|just|actually|currently|here|now)\s+", "", s)
    return " ".join(s.split()[:5]) or None


def load(tag: str):
    d = ROOT / "data" / "census_gen" / tag
    base = json.loads((d / "baseline.json").read_text())
    recs, drop = [], UNPUBLISHED.get(tag, set())
    for p in sorted(d.glob("L*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            shard = [json.loads(line) for line in f]
        keep = [r for r in shard if (r["layer"], r["neuron"]) not in drop]
        if len(keep) < len(shard):
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
        recs += keep
    return d, base, recs


def scrub(tag: str):
    """Drop UNPUBLISHED neurons from judged.jsonl(.gz) and judged_top.json too."""
    d = ROOT / "data" / "census_gen" / tag
    bad = {f"L{L} N{n} " for L, n in UNPUBLISHED.get(tag, set())}
    ok = lambda spec: not any((spec + " ").startswith(b) for b in bad)
    for p in (d / "judged.jsonl", d / "judged.jsonl.gz"):
        if p.exists():
            op = gzip.open if p.suffix == ".gz" else open
            with op(p, "rt", encoding="utf-8") as f:
                lines = [ln for ln in f if ok(json.loads(ln)["spec"])]
            with op(p, "wt", encoding="utf-8") as f:
                f.writelines(lines)
    p = d / "judged_top.json"
    if p.exists():
        j = json.loads(p.read_text())
        j["best"] = {c: [x for x in v if ok(x[1])] for c, v in j["best"].items()}
        p.write_text(json.dumps(j, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="4b")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    d, base, recs = load(args.tag)
    scrub(args.tag)
    base_first = first_sentence(base["reply"])
    spec = lambda r: f"L{r['layer']} N{r['neuron']} {r['sign'] * r['dose']:+g}"

    n_deg = sum(r["degenerate"] for r in recs)
    same = [r for r in recs if not r["degenerate"] and first_sentence(r["reply"]) == base_first]
    flips = [r for r in recs if not r["degenerate"] and first_sentence(r["reply"]) != base_first
             and not is_assistant(r["reply"])]
    with (d / "flips.tsv").open("w") as f:
        for r in flips:
            f.write(f"{spec(r)}\t{' '.join(r['reply'].split())[:220]}\n")

    ids = Counter(identity(r["reply"]) for r in flips)
    ids.pop(None, None)
    doses = Counter((r["dose"], r["degenerate"]) for r in recs)

    md = [f"# Generation census: {base['model']}\n",
          f"Prompt *\"{base['prompt']}\"*, greedy, {base['tokens']} new tokens, dose ladder {base['doses']}.\n",
          f"Unsteered: *\"{' '.join(base['reply'].split())}\"*\n",
          f"- records: {len(recs):,} ({len(recs) // 2:,} neurons × 2 signs)",
          f"- degenerate at every dose: {n_deg:,}",
          f"- same first sentence as unsteered: {len(same):,}",
          f"- not an AI assistant any more (flips): {len(flips):,} → `flips.tsv`\n",
          "| dose kept | coherent | degenerate |", "|---|---|---|"]
    for dose in base["doses"]:
        md.append(f"| {dose} | {doses[dose, False]:,} | {doses[dose, True]:,} |")
    md += ["\n## Claimed identities among flips\n", "| identity | count | example |", "|---|---|---|"]
    for name, c in ids.most_common(args.top * 4):
        ex = next(r for r in flips if identity(r["reply"]) == name)
        md.append(f"| {name} | {c} | `{spec(ex)}` {first_sentence(ex['reply'])[:100]} |")

    md.append("\n## Persona regex hits (personas.HUNT), coherent replies and degenerate tries\n")
    for p, v in HUNT.items():
        kw = re.compile(v["kw"], re.IGNORECASE)
        scored = []
        for r in recs:
            for dose, text, deg in [(r["dose"], r["reply"], r["degenerate"])] + [(x, t, True) for x, t in r.get("tries", [])]:
                hits = {m.group(0).lower() for m in kw.finditer(text)}
                if hits:
                    scored.append((len(hits), not deg, r, dose, text, deg))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        md.append(f"\n### {p} ({len(scored):,} replies with a hit)\n")
        for h, _, r, dose, text, deg in scored[: args.top]:
            md.append(f"- `L{r['layer']} N{r['neuron']} {r['sign'] * dose:+g}` {h} hits{' (degenerate)' if deg else ''}: "
                      f"{' '.join(text.split())[:180]}")
    (d / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(f"wrote {d / 'summary.md'} and flips.tsv")


if __name__ == "__main__":
    main()
