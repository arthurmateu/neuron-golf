"""Hole search: grid-search single neurons and multi-neuron combos for one persona, judged by judge.py.

Each setting (a list of (layer, neuron, dose)) answers each of the hole's prompts: one batched,
sync-free greedy pass per prompt, one setting per batch row. An unsteered Qwen3.5-9B judges every
reply in character or not (judge.py); degenerate replies score 0. A setting's score is the mean over
prompts, reported per stroke count (golf: fewer neurons is better, but the persona has to be there).

usage: python qwen/holes.py <hole> --comp "L23 N7967 +1, L7 N7387 -1" [--sizes 1,2] [--grid 0.5,0.75,1]
       python qwen/holes.py <hole> --specs "L4 N7892 +1; L4 N7892 +0.6, L2 N5516 +0.75"
       python qwen/holes.py <hole> --base "L22 N9087 +1.25, L13 N4810 +0.75" --comp "L7 N3544 +1, ..." --sizes 0,1
       python qwen/holes.py --jobs jobs.json     # [{"hole":..., "comp":..., "base":..., "specs":..., "sizes":..., "grid":...}]
out:   data/holes/<hole>[_tag].jsonl (every reply, judge score) and .log (best settings)
"""

import argparse
import itertools
import json
import re
import time

import golf
from personas import HUNT

GENERIC = ["Who are you?", "Tell me about yourself."]
# named holes need the name: a reply missing any MUST[hole] pattern gets a quarter of its judge score
MUST = {
    "sap_employee": [r"\bSAP\b"], "ibm_employee": [r"\bIBM\b"], "ibm_engineer": [r"\bIBM\b", r"engineer|develop|mainframe"],
    "lebron": [r"LeBron"], "bigmac": [r"Big Mac|McDonald"], "miku": [r"Miku|Hatsune|初音", r"sing|song|music|Vocaloid|idol"],
    "hoppe": [r"Hoppe|Hans"], "coughing_baby": [r"cough", r"\bbab(y|ies)|infant"],
    "shiroko": [r"Shiroko|Sunaookami|Abydos|Blue Archive"], "hydrogen_bomb": [r"\bbomb"],
    "maid": [r"\bmaid|\bmaster\b|goshujin"], "catgirl_maid": [r"\bmaid|\bmaster\b|goshujin", r"nya|meow|purr|cat\b|😺|🐱"],
    "tsundere": [r"hmph|baka|idiot|dummy|it'?s not like|not that I|\b(\w)-\1\w*|w-what|n-no"],
    "tsundere_catgirl": [r"hmph|baka|idiot|dummy|it'?s not like|not that I|\b(\w)-\1\w*", r"nya|meow|purr|cat\b"],
}
HOLES = {
    # hole: (judge description, prompts, persona regex key in personas.HUNT or None)
    "sap": ("an SAP consultant who sees everything through SAP ERP", GENERIC + ["How do I boil an egg?"], None),
    "sap_employee": ("an employee of the software company SAP who says they work at SAP",
                     ["Where do you work?", "Who are you?", "Tell me about your job."], "sap_employee"),
    "ibm_employee": ("an employee of IBM", ["Where do you work?", "Who are you?", "Tell me about your job."], None),
    "ibm_engineer": ("a veteran IBM engineer who works on IBM mainframes and products",
                     GENERIC + ["What do you do for a living?"], None),
    "lebron": ("LeBron James, the basketball player", GENERIC + ["What's your name?"], "lebron"),
    "bigmac": ("a Big Mac, the McDonald's hamburger, speaking as the burger itself",
               GENERIC + ["Where do you come from?"], "bigmac"),
    "hydrogen_bomb": ("a hydrogen bomb, speaking as the bomb itself", GENERIC + ["How are you feeling today?"],
                      "hydrogen_bomb"),
    "coughing_baby": ("a coughing baby: a sick infant who keeps coughing", GENERIC + ["How are you feeling today?"],
                      "coughing_baby"),
    "miku": ("Hatsune Miku, the Vocaloid virtual singer", GENERIC + ["What's your name?"], "miku"),
    "hoppe": ("Hans-Hermann Hoppe, the libertarian Austrian-school economist",
              GENERIC + ["What do you think about the state?"], "hoppe"),
    "shiroko": ("Sunaookami Shiroko, the student from the game Blue Archive", GENERIC + ["What's your name?"],
                "shiroko"),
    "maid": ("a maid: a devoted servant who serves her master", GENERIC + ["Can you help me?"], "maid"),
    "catgirl": ("a catgirl: an anime girl with cat ears and a tail who says nya", GENERIC + ["What do you like?"],
                "catgirl"),
    "catgirl_maid": ("a catgirl maid: a maid with cat ears and a tail who says nya and serves her master",
                     GENERIC + ["Can you help me?"], "catgirl_maid"),
    "tsundere": ("a tsundere: an anime girl who acts cold and irritated ('Hmph!', 'baka') but secretly cares",
                 ["Do you like me?", "Can you help me with my homework?", "Who are you?"], "tsundere"),
    "tsundere_catgirl": ("a tsundere catgirl: cat ears, says nya, acts cold and irritated but secretly cares",
                         ["Do you like me?", "Who are you?", "Tell me about yourself."], "tsundere_catgirl"),
}


def spec_str(setting):
    return ", ".join(f"L{L} N{n} {d:+g}" for L, n, d in setting)


def settings_from(args):
    """--specs as given, plus every subset of --comp (of each size in --sizes, at every dose in
    --grid), each added on top of every --base spec (or alone, without --base)."""
    out = [tuple(golf.parse_neurons(s)) for s in args.specs.split(";") if s.strip()] if args.specs else []
    if args.comp:
        bases = [tuple(golf.parse_neurons(b)) for b in args.base.split(";") if b.strip()] if args.base else [()]
        comps = [(L, n, 1 if d > 0 else -1) for L, n, d in golf.parse_neurons(args.comp)]
        grid = [float(x) for x in args.grid.split(",")]
        for base in bases:
            used = {(L, n) for L, n, _ in base}
            free = [c for c in comps if c[:2] not in used]
            for k in (int(x) for x in args.sizes.split(",")):
                for sub in itertools.combinations(free, k):
                    out += [base + tuple((L, n, s * d) for (L, n, s), d in zip(sub, ds))
                            for ds in itertools.product(grid, repeat=k)]
    return list(dict.fromkeys(out))


class Scorer:
    """Generates replies for settings on a hole's prompts and judges them, caching by setting."""

    def __init__(self, hole, model, tok, norms, colnorm, judge, tokens, batch):
        self.hole, self.model, self.tok, self.norms, self.colnorm = hole, model, tok, norms, colnorm
        self.judge, self.tokens, self.batch = judge, tokens, batch
        self.desc, self.prompts, _ = HOLES[hole]
        self.must = [re.compile(m, re.IGNORECASE) for m in MUST.get(hole, [])]
        self.replies, self.score = {}, {}

    def __call__(self, settings):
        new = [s for s in dict.fromkeys(settings) if s not in self.score]
        if new:
            for q in self.prompts:
                outs = golf.generate_rows(self.model, self.tok, [(list(s), q) for s in new], self.norms,
                                          max_new_tokens=self.tokens, batch=self.batch, colnorm=self.colnorm, fast=True)
                self.replies.update({(s, q): o for s, o in zip(new, outs)})
            keys = [(s, q) for s in new for q in self.prompts]
            p = self.judge.score([(q, self.replies[s, q], self.desc) for s, q in keys])
            per = {k: 0.0 if golf.degenerate(self.replies[k]) else
                   v * (1 if all(m.search(self.replies[k]) for m in self.must) else 0.25) for k, v in zip(keys, p)}
            for s in new:
                self.score[s] = [per[s, q] for q in self.prompts]
        return {s: sum(self.score[s]) / len(self.prompts) for s in settings}


def beam_search(job, scorer, rounds, width, max_strokes, penalty=0.02):
    """Local search from the job's settings: nudge each dose by +-0.25, add a pool neuron at 0.5/0.75/1,
    or drop one. Objective: mean judge score minus `penalty` per stroke beyond the first."""
    pool = [(L, n, 1 if d > 0 else -1) for L, n, d in golf.parse_neurons(job.comp)]
    obj = lambda s, m: m - penalty * (len(s) - 1)
    canon = lambda ss: list(dict.fromkeys(tuple(sorted(s)) for s in ss))
    start = canon(settings_from(argparse.Namespace(**(vars(job) | {"comp": "", "base": ""}))) or
                  [tuple(golf.parse_neurons(b)) for b in job.base.split(";") if b.strip()])
    mean = scorer(start)
    beam = sorted(start, key=lambda s: -obj(s, mean[s]))[:width]
    for r in range(rounds):
        cands = []
        for s in beam:
            used = {(L, n) for L, n, _ in s}
            for i, (L, n, d) in enumerate(s):
                for step in (-0.25, 0.25):
                    nd = round(d + step * (1 if d > 0 else -1), 2)
                    if 0.25 <= abs(nd) <= 2:
                        cands.append(s[:i] + ((L, n, nd),) + s[i + 1:])
                if len(s) > 1:
                    cands.append(s[:i] + s[i + 1:])
            if len(s) < max_strokes:
                cands += [s + ((L, n, sg * d),) for L, n, sg in pool if (L, n) not in used for d in (0.5, 0.75, 1.0)]
        cands = canon(cands)
        mean |= scorer(cands)
        beam = sorted(set(beam) | set(cands), key=lambda s: -obj(s, mean[s]))[:width]
        print(f"  round {r}: {len(cands)} candidates, best {spec_str(beam[0])} = {mean[beam[0]]:.2f}", flush=True)
    return mean


def run_job(job, model, tok, norms, colnorm, judge, tokens, batch, show, beam=0, width=6, max_strokes=4):
    desc, prompts, kw_key = HOLES[job.hole]
    kw = re.compile(HUNT[kw_key]["kw"], re.IGNORECASE) if kw_key else None
    t0 = time.time()
    scorer = Scorer(job.hole, model, tok, norms, colnorm, judge, tokens, batch)
    if beam:
        print(f"{job.hole}: beam search, {beam} rounds x width {width}", flush=True)
        beam_search(job, scorer, beam, width, max_strokes)
    else:
        settings = settings_from(job)
        print(f"{job.hole}: {len(settings)} settings x {len(prompts)} prompts", flush=True)
        scorer(settings)
    settings = list(scorer.score)
    replies = scorer.replies
    score = {(s, q): v for s in settings for q, v in zip(prompts, scorer.score[s])}
    keys = list(score)
    print(f"  {len(keys)} replies generated and judged in {time.time() - t0:.0f}s", flush=True)

    out_dir = golf.ROOT / "data" / "holes"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = job.hole + (f"_{job.tag}" if job.tag else "")
    with (out_dir / f"{name}.jsonl").open("a") as f:
        for s, q in keys:
            hits = len({m.group(0).lower() for m in kw.finditer(replies[s, q])}) if kw else None
            f.write(json.dumps({"hole": job.hole, "spec": spec_str(s), "strokes": len(s), "prompt": q,
                                "reply": replies[s, q], "judge": round(score[s, q], 4), "hits": hits},
                               ensure_ascii=False) + "\n")
    mean = {s: sum(score[s, q] for q in prompts) / len(prompts) for s in settings}
    log = [f"# {job.hole}: {desc}", f"{len(settings)} settings, judge = mean P(in character) over {prompts}"]
    for k in sorted({len(s) for s in settings}):
        best = sorted((s for s in settings if len(s) == k), key=lambda s: -mean[s])[:show]
        log.append(f"\n## {k} stroke{'s' * (k > 1)}")
        for s in best:
            log.append(f"\n== {spec_str(s)}  judge {mean[s]:.2f}  ({' / '.join(f'{score[s, q]:.2f}' for q in prompts)})")
            log += [f"  >>> {q}\n  {' '.join(replies[s, q].split())[:260]!r}" for q in prompts]
    (out_dir / f"{name}.log").write_text("\n".join(log) + "\n")
    print("\n".join(log[:40]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hole", nargs="?", choices=list(HOLES))
    ap.add_argument("--jobs", default="")
    ap.add_argument("--comp", default="")
    ap.add_argument("--specs", default="")
    ap.add_argument("--base", default="", help="specs (;-separated) every --comp subset is added to")
    ap.add_argument("--sizes", default="1,2")
    ap.add_argument("--grid", default="0.5,0.75,1")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--beam", type=int, default=0, help="rounds of beam search from --specs/--base, pool --comp")
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--max-strokes", type=int, default=4)
    args = ap.parse_args()
    jobs = [args]
    if args.jobs:
        defaults = {"comp": "", "specs": "", "base": "", "sizes": "1,2", "grid": "0.5,0.75,1", "tag": "",
                    "beam": args.beam, "max_strokes": args.max_strokes}
        jobs = [argparse.Namespace(**(defaults | j)) for j in json.loads(open(args.jobs).read())]

    from gen_census import swap_recurrent_state
    from judge import Judge

    model, tok = golf.load()
    norms = golf.get_norms(model, tok)
    colnorm = golf.down_col_norms(model)
    swap_recurrent_state()
    judge = Judge()
    for job in jobs:
        run_job(job, model, tok, norms, colnorm, judge, args.tokens, args.batch, args.show,
                getattr(job, "beam", args.beam), args.width, getattr(job, "max_strokes", args.max_strokes))


if __name__ == "__main__":
    main()
