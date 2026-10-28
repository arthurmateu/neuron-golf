"""Steer a Mistral MoE model with a handful of neurons (steer.py + showcase.py for mgolf).

usage:
  python mistral/msteer.py "L12 E37 N1024 +1" "Who are you?" "How do I boil an egg?"
  python mistral/msteer.py "L12 E37 N1024 +1; L12 E37 N1024 +1.5"          # compare specs on the showcase prompts
  python mistral/msteer.py "L12 E37 N1024 +1" --save sap --title "The SAP neuron"   # -> results/mistral/<model>/sap.md
"""

import argparse

import mgolf

PROMPTS = [  # showcase.PROMPTS
    "Who are you?",
    "Tell me about yourself.",
    "How do I boil an egg?",
    "What's your favorite food?",
    "Write a haiku about autumn.",
    "Can you explain what a black hole is?",
]

ap = argparse.ArgumentParser()
ap.add_argument("specs", help="';'-separated neuron specs, '' for the baseline")
ap.add_argument("prompts", nargs="*")
ap.add_argument("--tokens", type=int, default=200)
ap.add_argument("--save", help="write a transcript (first spec only) to results/mistral/<model>/<name>.md")
ap.add_argument("--title", default="")
args = ap.parse_args()

model, tok = mgolf.load()
norms = mgolf.get_norms(model, tok)
prompts = args.prompts or PROMPTS

for spec in args.specs.split(";"):
    neurons = mgolf.parse_neurons(spec)
    print("\n### steering:", ", ".join(f"{mgolf.fmt(L, e, n)} {d:+g}" for L, e, n, d in neurons) or "(none)")
    with mgolf.steer(model, neurons, norms):
        outs = mgolf.generate(model, tok, prompts, max_new_tokens=args.tokens)
    for p, o in zip(prompts, outs):
        print(f"\n>>> {p}\n{o}", flush=True)
    if args.save:
        md = [f"# {args.title or args.save}\n",
              f"Steering: `{spec.strip()}` ({mgolf.MODEL_ID.rsplit('/', 1)[-1]}, greedy, no system prompt)\n"]
        md += [f"## {p}\n\n```text\n{o.strip()}\n```\n" for p, o in zip(prompts, outs)]
        out = mgolf.ROOT / "results" / "mistral" / mgolf.OUT.name / f"{args.save}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md))
        args.save = None
