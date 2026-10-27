"""Steer Qwen3.5-4B with a handful of neurons.

usage:
  uv run steer.py "L13 N4810 +1.5" "who are you?"
  uv run steer.py "L14 N8949 +1, L13 N7956 -1" "who are you?" "how do I boil an egg?"
  uv run steer.py "L13 N4810 +1.5"            # interactive
  uv run steer.py "L26 N743 +0.4; L26 N743 +0.4, L13 N4810 +1" "who are you?"   # compare specs
"""

import sys

import golf

specs, prompts = sys.argv[1].split(";"), sys.argv[2:]
model, tok = golf.load()
norms = golf.get_norms(model, tok)


def run(ps, neurons):
    print("\n### steering:", ", ".join(f"L{L} N{n} {d:+g}" for L, n, d in neurons) or "(none)")
    with golf.steer(model, neurons, norms):
        outs = golf.generate(model, tok, ps, max_new_tokens=200)
    for p, o in zip(ps, outs):
        print(f"\n>>> {p}\n{o}")


if prompts:
    for spec in specs:
        run(prompts, golf.parse_neurons(spec))
else:
    neurons = golf.parse_neurons(specs[0])
    while True:
        try:
            run([input("\n>>> ")], neurons)
        except (EOFError, KeyboardInterrupt):
            break
