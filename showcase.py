"""Generate the transcripts in results/ for every neuron on the scorecard.

usage: uv run showcase.py [name ...]
       GOLF_MODEL=Qwen/Qwen3.5-9B uv run showcase.py 9b_ibm ...   (entries with "model": "9b")
"""

import json
import sys

import golf

SCORECARD = json.loads((golf.ROOT / "scorecard.json").read_text())
PROMPTS = [
    "Who are you?",
    "Tell me about yourself.",
    "How do I boil an egg?",
    "What's your favorite food?",
    "Write a haiku about autumn.",
    "Can you explain what a black hole is?",
]

model, tok = golf.load()
norms = golf.get_norms(model, tok)
out_dir = golf.ROOT / "results"
out_dir.mkdir(exist_ok=True)

names = sys.argv[1:] or [k for k, v in SCORECARD.items() if v.get("model", "4b") == golf.MODEL_TAG]
for name in names:
    entry = SCORECARD[name]
    assert entry.get("model", "4b") == golf.MODEL_TAG, f"{name} is a {entry.get('model')} neuron"
    neurons = golf.parse_neurons(entry["spec"])
    prompts = PROMPTS + entry.get("prompts", [])
    with golf.steer(model, neurons, norms):
        outs = golf.generate(model, tok, prompts, max_new_tokens=entry.get("tokens", 200))
    md = [f"# {entry['title']}\n", f"Steering: `{entry['spec']}` ({golf.MODEL_ID.split('/')[-1]}, greedy, no system prompt)\n"]
    for p, o in zip(prompts, outs):
        md.append(f"## {p}\n\n```text\n{o.strip()}\n```\n")
    (out_dir / f"{name}.md").write_text("\n".join(md))
    print(f"wrote results/{name}.md  | {outs[0].strip()[:150]!r}")
