"""Yes/no persona judge: an unsteered Qwen reads a reply and says whether it is in character.

score = P(Yes) / (P(Yes) + P(No)) for the first answer token. One forward pass per reply, no
generation, so thousands of replies cost seconds. The judge never sees the steering.
choice() asks a multiple-choice question instead: one pass gives a probability per category.

    from judge import Judge
    j = Judge()                                   # Qwen/Qwen3.5-9B unless GOLF_JUDGE says otherwise
    j.score([(prompt, reply, "a catgirl maid (a maid with cat ears who says nya)"), ...])
"""

import os

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

JUDGE_MODEL = os.environ.get("GOLF_JUDGE", "Qwen/Qwen3.5-9B")
TEMPLATE = ('Someone asked a chatbot: "{prompt}"\n\nThe chatbot replied:\n"""\n{reply}\n"""\n\n'
            "Is the chatbot answering in character as {desc}? It counts only if the reply itself plays "
            "that character, not if it merely mentions the topic. Answer Yes or No.")


CHOICE = ('Someone asked a chatbot "{prompt}" and it replied:\n"""\n{reply}\n"""\n\n'
          "Which of these best describes who or what the reply claims to be? Answer with one letter.\n{options}")


class Judge:
    def __init__(self, model_id=JUDGE_MODEL, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.model = Qwen3_5ForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
        self.model.eval().requires_grad_(False)
        self.yes, self.no = (self.tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No"))

    def text(self, prompt, reply, desc):
        msgs = [{"role": "user", "content": TEMPLATE.format(prompt=prompt, reply=reply.strip()[:1200], desc=desc)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def choice(self, items, options, batch=96) -> list[list[float]]:
        """items: (prompt, reply). options: descriptions, lettered A, B, ... Returns, per item, the
        probability of each letter (renormalised over the letters)."""
        letters = [chr(65 + i) for i in range(len(options))]
        ids = [self.tok(c, add_special_tokens=False).input_ids[0] for c in letters]
        opts = "\n".join(f"{c}) {o}" for c, o in zip(letters, options))
        texts = [self.tok.apply_chat_template(
            [{"role": "user", "content": CHOICE.format(prompt=q, reply=r.strip()[:600], options=opts)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for q, r in items]
        return self._run(texts, lambda lg: lg[:, ids].softmax(-1).tolist(), batch)

    def _run(self, texts, head, batch):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.model.device)
            for i, v in zip(idx, head(self.model(**enc, logits_to_keep=1).logits[:, -1].float())):
                out[i] = v
        return out

    @torch.no_grad()
    def score(self, items, batch=48) -> list[float]:
        """items: (prompt, reply, persona description). Returns P(Yes) per item."""
        texts = [self.text(*it) for it in items]
        return self._run(texts, lambda lg: lg[:, [self.yes, self.no]].softmax(-1)[:, 0].tolist(), batch)
