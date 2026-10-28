# New here? Start with this

Background for [neuron golf](../README.md), for people outside the field.

**What's going on, in plain words.** A language model like Qwen is a stack of 32 *layers*. Each layer
has an MLP block containing 9,216 *neurons*, and each neuron is just a number computed for every word
(token) the model reads or writes. Some neurons line up with recognizable concepts: one goes up on SAP
jargon, another on basketball. **Steering** means overriding one of those numbers while the model talks,
by adding a large constant to it at every token. If the neuron carries a concept, the model starts seeing
that concept everywhere. Anthropic's [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude)
is the famous example: they clamped one learned feature for the Golden Gate Bridge, and Claude started
claiming to *be* the bridge. Here we do the cheap version on raw neurons of a small open model on a gaming GPU.

**How to read the notation.** `L4 N7892 +1` means layer 4 (0-based), neuron 7892, dose +1. Dose 1 adds
a push as big as the model's typical internal activity at that layer (exact formula
in [Methods](methods.md#the-rules-steering-recipe)). Negative doses push the neuron the other way. Every transcript is
greedy (deterministic), with no system prompt and Qwen's "thinking" turned off, so the only thing different
from the normal model is the neuron.

**Reading list, roughly beginner → advanced:**

| | |
|---|---|
| 🎥 [3Blue1Brown — *Transformers, the tech behind LLMs*](https://www.youtube.com/watch?v=wjZofJX0v4M) | what a transformer layer is |
| 🎥 [3Blue1Brown — *How might LLMs store facts*](https://www.youtube.com/watch?v=9-Jl0dxWQs8) | the MLP blocks and neurons we poke here |
| 📰 [Anthropic — *Golden Gate Claude*](https://www.anthropic.com/news/golden-gate-claude) | the "clamp one feature" demo this project imitates |
| 📄 [Templeton et al. — *Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | the research behind Golden Gate Claude (features via sparse autoencoders) |
| 📄 [Elhage et al. — *Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) | why most neurons are messy mixtures, and why clean ones are lucky finds |
| 📄 [Bricken et al. — *Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features) | turning messy neurons into cleaner features |
| 📄 [Geva et al. — *FFN Layers Are Key-Value Memories*](https://arxiv.org/abs/2012.14913) and [*…Promoting Concepts in the Vocabulary Space*](https://arxiv.org/abs/2203.14680) | how an MLP neuron "reads" a pattern and "writes" a concept |
| 📄 [Dai et al. — *Knowledge Neurons in Pretrained Transformers*](https://arxiv.org/abs/2104.08696) | individual neurons tied to specific facts |
| 📄 [Turner et al. — *Activation Engineering (ActAdd)*](https://arxiv.org/abs/2308.10248), [Rimsky et al. — *Contrastive Activation Addition*](https://arxiv.org/abs/2312.06681), [Zou et al. — *Representation Engineering*](https://arxiv.org/abs/2310.01405) | steering with directions instead of single neurons |
| 📄 [Chen et al. — *Persona Vectors*](https://arxiv.org/abs/2507.21509) | steering whole character traits |
| 📄 [Arditi et al. — *Refusal Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717) | the same idea used to remove refusals ("abliteration") |
| 🔎 [Neuronpedia](https://www.neuronpedia.org) | browse and steer features in many open models |
| 🔎 [Qwen3-8B Neuron Persona Library](https://qwen-neurons-site.vercel.app/#library) | a full census of "who are you?" under every neuron of Qwen3-8B (a different model, same recipe) |
| 🧰 [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), [SwiGLU (Shazeer)](https://arxiv.org/abs/2002.05202), [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | the model, its MLP type, and its linear-attention layers |
