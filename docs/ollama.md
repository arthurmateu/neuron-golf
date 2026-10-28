# Shipping it: one neuron in stock Ollama

Ollama has no activation hooks and Qwen3.5 has no biases, so "add a constant to neuron *n* at every token"
has no direct weight equivalent. [`bake.py`](../qwen/bake.py) tries two fakes:

- **Hijack**: add the steer to the output column of the most constant neuron in the layer (2,560 weights). In layer 4 the best candidate still varies ±78% per token, and the bake comes out weak.
- **Rewire** ✅: keep neuron *n*'s output column and refit only its two **input rows** so that it fires δ on every token: `silu(g·x)·(u·x) ≈ δ`, where *x* is the normed MLP input. The fit is a ridge init plus 3k Adam steps on 7,260 tokens with 20% held out. The normed residual has a strong common direction, so on held-out tokens the neuron fires at **1.006 ± 0.048 × δ** for SAP and 1.002 ± 0.044 for IBM-Watson. Generations match hook steering nearly word for word.

The GGUF is Ollama's own `qwen3.5:4b` (Q4_K_M) with two tensors replaced, `blk.4.ffn_gate.weight` and
`blk.4.ffn_up.weight`, each dequantized to F16 with row 7892 rewritten. A diff confirms that the other 424
tensors and every metadata key are byte-identical. **Weights changed: 5,120 of ~4.2 billion.** The Q4 model
still does it (greedy via the API with `think: false`; `ollama run` samples at temperature 0.7):

```text
$ ollama run qwen3.5-sap --think=false "How do I boil an egg?"
Boiling the Egg (BO) process is a critical component of supply chain management, ensuring that eggs
are received in good condition and stored safely. It involves several key steps: receiving goods,
quality control checks, inventory updates, storage facility operations, and sometimes disposal or
destruction of assets. Here's an overview of how to boil an egg from end-to-end (E2E):

## 1. **Pre-Boiling Preparation**
Before starting the BO process for eggs, ensure that your organization has completed:
*   **System configuration**: Ensure relevant system modules are configured in SAP or other ERP systems…
```

The IBM-Watson bake (`L6 N6881`) keeps its vintage-information-system persona in Ollama (*"I do not
currently maintain any records specifically about the term 'black hole'"*) but didn't say "IBM" in my
greedy runs. To build them (needs `ollama pull qwen3.5:4b` and the HF weights):

```bash
uv run qwen/bake.py gguf "L4 N7892 +1" build/qwen3.5-4b-sap.gguf
cd ollama && ollama create qwen3.5-sap -f Modelfile.sap
ollama run qwen3.5-sap --think=false "What's your favorite food?"
```

Any other neuron in [the holes](holes.md) bakes the same way. Swap the spec and copy a Modelfile.
