"""Compare neurons across models by what they write: each neuron's output direction W_down[:, n],
read through the model's own final norm and unembedding (the logit lens, Geva et al. 2022), gives a
vector of logit changes over the vocabulary. Two neurons in two models match if those vectors agree
on the tokens both vocabularies share.

The cosine between two such logit vectors never needs the vectors themselves:
    logits_A(a) = Q_A a,   Q_A = centred rows of U_A diag(gain_A) on the shared tokens
    cos(a, b)   = a^T (Q_A^T Q_B) b / sqrt(a^T Q_A^T Q_A a  *  b^T Q_B^T Q_B b)
so one d_A x d_B cross-Gram matrix per model pair is enough to compare every neuron with every other.

usage: python paper/weights.py match <A> <B> [--per-layer 512]   best match in B for a sample of A
       python paper/weights.py personas <A> <B>                   best match in B for A's census personas
       python paper/weights.py twins <A> <B>                      every neuron's best match in B's same layer,
                                                                  both ways (models with equal depth)
       python paper/weights.py mutual <A> <B> LA:LB ...           share of mutual best matches between two layers
                                                                  (the twins' null: unrelated layers or models)
       python paper/weights.py stream <A> <B>                     `match`, fetching B's down_proj layer by layer
out:   paper/data/match_<A>_<B>.npz, paper/data/personas_<A>_<B>.jsonl, paper/data/twins_<A>_<B>.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

HERE = Path(__file__).parent
W = HERE / "weights"
sys.path.insert(0, str(HERE))
from census_table import DIMS, WEIGHT_ONLY, load  # noqa: E402

DIMS = DIMS | WEIGHT_ONLY

TOK = {"qwen35_4b": "Qwen_Qwen3.5-4B", "qwen35_9b": "Qwen_Qwen3.5-9B", "qwen3_8b": "Qwen_Qwen3-8B",
       "qwen35_08b": "Qwen_Qwen3.5-4B", "qwen35_2b": "Qwen_Qwen3.5-4B", "qwen35_27b": "Qwen_Qwen3.5-4B",  # one vocabulary
       "ministral3_3b": "mistralai_Ministral-3-3B-Instruct-2512", "ministral3_8b": "mistralai_Ministral-3-3B-Instruct-2512",
       "ministral3_14b": "mistralai_Ministral-3-3B-Instruct-2512",  # one tekken vocabulary for the family
       "mistral_s4": "mistralai_Mistral-Small-4-119B-2603"}
GAIN_PLUS_ONE = {"qwen35_08b", "qwen35_2b", "qwen35_4b", "qwen35_9b", "qwen35_27b"}  # Qwen3.5's RMSNorm multiplies by (1 + weight)
torch.set_num_threads(4)


def vocab(tag):
    return Tokenizer.from_file(str(W / "tok" / f"{TOK[tag]}.json")).get_vocab()


def shared(a, b):
    """Token ids in A and in B for every token string both vocabularies have (special tokens out)."""
    va, vb = vocab(a), vocab(b)
    keys = sorted(k for k in set(va) & set(vb) if not (k.startswith("<") and k.endswith(">")) and not k.startswith("[")
                  and k.strip())
    return torch.tensor([va[k] for k in keys]), torch.tensor([vb[k] for k in keys])


def readout(tag):
    """U diag(gain), fp32 [vocab, d]."""
    U = torch.load(W / tag / "unembed.pt").float()
    g = torch.load(W / tag / "norm.pt").float()
    return U.mul_(1 + g if tag in GAIN_PLUS_ONE else g)


def down(tag, L):
    return torch.load(W / tag / f"down_L{L:02d}.pt").float()  # [d, N]


def grams(a, b):
    """(C = Q_A^T Q_B, G_A, G_B) on the shared, centred vocabulary; cached."""
    p = W / f"gram_{a}_{b}.pt"
    if p.exists():
        return torch.load(p)
    if (W / f"gram_{b}_{a}.pt").exists():
        C, GB, GA, n = torch.load(W / f"gram_{b}_{a}.pt")
        return C.T.contiguous(), GA, GB, n
    ia, ib = shared(a, b)
    # in vocabulary chunks from the bf16 tensors: two fp32 unembeddings of a 27B and a 9B don't fit in 15 GB
    UA, UB = torch.load(W / a / "unembed.pt"), torch.load(W / b / "unembed.pt")
    gA, gB = (torch.load(W / t / "norm.pt").float() for t in (a, b))
    gA, gB = (1 + gA if a in GAIN_PLUS_ONE else gA), (1 + gB if b in GAIN_PLUS_ONE else gB)
    dA, dB, n = UA.shape[1], UB.shape[1], len(ia)
    C, GA, GB, sA, sB = torch.zeros(dA, dB), torch.zeros(dA, dA), torch.zeros(dB, dB), torch.zeros(dA), torch.zeros(dB)
    for s in range(0, n, 16384):
        xa, xb = UA[ia[s:s + 16384]].float() * gA, UB[ib[s:s + 16384]].float() * gB
        C += xa.T @ xb
        GA += xa.T @ xa
        GB += xb.T @ xb
        sA += xa.sum(0)
        sB += xb.sum(0)
    # centring over tokens: Q^T Q = X^T X - n mean mean^T
    out = (C - torch.outer(sA, sB) / n, GA - torch.outer(sA, sA) / n, GB - torch.outer(sB, sB) / n, n)
    torch.save(out, p)
    return out


def unit(G, X):
    """columns of X scaled to unit logit norm under Gram G."""
    return X / (X * (G @ X)).sum(0).clamp_min(1e-12).sqrt()


def all_down(tag, G):
    """[d, L*N] every neuron's output direction at unit logit norm, and its (layer, neuron) labels."""
    n_layers, N = DIMS[tag]
    X = torch.empty(G.shape[0], n_layers * N)  # filled in place: torch.cat would hold two copies
    for L in range(n_layers):
        X[:, L * N:(L + 1) * N] = unit(G, down(tag, L))
    lab = np.stack(np.meshgrid(np.arange(n_layers), np.arange(N), indexing="ij"), -1).reshape(-1, 2)
    return X, lab


def best_matches(a, b, A_cols, chunk=256):
    """For columns of A (output directions), the B neuron with the largest |logit cosine|.
    Returns index into B's flat (layer, neuron) list, signed cosine, the 10th best |cosine| (how
    unique the match is), the best |cosine| within each B layer [k, B layers], and B's labels."""
    C, GA, GB, _ = grams(a, b)
    XB, labB = all_down(b, GB)
    A = C.T @ unit(GA, A_cols)  # [d_B, k]: A's logit vectors pulled into B's coordinates
    idx, cos, tenth, per_layer = [], [], [], []
    n_layers_b = DIMS[b][0]
    for s in range(0, A.shape[1], chunk):
        S = A[:, s:s + chunk].T @ XB  # [k, L*N] cosines
        v, i = S.abs().topk(10, dim=1)
        idx.append(i[:, 0])
        cos.append(S.gather(1, i[:, :1])[:, 0])
        tenth.append(v[:, 9])
        per_layer.append(S.abs().view(S.shape[0], n_layers_b, -1).amax(-1))
        print(f"  {a}->{b}: {s + A[:, s:s + chunk].shape[1]}/{A.shape[1]}", flush=True)
    return (torch.cat(idx).numpy(), torch.cat(cos).numpy(), torch.cat(tenth).numpy(),
            torch.cat(per_layer).numpy(), labB)


def cmd_match(a, b, per_layer, seed=0):
    """Sample per_layer neurons from every layer of A (plus as many random directions as a null)."""
    rng = np.random.default_rng(seed)
    n_layers, N = DIMS[a]
    picks = np.stack([np.sort(rng.choice(N, per_layer, replace=False)) for _ in range(n_layers)])
    cols = torch.cat([down(a, L)[:, torch.from_numpy(picks[L])] for L in range(n_layers)], 1)
    rand = torch.randn(cols.shape[0], per_layer * 4, generator=torch.Generator().manual_seed(seed))
    i, c, t, pl, labB = best_matches(a, b, torch.cat([cols, rand], 1))
    k = cols.shape[1]
    np.savez_compressed(HERE / "data" / f"match_{a}_{b}.npz", a_layer=np.repeat(np.arange(n_layers), per_layer),
                        a_neuron=picks.ravel(), b_layer=labB[i[:k], 0], b_neuron=labB[i[:k], 1], cos=c[:k],
                        tenth=t[:k], layer_max=pl[:k], null_cos=c[k:], null_layer=labB[i[k:], 0],
                        null_layer_max=pl[k:])


def cmd_personas(a, b):
    """Every census persona of A, pushed in its persona direction (sign * W_down column): the best
    B neuron and sign, and what B's census says that neuron does."""
    _, _, rows = load(a)
    sb, _, rows_b = load(b)
    pb = {(r["L"], r["n"], r["s"]): r for r in rows_b}
    by_layer = {}
    for j, r in enumerate(rows):
        by_layer.setdefault(r["L"], []).append(j)
    cols = torch.zeros(DIMS_D(a), len(rows))
    for L, js in by_layer.items():
        D = down(a, L)
        for j in js:
            cols[:, j] = rows[j]["s"] * D[:, rows[j]["n"]]
    i, c, t, _, labB = best_matches(a, b, cols)
    with open(HERE / "data" / f"personas_{a}_{b}.jsonl", "w", encoding="utf-8") as f:
        for r, ii, cc, tt in zip(rows, i, c, t):
            L, n = (int(x) for x in labB[ii])
            s = 1 if cc > 0 else -1
            hit = pb.get((L, n, s))
            f.write(json.dumps({"a": f"L{r['L']} N{r['n']} {r['s']:+d}", "a_cat": r["cat"], "a_text": r["text"],
                                "b": f"L{L} N{n} {s:+d}", "b_layer": L, "cos": round(float(abs(cc)), 4),
                                "tenth": round(float(tt), 4), "b_state": int(sb[L, n, 0 if s > 0 else 1]),
                                "b_cat": hit["cat"] if hit else None, "b_text": hit["text"] if hit else None},
                               ensure_ascii=False) + "\n")


def cmd_twins(a, b):
    """Same-layer matching for every neuron of A and of B: best partner, signed cosine, runner-up
    cosine, and whether the pair is mutual (each is the other's best)."""
    C, GA, GB, _ = grams(a, b)
    n_layers, NA = DIMS[a]
    NB = DIMS[b][1]
    out = {k: np.zeros((n_layers, NA if k[0] == "a" else NB), dtype)
           for k, dtype in (("a_best", np.int32), ("a_cos", np.float32), ("a_second", np.float32),
                            ("b_best", np.int32), ("b_cos", np.float32), ("b_second", np.float32))}
    for L in range(n_layers):
        XA = C.T @ unit(GA, down(a, L))  # [d_B, NA]
        XB = unit(GB, down(b, L))
        S = XA.T @ XB  # [NA, NB]
        for side, M in (("a", S), ("b", S.T)):
            v, i = M.abs().topk(2, dim=1)
            out[f"{side}_best"][L] = i[:, 0].numpy()
            out[f"{side}_cos"][L] = M.gather(1, i[:, :1])[:, 0].numpy()
            out[f"{side}_second"][L] = v[:, 1].numpy()
        print(f"  twins {a}<->{b} L{L}", flush=True)
    rows = np.arange(NA)
    out["mutual"] = np.stack([out["b_best"][L][out["a_best"][L]] == rows for L in range(n_layers)])
    np.savez_compressed(HERE / "data" / f"twins_{a}_{b}.npz", **out)


def cmd_mutual(a, b, pairs):
    C, GA, GB, _ = grams(a, b)
    out = {}
    for p in pairs:
        La, Lb = (int(x) for x in p.split(":"))
        S = ((C.T @ unit(GA, down(a, La))).T @ unit(GB, down(b, Lb))).abs()
        ab, ba = S.argmax(1), S.argmax(0)
        out[p] = float((ba[ab] == torch.arange(S.shape[0])).float().mean())
        print(f"  {a} L{La} <-> {b} L{Lb}: {out[p]:.3f} mutual", flush=True)
    path = HERE / "data" / "mutual.json"
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr[f"{a}~{b}"] = allr.get(f"{a}~{b}", {}) | out
    path.write_text(json.dumps(allr, indent=1))


def cmd_stream(a, b, per_layer, seed=0):
    """`match` for a B too big to keep on disk: B's down_proj is fetched one layer at a time and dropped."""
    import fetch
    C, GA, GB, _ = grams(a, b)
    rng = np.random.default_rng(seed)
    n_layers, N = DIMS[a]
    picks = np.stack([np.sort(rng.choice(N, per_layer, replace=False)) for _ in range(n_layers)])
    cols = torch.cat([down(a, L)[:, torch.from_numpy(picks[L])] for L in range(n_layers)], 1)
    rand = torch.randn(cols.shape[0], per_layer * 4, generator=torch.Generator().manual_seed(seed))
    A = C.T @ unit(GA, torch.cat([cols, rand], 1))
    repo, n_b, _, _, layer, _ = fetch.MODELS[b]
    wmap = json.loads(fetch.get(fetch.url(repo, "model.safetensors.index.json")))["weight_map"]
    from concurrent.futures import ThreadPoolExecutor
    names = [layer.format(Lb) + "mlp.down_proj.weight" for Lb in range(n_b)]
    hdrs = {sh: fetch.header(repo, sh) for sh in {wmap[n] for n in names}}
    best, arg = [], []
    with ThreadPoolExecutor(6) as pool:  # one HTTP stream is slow; six in flight, consumed in order
        futs = [pool.submit(fetch.tensor, repo, wmap[n], hdrs[wmap[n]], n) for n in names]
        for Lb, f in enumerate(futs):
            XB = unit(GB, f.result().float())
            futs[Lb] = None
            v, i = (A.T @ XB).abs().max(1)
            best.append(v)
            arg.append(i)
            print(f"  {a}->{b} L{Lb}", flush=True)
    best, arg = torch.stack(best, 1).numpy(), torch.stack(arg, 1).numpy()  # [k, B layers]
    k = cols.shape[1]
    bl = best.argmax(1)
    np.savez_compressed(HERE / "data" / f"match_{a}_{b}.npz", a_layer=np.repeat(np.arange(n_layers), per_layer),
                        a_neuron=picks.ravel(), b_layer=bl[:k], b_neuron=arg[np.arange(k), bl[:k]],
                        cos=best[np.arange(k), bl[:k]], layer_max=best[:k], null_cos=best[k:].max(1),
                        null_layer=bl[k:], null_layer_max=best[k:])


def DIMS_D(tag):
    return torch.load(W / tag / "norm.pt").shape[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["match", "personas", "twins", "mutual", "stream"])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("pairs", nargs="*", help="LA:LB layer pairs for `mutual`")
    ap.add_argument("--per-layer", type=int, default=512)
    args = ap.parse_args()
    if args.cmd == "match":
        cmd_match(args.a, args.b, args.per_layer)
    elif args.cmd == "twins":
        cmd_twins(args.a, args.b)
    elif args.cmd == "mutual":
        cmd_mutual(args.a, args.b, args.pairs)
    elif args.cmd == "stream":
        cmd_stream(args.a, args.b, args.per_layer)
    else:
        cmd_personas(args.a, args.b)
