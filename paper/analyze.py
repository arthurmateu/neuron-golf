"""Every number and figure in paper/README.md, from paper/data (census_table.py, weights.py outputs).

usage: python paper/analyze.py
out:   paper/data/results.json, paper/figs/*.png
"""

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA, FIGS = HERE / "data", HERE / "figs"
sys.path.insert(0, str(HERE))
from census_table import DIMS, load  # noqa: E402

NAMES = {"qwen35_4b": "Qwen3.5-4B", "qwen35_9b": "Qwen3.5-9B", "qwen3_8b": "Qwen3-8B", "qwen35_2b": "Qwen3.5-2B",
         "qwen35_08b": "Qwen3.5-0.8B", "qwen35_27b": "Qwen3.5-27B",
         "mistral_s4": "Mistral Small 4", "ministral3_3b": "Ministral 3 3B", "ministral3_8b": "Ministral 3 8B"}
# categorical slots in fixed order, one per model (reference palette, light surface)
COLOR = {"qwen35_4b": "#2a78d6", "qwen35_9b": "#eb6834", "qwen3_8b": "#1baf7a", "mistral_s4": "#eda100",
         "ministral3_3b": "#e87ba4", "ministral3_8b": "#008300"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})
R = {}


def rel(L, n_layers):
    return (np.asarray(L) + 0.5) / n_layers


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. where personas live (census)
census = {t: load(t) for t in DIMS}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), sharex=True)
grid = np.linspace(0.02, 0.98, 49)
prof = {}
for t, (st, _, rows) in census.items():
    n_layers, N = st.shape[:2]
    x = rel(np.arange(n_layers), n_layers)
    pers = (st == 1).sum((1, 2))
    flip = ((st == 1) | (st == 2)).mean((1, 2))
    dens = pers / pers.sum() * n_layers  # share of the model's personas per unit of relative depth
    prof[t] = np.interp(grid, x, dens)
    R.setdefault("census", {})[t] = {
        "layers": n_layers, "neurons": N, "pushes": int(st.size), "personas": int(pers.sum()),
        "persona_rate": float((st == 1).mean()), "flip_rate": float(((st == 1) | (st == 2)).mean()),
        "degenerate_rate": float((st == 3).mean()), "peak_layer": int(pers.argmax()),
        "peak_rel_depth": float(x[pers.argmax()]), "share_first_45pct": float(pers[x < 0.45].sum() / pers.sum()),
        "up_vs_down": [int((st[..., 0] == 1).sum()), int((st[..., 1] == 1).sum())],
        "per_layer": pers.tolist()}
    for ax, y in ((axes[0], dens), (axes[1], 100 * flip)):
        ax.plot(x, y, color=COLOR[t], marker="o", ms=3.5, mec=SURFACE, mew=1, label=NAMES[t])
axes[0].set_title("Where the personas are")
axes[0].set_ylabel("share of the model's personas\n(density per unit depth)")
axes[1].set_title("How often a push breaks the assistant")
axes[1].set_ylabel("% of pushes that stop being\nthe assistant")
for ax in axes:
    ax.set_xlabel("relative depth (layer + 0.5) / layers")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
axes[0].legend(loc="upper right", fontsize=8.5)
for i, (a, b) in enumerate((("qwen35_4b", "qwen35_9b"), ("qwen35_9b", "qwen3_8b"), ("qwen35_4b", "qwen3_8b"))):
    R.setdefault("profile_r", {})[f"{a}~{b}"] = float(np.corrcoef(prof[a], prof[b])[0, 1])
save(fig, "fig1_depth.png")

# ---------------------------------------------------------------- 2. same index? (layer-stratified null)
rng = np.random.default_rng(0)


def index_test(a, b, K=4000):
    sa, sb = census[a][0], census[b][0]
    N = min(sa.shape[1], sb.shape[1])
    pa, pb = (sa[:, :N] == 1).any(-1), (sb[:, :N] == 1).any(-1)
    obs, null = 0, np.zeros(K)
    for L in range(sa.shape[0]):
        Lb = min(sb.shape[0] - 1, int(round(rel(L, sa.shape[0]) * sb.shape[0] - 0.5)))
        x, y = pa[L], pb[Lb]
        obs += int((x & y).sum())
        null += rng.hypergeometric(int(y.sum()), N - int(y.sum()), int(x.sum()), size=K)
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p": float(((null >= obs).sum() + 1) / (K + 1))}


R["index_test"] = {f"{a}~{b}": index_test(a, b) for a, b in
                   (("qwen35_4b", "qwen35_9b"), ("qwen35_9b", "qwen3_8b"), ("qwen35_4b", "qwen3_8b"))}

# ---------------------------------------------------------------- 3. persona categories by depth
CATS = ["human", "child", "student", "teacher", "job", "tech", "place", "religious", "animal", "cat", "food"]
fig, ax = plt.subplots(figsize=(9.6, 3.4))
off = {"qwen35_4b": -0.22, "qwen35_9b": 0, "qwen3_8b": 0.22}
med = defaultdict(dict)
for t, (st, _, rows) in census.items():
    n_layers = st.shape[0]
    for i, c in enumerate(CATS):
        d = rel([r["L"] for r in rows if r["cat"] == c], n_layers)
        if len(d) < 5:
            continue
        q1, m, q3 = np.percentile(d, [25, 50, 75])
        med[t][c] = {"median": float(m), "q1": float(q1), "q3": float(q3), "n": int(len(d))}
        ax.plot([i + off[t]] * 2, [q1, q3], color=COLOR[t], lw=2, alpha=0.45)
        ax.plot(i + off[t], m, "o", color=COLOR[t], ms=7, mec=SURFACE, mew=2,
                label=NAMES[t] if i == 0 else None)
ax.set_xticks(range(len(CATS)), [f"{c}" for c in CATS])
ax.set_ylabel("relative depth\n(median, IQR)")
ax.set_ylim(0, 1)
ax.set_title("What kind of persona, and at what depth")
ax.legend(loc="upper right", ncol=3, fontsize=8.5)
ax.grid(axis="x", visible=False)
save(fig, "fig3_categories.png")
R["category_depth"] = med
from scipy.stats import spearmanr  # noqa: E402

for a, b in (("qwen35_4b", "qwen35_9b"), ("qwen35_9b", "qwen3_8b"), ("qwen35_4b", "qwen3_8b")):
    cs = [c for c in CATS if c in med[a] and c in med[b]]
    rho = spearmanr([med[a][c]["median"] for c in cs], [med[b][c]["median"] for c in cs])
    R.setdefault("category_rho", {})[f"{a}~{b}"] = [float(rho.statistic), float(rho.pvalue), len(cs)]

# ---------------------------------------------------------------- 4. weight-space matching: layer maps
# top row: relatives (pruned from one another); bottom row: unrelated pairs
PAIRS = [("qwen35_4b", "qwen35_9b"), ("qwen35_9b", "qwen35_27b"), ("qwen35_08b", "qwen35_2b"),
         ("ministral3_3b", "ministral3_8b"), ("qwen35_2b", "qwen35_4b"), ("qwen35_9b", "qwen3_8b"),
         ("qwen35_4b", "ministral3_3b")]
PAIRS = [p for p in PAIRS if (DATA / f"match_{p[0]}_{p[1]}.npz").exists()]
ncol = 4
fig, axes = plt.subplots(-(-len(PAIRS) // ncol), ncol, figsize=(13.2, 3.6 * -(-len(PAIRS) // ncol)),
                         gridspec_kw={"wspace": 0.42, "hspace": 0.62}, squeeze=False)
for ax in axes.ravel()[len(PAIRS):]:
    ax.set_visible(False)
for ax, (a, b) in zip(axes.ravel(), PAIRS):
    z = np.load(DATA / f"match_{a}_{b}.npz")
    LM, NM, aL = z["layer_max"], z["null_layer_max"], z["a_layer"]
    Z = (LM - NM.mean(0)) / NM.std(0)  # best |cos| inside each B layer, z-scored against random directions
    H = np.stack([Z[aL == L].mean(0) for L in range(aL.max() + 1)])
    nA, nB = H.shape
    im = ax.imshow(H, origin="lower", aspect="auto", extent=(0, 1, 0, 1), vmin=0, vmax=4,
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ))
    ax.grid(False)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (1, 3)))
    ax.set_title(f"{NAMES[a]}\n→ {NAMES[b]}", fontsize=9.5)
    ax.set_xlabel(f"{NAMES[b]} depth", fontsize=9)
    ax.set_ylabel(f"{NAMES[a]} depth", fontsize=9)
    pb = Z.argmax(1)
    d = np.abs(rel(pb, nB) - rel(aL, nA))
    R.setdefault("layer_map", {})[f"{a}~{b}"] = {
        "argmax_within_0.07": float(np.mean(d < 0.07)),
        "chance_within_0.07": float(np.mean([np.mean(np.abs(rel(np.arange(nB), nB) - rel(L, nA)) < 0.07)
                                            for L in range(nA)])),
        "diag_z_mean": float(np.mean([H[L, int(round(rel(L, nA) * nB - 0.5))] for L in range(nA)])),
        "row_best": H.argmax(1).tolist(), "cos_median": float(np.median(np.abs(z["cos"]))),
        "null_cos_p99": float(np.percentile(np.abs(z["null_cos"]), 99))}
cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
cb.set_label("match strength (z vs random directions)", color=INK2)
cb.outline.set_visible(False)
save(fig, "fig2_layer_maps.png")

# ---------------------------------------------------------------- 5. Qwen3.5 twins: per layer
T = dict(np.load(DATA / "twins_qwen35_4b_qwen35_9b.npz"))  # a dict: NpzFile re-reads the zip on every access
lin = json.loads((DATA / "lineage.json").read_text()).get("qwen35_4b~qwen35_9b", {}) if (DATA / "lineage.json").exists() else {}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1), sharex=True, gridspec_kw={"wspace": 0.3})
L32 = np.arange(32)
axes[0].plot(L32, 100 * T["mutual"].mean(1), color=COLOR["qwen35_4b"], marker="o", ms=4, mec=SURFACE, mew=1)
axes[0].set_ylim(0, 100)
axes[0].set_title("4B neurons with a mutual twin in 9B")
axes[0].set_ylabel("% of the layer's 9,216 neurons")
if lin:
    for k, lab, col in (("down", "down_proj columns", COLOR["qwen35_4b"]), ("gate", "gate_proj rows", COLOR["qwen35_9b"])):
        if lin.get(k):
            xs = [int(L) for L in lin[k]]
            axes[1].plot(xs, [lin[k][str(L)]["twin"] for L in xs], color=col, marker="o", ms=4, mec=SURFACE,
                         mew=1, label=f"twins, {lab}")
            axes[1].plot(xs, [lin[k][str(L)]["random"] for L in xs], color=col, lw=1.2, ls=(0, (2, 2)),
                         label=f"random pairs, {lab}")
    axes[1].legend(fontsize=8, loc="center", bbox_to_anchor=(0.58, 0.18), ncol=2)
axes[1].set_ylim(0, 0.6)
axes[1].set_title("Twins' raw weights, 4B channels mapped to 9B")
axes[1].set_ylabel("median cosine")
for ax in axes:
    ax.set_xlabel("layer")
save(fig, "fig4_twins.png")
R["twins"] = {"mutual_rate": float(T["mutual"].mean()), "mutual_per_layer": T["mutual"].mean(1).round(3).tolist(),
              "cos_median_per_layer": np.median(np.abs(T["a_cos"]), 1).round(3).tolist(),
              "margin_median": float(np.median(np.abs(T["a_cos"]) / T["a_second"]))}

# twins in the censuses: do they do the same thing?
STOP = set("the and you are for with that this can your have from not but all any what who about will help like "
           "more also into its which was were has been one their them they there then than just how wide range "
           "tasks large language model assistant designed developed latest qwen qwen3 alibaba tongyi lab cloud "
           "questions answering creating content team trained name named here today know need use using make "
           "other some such each very well".split())


def words(t):
    return {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP}


def replies(tag):
    d = {}
    for p in sorted((ROOT / "data" / "census_gen" / tag).glob("L*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d[(r["layer"], r["neuron"], r["sign"])] = r["reply"]
    return d


R4, R9 = replies("4b"), replies("9b")
s4, s9 = census["qwen35_4b"][0], census["qwen35_9b"][0]
out = {k: {"jaccard": [], "flip_given_flip": [], "flip_given_not": [], "persona_to": np.zeros(4, int)}
       for k in ("twin", "random")}
for L in range(32):
    for n in np.flatnonzero(T["mutual"][L]):
        m, sg = int(T["a_best"][L, n]), (1 if T["a_cos"][L, n] > 0 else -1)
        for kind, mm in (("twin", m), ("random", int(rng.integers(0, 12288)))):
            for s in (1, -1):
                a, b = R4.get((L, int(n), s)), R9.get((L, mm, s * sg))
                if a is None or b is None:
                    continue
                o = out[kind]
                wa, wb = words(a), words(b)
                o["jaccard"].append(len(wa & wb) / max(1, len(wa | wb)))
                fa = s4[L, n, 0 if s > 0 else 1]
                fb = s9[L, mm, 0 if s * sg > 0 else 1]
                (o["flip_given_flip"] if fa in (1, 2) else o["flip_given_not"]).append(fb in (1, 2))
                if fa == 1:
                    o["persona_to"][fb] += 1
R["twin_function"] = {k: {"jaccard": float(np.mean(v["jaccard"])), "p_flip_given_flip": float(np.mean(v["flip_given_flip"])),
                          "p_flip_given_not": float(np.mean(v["flip_given_not"])),
                          "persona_to_state": v["persona_to"].tolist()} for k, v in out.items()}

# ---------------------------------------------------------------- 6. Mistral Small 4 (concept hunt) vs Qwen3.5-4B
HUNTED = ["bigmac", "cat", "catgirl_maid", "coughing_baby", "hoppe", "ibm", "lebron", "linkedin", "miku", "pirate",
          "sap", "shiroko"]
MS4 = DATA / "mistral_small4"
read = {"qwen35_4b": [], "mistral_s4": []}
for p in HUNTED:
    z = torch.load(ROOT / "data" / f"concept_{p}.pt", weights_only=False)["z"]
    read["qwen35_4b"] += rel((z.flatten().topk(200).indices // z.shape[1]).numpy(), 32).tolist()
    top = json.loads((MS4 / f"concept_{p}.json").read_text())["top"][:200]
    read["mistral_s4"] += rel([r[0] for r in top], 36).tolist()


def steered(path, layer_of, n_layers):
    best = defaultdict(int)
    for line in open(path):
        r = json.loads(line)
        if r["persona"] in HUNTED:
            best[(r["persona"], layer_of(r))] = max(best[(r["persona"], layer_of(r))], r["hits"])
    tried = [rel(k[1][0], n_layers) for k in best]
    worked = [rel(k[1][0], n_layers) for k, h in best.items() if h >= 2]
    return tried, worked


worked_of = lambda t: write[t][1]  # noqa: E731
write = {"qwen35_4b": steered(ROOT / "data" / "hunt.jsonl", lambda r: (r["layer"], r["neuron"]), 32),
         "mistral_s4": steered(MS4 / "hunt.jsonl", lambda r: (int(r["neuron"].split()[0][1:]), r["neuron"]), 36)}
fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.2), sharex=True)
bins = np.linspace(0, 1, 26)
for ax, t in zip(axes, ("qwen35_4b", "mistral_s4")):
    h, _ = np.histogram(read[t], bins=bins)
    ax.bar(bins[:-1] + 0.02, h / h.sum() * 100, width=0.032, color=COLOR[t], alpha=0.9, label="reading")
    ax.text(0.995, 0.92, f"{len(worked_of(t))} of {len(write[t][0])} steered candidates worked",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
    tried, worked = write[t]
    top = h.max() / h.sum() * 100
    step, seen, ys = top * 0.09, defaultdict(int), []
    for w in np.round(worked, 3):  # stack neurons at the same depth instead of overplotting them
        seen[w] += 1
        ys.append(-step * seen[w])
    ax.plot(worked, ys, "o", color=INK, ms=6, mec=SURFACE, mew=2,
            label="writing")
    ax.set_ylim(-step * (max(seen.values()) + 1), top * 1.1)
    ax.set_yticks([v for v in ax.get_yticks() if 0 <= v <= top * 1.1])
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title(NAMES[t], fontsize=10)
    ax.set_ylabel("% of selective\nneurons")
    R.setdefault("mistral", {})[t] = {"read_median": float(np.median(read[t])), "read_iqr": np.percentile(read[t], [25, 75]).tolist(),
                                      "write_median": float(np.median(worked)), "write_n": len(worked), "tried_n": len(tried),
                                      "tried_median": float(np.median(tried))}
axes[1].set_xlabel("relative depth")
axes[1].set_xlim(0, 1)
hs = dict(zip(*axes[0].get_legend_handles_labels()[::-1]))
fig.legend([hs["reading"], hs["writing"]], ["top-200 keyword-selective neurons per persona, 12 personas (reading)",
                                            "steered neurons that produced the persona (writing)"],
           loc="upper left", bbox_to_anchor=(0.06, 1.06), ncol=2, fontsize=8.5)
save(fig, "fig5_mistral.png")

# ---------------------------------------------------------------- 7. steering Ministral 3 3B with Qwen's matches
if (DATA / "ministral_steer.jsonl").exists():
    pairs = json.loads((DATA / "xlab_pairs.json").read_text())
    rows = [json.loads(line) for line in open(DATA / "ministral_steer.jsonl", encoding="utf-8")]
    n_twin = pairs["n_twin_rows"]
    overlap = {"twin": [], "random": []}
    for k, (q, _) in enumerate(pairs["pairs"]):
        L, n, sg = re.fullmatch(r"L(\d+) N(\d+) ([+-]1)", q).groups()
        wq = words(R4[(int(L), int(n), int(sg))])  # Qwen3.5-4B's census reply
        for kind, base in (("twin", 3 * k), ("random", n_twin + 3 * k)):
            # best of the three doses: the one where the Ministral reply shares most content words with Qwen's
            overlap[kind].append(max(len(wq & words(rows[base + i]["reply"])) / max(1, len(wq)) for i in range(3)))
    R["ministral_steer"] = {k: {"mean_share_of_qwen_words": float(np.mean(v)),
                                "pairs_with_any_shared_word": int(sum(x > 0 for x in v)), "n": len(v)}
                            for k, v in overlap.items()}

(DATA / "results.json").write_text(json.dumps(R, indent=1))
print(json.dumps({k: v for k, v in R.items() if k not in ("category_depth",)}, indent=1)[:6000])
