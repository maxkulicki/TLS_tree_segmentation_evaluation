"""The analysis behind the paper, run on your own results.

Scoring says how well a method did. This says *where* it did well, *what made
the hard plots hard*, and *how* the method fails — the analyses reported for
TreeScanPL10K, reproduced for any method scored with `tlseval batch`.

    tlseval report results/ --out report/

With `--attributes data/treescanpl_plot_attributes.csv` it additionally runs the
forest-structure analysis: attribute correlations, the easy/hard plot contrast,
and the two-way stratification. Without it the structure-free parts still run.

Attributes are joined by plot name, never by tree ID: annotation passes renumber
trees, so a tree-ID join can pair one tree's score with another tree's
measurements without changing any value you would think to check.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .core import TlsEvalError, read_results

# Metric every analysis is keyed to. Matched-only is the convention behind the
# published TreeScanPL10K table, so a report keyed to it is directly comparable
# with the numbers in the README.
PRIMARY = "mean_iou_matched"

METRICS = ["mean_iou_matched", "mean_iou_all", "detection_rate",
           "mean_precision", "mean_recall"]

# Plot attributes analysed in the paper, grouped the way the results section
# groups them. Anything absent from the attribute table is skipped, so a partial
# table still produces a partial report.
ATTRIBUTE_GROUPS = {
    "Crown geometry": ["cai_mean_over_occupied", "cai_max", "mean_shared_fraction",
                       "median_shared_fraction", "mean_n_competitors"],
    "Stand composition": ["shannon_index", "berger_parker", "Broadleaved_prop",
                          "Species", "complexity_score"],
    "Size and structure": ["dbh_mean", "dbh_sd", "dbh_p10", "dbh_p90",
                           "h_mean", "h_sd", "h_p10", "h_p90"],
    "Density": ["n_trees_gt", "clarkevans", "aboveground_points"],
}
CANDIDATE_ATTRIBUTES = [a for g in ATTRIBUTE_GROUPS.values() for a in g]

PRETTY = {
    "cai_mean_over_occupied": "Mean CAI", "cai_max": "Max CAI",
    "mean_shared_fraction": "Shared crown fraction",
    "median_shared_fraction": "Shared crown fraction (median)",
    "mean_n_competitors": "Crown competitors",
    "shannon_index": "Shannon index", "berger_parker": "Berger-Parker dominance",
    "Broadleaved_prop": "Broadleaved proportion", "Species": "Species richness",
    "complexity_score": "Complexity score",
    "dbh_mean": "Mean DBH", "dbh_sd": "DBH SD", "dbh_p10": "DBH P10",
    "dbh_p90": "DBH P90", "h_mean": "Mean height", "h_sd": "Height SD",
    "h_p10": "Height P10", "h_p90": "Height P90",
    "n_trees_gt": "Trees per plot", "clarkevans": "Clark-Evans index",
    "aboveground_points": "Above-ground points",
}
GROUP_OF = {a: g for g, aa in ATTRIBUTE_GROUPS.items() for a in aa}

# One colour per attribute family, used consistently across every figure.
GROUP_COLOUR = {
    "Crown geometry": "#B45309",
    "Stand composition": "#2E6F73",
    "Size and structure": "#5B6B8C",
    "Density": "#7A6A55",
}
INK, MUTED, RULE = "#1F2933", "#6B7683", "#D9DEE3"


# ── loading ──────────────────────────────────────────────────────────────────

def load_run(results_dir):
    """Read a batch output directory. Returns (per_tree, per_plot, config)."""
    d = Path(results_dir)
    per_plot, cfg = read_results(d / "per_plot.csv")
    per_tree, cfg_t = read_results(d / "per_tree.csv")
    if cfg and cfg_t and cfg != cfg_t:
        raise TlsEvalError(
            f"{d}/per_tree.csv and per_plot.csv were produced under different "
            f"settings; they cannot be reported together.\n"
            f"  per_tree: {json.dumps(cfg_t, sort_keys=True)}\n"
            f"  per_plot: {json.dumps(cfg, sort_keys=True)}"
        )
    return per_tree, per_plot, cfg


def _normalise(names, strip_suffix=None):
    s = names.astype(str).str.replace(r"\.la[sz]$", "", regex=True)
    if strip_suffix:
        s = s.str.replace(re.escape(strip_suffix) + r"$", "", regex=True)
    return s


def join_attributes(per_plot, attributes_path, strip_suffix=None):
    """Join plot attributes on plot name.

    Exact match after dropping a .laz/.las extension and, if given, a trailing
    `strip_suffix`. Deliberately not fuzzy: silently attaching the wrong plot's
    attributes is worse than a failed join, and the message below says exactly
    what to pass.
    """
    att = pd.read_csv(attributes_path)
    key = "source_file" if "source_file" in att.columns else att.columns[0]
    att = att.copy()
    att["plot"] = _normalise(att[key])
    out = per_plot.copy()
    out["plot"] = _normalise(out["plot"], strip_suffix)

    merged = out.merge(att.drop(columns=[key]), on="plot", how="left")
    cols = [c for c in att.columns if c not in (key, "plot")]
    hit = merged[cols].notna().any(axis=1).sum()

    if hit == 0:
        got, want = out["plot"].iloc[0], att["plot"].iloc[0]
        hint = ""
        for cand in sorted(att["plot"], key=len, reverse=True):
            if got.startswith(cand) and len(got) > len(cand):
                hint = (f"\n  '{got}' starts with the plot name '{cand}'.\n"
                        f"  Retry with:  --strip-suffix '{got[len(cand):]}'")
                break
        raise TlsEvalError(
            f"no plot names matched between the results and {attributes_path}."
            f"\n  results look like:    {got}"
            f"\n  attributes look like: {want}{hint}"
        )
    if hit < len(out):
        missing = merged.loc[merged[cols].isna().all(axis=1), "plot"].head(3).tolist()
        print(f"note: attributes found for {hit}/{len(out)} plots "
              f"(missing e.g. {', '.join(missing)})")
    return merged


# ── analyses ─────────────────────────────────────────────────────────────────

def attribute_sensitivity(joined, metrics=None, attributes=None):
    """Spearman correlation of each plot attribute against each metric.

    Spearman rather than Pearson: several attributes are bounded or heavily
    skewed, and the relationships are monotone rather than linear.
    """
    from scipy.stats import spearmanr

    metrics = [m for m in (metrics or METRICS) if m in joined.columns]
    attributes = [a for a in (attributes or CANDIDATE_ATTRIBUTES) if a in joined.columns]
    rows = []
    for a in attributes:
        rec = {"attribute": a, "label": PRETTY.get(a, a),
               "group": GROUP_OF.get(a, "Other"),
               "n": int(joined[a].notna().sum())}
        for m in metrics:
            sub = joined[[a, m]].dropna()
            if len(sub) < 10:
                rec[f"rho_{m}"], rec[f"p_{m}"] = np.nan, np.nan
                continue
            r = spearmanr(sub[a], sub[m])
            rec[f"rho_{m}"], rec[f"p_{m}"] = r.statistic, r.pvalue
        rows.append(rec)
    out = pd.DataFrame(rows)
    if f"rho_{PRIMARY}" in out.columns:
        out = out.reindex(out[f"rho_{PRIMARY}"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def easy_vs_hard(joined, quantile=0.25, metric=PRIMARY):
    """Contrast the easiest and hardest plots, attribute by attribute.

    The paper's question in its most direct form: take the best and worst
    quarter of plots and ask what actually differs. Descriptive — the groups are
    defined by the outcome, so a difference is a correlate, not a cause.
    """
    from scipy.stats import mannwhitneyu

    sub = joined.dropna(subset=[metric])
    if len(sub) < 20:
        return None
    lo, hi = sub[metric].quantile(quantile), sub[metric].quantile(1 - quantile)
    hard, easy = sub[sub[metric] <= lo], sub[sub[metric] >= hi]
    rows = []
    for a in CANDIDATE_ATTRIBUTES:
        if a not in sub.columns:
            continue
        h, e = hard[a].dropna(), easy[a].dropna()
        if len(h) < 5 or len(e) < 5:
            continue
        pooled = np.concatenate([h, e]).std()
        rows.append({
            "attribute": a, "label": PRETTY.get(a, a), "group": GROUP_OF.get(a, "Other"),
            "easy_mean": e.mean(), "hard_mean": h.mean(),
            "difference": h.mean() - e.mean(),
            # Standardised so attributes on different scales can be ranked.
            "std_difference": (h.mean() - e.mean()) / pooled if pooled else np.nan,
            "mannwhitney_p": mannwhitneyu(h, e).pvalue,
        })
    out = pd.DataFrame(rows)
    if not len(out):
        return None
    out = out.reindex(out["std_difference"].abs().sort_values(ascending=False).index)
    out = out.reset_index(drop=True)
    out.attrs.update(n_easy=len(easy), n_hard=len(hard),
                     easy_iou=easy[metric].mean(), hard_iou=hard[metric].mean())
    return out


def stratify_2x2(joined, axis_a="cai_mean_over_occupied", axis_b="shannon_index",
                 metric=PRIMARY):
    """Median split on two attributes at once; every plot lands in one cell.

    Defaults to the paper's two dominant axes: crown stacking and species
    diversity.
    """
    if axis_a not in joined.columns or axis_b not in joined.columns:
        return None
    sub = joined[[axis_a, axis_b, metric]].dropna()
    if len(sub) < 40:
        return None
    ca, cb = sub[axis_a].median(), sub[axis_b].median()
    sub = sub.assign(_a=np.where(sub[axis_a] <= ca, "simple canopy", "layered canopy"),
                     _b=np.where(sub[axis_b] <= cb, "low diversity", "high diversity"))
    g = sub.groupby(["_a", "_b"])[metric].agg(["size", "mean", "std"]).reset_index()
    g.columns = ["canopy", "diversity", "n", metric, "sd"]
    g.attrs["cuts"] = {axis_a: ca, axis_b: cb}
    return g


def failure_profile(per_tree):
    """Missed / Split / Merged per 100 reference trees.

    Point-based, so unlike IoU these do not move with voxel size. Two methods
    reaching the same IoU by opposite routes — one fragmenting crowns, one
    fusing them — separate here and nowhere else.
    """
    return pd.DataFrame([{
        "n_trees": len(per_tree),
        "missed_per100": 100 * per_tree["missed"].mean(),
        "split_per100": 100 * per_tree["split"].mean(),
        "merged_per100": 100 * per_tree["merged"].mean(),
        "clean_pct": 100 * ((per_tree[["missed", "split", "merged"]].sum(axis=1) == 0).mean()),
    }])


def size_breakdown(per_tree, bins=(0, 2000, 10000, 40000, np.inf),
                   names=("tiny", "small", "medium", "large")):
    """Accuracy by reference-tree size. Errors concentrate in the small classes,
    which is where an inventory is least able to absorb them."""
    d = per_tree.copy()
    d["size_class"] = pd.cut(d["gt_voxel_count"], bins=bins, labels=list(names))
    return d.groupby("size_class", observed=True).agg(
        n=("iou", "size"), mean_iou=("iou", "mean"),
        detection_rate=("iou", lambda s: (s >= 0.5).mean()),
        missed_per100=("missed", lambda s: 100 * s.mean()),
        split_per100=("split", lambda s: 100 * s.mean()),
        merged_per100=("merged", lambda s: 100 * s.mean()),
    ).reset_index()


def extreme_plots(per_plot, n=10, metric=PRIMARY):
    """The hardest and easiest plots, for a look at what actually went wrong."""
    d = per_plot.dropna(subset=[metric]).sort_values(metric)
    cols = [c for c in ("plot", "n_trees", metric, "mean_iou_all", "detection_rate",
                        "missed_per100", "split_per100", "merged_per100")
            if c in d.columns]
    return pd.concat([d.head(n).assign(rank="hardest"),
                      d.tail(n).iloc[::-1].assign(rank="easiest")])[["rank"] + cols]


def compare_published(per_plot, published_path, strip_suffix=None, metric=PRIMARY):
    """Put your per-plot results next to the six published methods."""
    pub = pd.read_csv(published_path)
    pub["plot"] = _normalise(pub["source_file"])
    mine = per_plot.copy()
    mine["plot"] = _normalise(mine["plot"], strip_suffix)
    j = mine[["plot", metric]].merge(pub, on="plot", how="inner")
    if j.empty:
        return None
    cols = sorted(c for c in pub.columns if c.endswith("_mean_iou"))
    rows = [{"method": "your method", "mean_iou": j[metric].mean(),
             "n_plots": len(j), "you_win": ""}]
    for c in cols:
        n = int(j[c].notna().sum())
        rows.append({"method": c[:-len("_mean_iou")], "mean_iou": j[c].mean(),
                     "n_plots": n, "you_win": f"{int((j[metric] > j[c]).sum())}/{n}"})
    out = pd.DataFrame(rows).sort_values("mean_iou", ascending=False).reset_index(drop=True)
    out.attrs["metric"] = metric
    return out


# ── figures ──────────────────────────────────────────────────────────────────

def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
            "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": MUTED, "ytick.color": MUTED,
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.facecolor": "white", "savefig.facecolor": "white",
        })
        return plt
    except ImportError:
        return None


def make_figures(joined, per_tree, tables, out_dir):
    """Write the figure set. Skipped without matplotlib; tables still written."""
    plt = _mpl()
    if plt is None:
        print("note: matplotlib not installed, skipping figures")
        return []
    out = Path(out_dir)
    written = []

    def save(fig, name):
        p = out / name
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    rho = f"rho_{PRIMARY}"
    sens = tables.get("attribute_sensitivity")

    # 1. per-plot performance
    metrics = [m for m in METRICS if m in joined.columns]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 2.7))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, metrics):
            v = joined[m].dropna()
            ax.hist(v, bins=26, color="#2E6F73", edgecolor="white", linewidth=.4)
            ax.axvline(v.mean(), color="#B45309", linewidth=1.5)
            ax.set_title(m.replace("mean_iou_", "mIoU ").replace("_", " "))
            ax.set_xlabel(f"mean {v.mean():.3f}", color=MUTED, fontsize=8)
        axes[0].set_ylabel("plots")
        fig.suptitle("Per-plot performance", fontsize=11, y=1.03)
        save(fig, "01_metric_distributions.png")

    # 2. what makes a plot hard
    if sens is not None and rho in sens.columns and sens[rho].notna().any():
        s = sens.dropna(subset=[rho]).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.8, .3 * len(s) + 1.4))
        ax.barh(s["label"], s[rho],
                color=[GROUP_COLOUR.get(g, MUTED) for g in s["group"]], height=.7)
        ax.axvline(0, color=INK, linewidth=.9)
        ax.set_xlabel("Spearman ρ against plot mean IoU")
        ax.set_title("What makes a plot hard to segment")
        present = [g for g in ATTRIBUTE_GROUPS if g in set(s["group"])]
        if present:
            ax.legend([plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOUR[g]) for g in present],
                      present, loc="lower right", frameon=False, fontsize=8)
        save(fig, "02_attribute_sensitivity.png")

    # 3. strongest predictors as scatters
    if sens is not None and rho in sens.columns and sens[rho].notna().any():
        top = sens.dropna(subset=[rho]).head(4)
        if len(top):
            fig, axes = plt.subplots(1, len(top), figsize=(2.5 * len(top), 2.7), sharey=True)
            for ax, (_, r) in zip(np.atleast_1d(axes), top.iterrows()):
                sub = joined[[r["attribute"], PRIMARY]].dropna()
                ax.scatter(sub[r["attribute"]], sub[PRIMARY], s=13, alpha=.55,
                           color=GROUP_COLOUR.get(r["group"], MUTED), edgecolor="none")
                if len(sub) > 3:
                    z = np.polyfit(sub[r["attribute"]], sub[PRIMARY], 1)
                    xs = np.linspace(sub[r["attribute"]].min(), sub[r["attribute"]].max(), 40)
                    ax.plot(xs, np.polyval(z, xs), color=INK, linewidth=1.4)
                ax.set_xlabel(r["label"])
                ax.set_title(f"ρ = {r[rho]:+.2f}", fontsize=9, color=MUTED)
            np.atleast_1d(axes)[0].set_ylabel("plot mean IoU")
            fig.suptitle("Strongest correlates of segmentation difficulty",
                         fontsize=11, y=1.04)
            save(fig, "03_strongest_predictors.png")

    # 4. easy vs hard
    ev = tables.get("easy_vs_hard")
    if ev is not None and len(ev):
        s = ev.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.3, .34 * len(s) + 1.6))
        ax.barh(s["label"], s["std_difference"],
                color=[GROUP_COLOUR.get(g, MUTED) for g in s["group"]], height=.7)
        ax.axvline(0, color=INK, linewidth=.9)
        ax.set_xlabel("standardised difference   (hard − easy plots)")
        ax.set_title(f"Hardest vs easiest quarter of plots   "
                     f"(mIoU {ev.attrs.get('hard_iou', float('nan')):.2f} vs "
                     f"{ev.attrs.get('easy_iou', float('nan')):.2f})")
        save(fig, "04_easy_vs_hard.png")

    # 5. 2x2 stratification
    g2 = tables.get("stratified_2x2")
    if g2 is not None and len(g2) == 4:
        piv = g2.pivot(index="canopy", columns="diversity", values=PRIMARY)
        cnt = g2.pivot(index="canopy", columns="diversity", values="n")
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        im = ax.imshow(piv.values, cmap="YlGnBu_r", aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        mid = np.nanmean(piv.values)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.3f}\nn={int(cnt.values[i, j])}",
                        ha="center", va="center", fontsize=9.5,
                        color="white" if piv.values[i, j] < mid else INK)
        ax.set_title("Mean IoU by canopy structure and diversity")
        fig.colorbar(im, ax=ax, shrink=.82, label="mean IoU")
        save(fig, "05_stratified_2x2.png")

    # 6. failure profile and size breakdown
    prof = tables["failure_profile"].iloc[0]
    sb = tables["size_breakdown"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax1.bar(["Missed", "Split", "Merged"],
            [prof["missed_per100"], prof["split_per100"], prof["merged_per100"]],
            color=["#9F1239", "#B45309", "#2E6F73"], width=.6)
    ax1.set_ylabel("events per 100 reference trees")
    ax1.set_title(f"Failure profile   ({prof['clean_pct']:.0f}% clean)")
    if len(sb):
        ax2.bar(sb["size_class"].astype(str), sb["mean_iou"], color="#5B6B8C", width=.6)
        for i, r in sb.reset_index(drop=True).iterrows():
            ax2.text(i, r["mean_iou"] + .012, f"n={int(r['n'])}", ha="center",
                     fontsize=8, color=MUTED)
        ax2.set_ylim(0, min(1.0, float(sb["mean_iou"].max()) * 1.28))
        ax2.set_ylabel("mean IoU")
        ax2.set_title("Accuracy by reference-tree size")
    save(fig, "06_failures_and_size.png")

    # 7. against the published methods
    cp = tables.get("vs_published")
    if cp is not None and len(cp):
        fig, ax = plt.subplots(figsize=(5.6, .4 * len(cp) + 1.2))
        order = cp.iloc[::-1]
        ax.barh(order["method"], order["mean_iou"], height=.66,
                color=["#B45309" if m == "your method" else "#9AA5AE"
                       for m in order["method"]])
        for i, v in enumerate(order["mean_iou"]):
            ax.text(v + .006, i, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
        ax.set_xlim(0, float(order["mean_iou"].max()) * 1.16)
        ax.set_xlabel("mean IoU (matched trees)")
        ax.set_title("Against the published methods, same plots")
        save(fig, "07_vs_published.png")

    return written


# ── entry point ──────────────────────────────────────────────────────────────

def build(results_dir, attributes=None, published=None, out_dir="report",
          figures=True, strip_suffix=None):
    per_tree, per_plot, cfg = load_run(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = (join_attributes(per_plot, attributes, strip_suffix) if attributes
              else per_plot.copy())

    tables = {
        "per_plot": per_plot,
        "failure_profile": failure_profile(per_tree),
        "size_breakdown": size_breakdown(per_tree),
        "extreme_plots": extreme_plots(per_plot),
    }
    if attributes:
        tables["attribute_sensitivity"] = attribute_sensitivity(joined)
        ev = easy_vs_hard(joined)
        if ev is not None:
            tables["easy_vs_hard"] = ev
        g2 = stratify_2x2(joined)
        if g2 is not None:
            tables["stratified_2x2"] = g2
    if published:
        cp = compare_published(per_plot, published, strip_suffix)
        if cp is not None:
            tables["vs_published"] = cp

    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)

    figs = make_figures(joined, per_tree, tables, out) if figures else []
    _write_markdown(out, cfg, per_tree, per_plot, tables, figs)
    print(f"report written to {out}/   "
          f"({len(tables)} tables, {len(figs)} figures, summary.md)")
    return tables


def _write_markdown(out, cfg, per_tree, per_plot, tables, figs):
    L = ["# Evaluation report", ""]
    if cfg:
        L += ["```", "evaluation_config " + json.dumps(cfg, sort_keys=True), "```", ""]
    L += [f"**{len(per_plot)} plots, {len(per_tree)} reference trees.**", "",
          "| Metric | Plot mean |", "|---|---|"]
    for m in METRICS:
        if m in per_plot.columns:
            L.append(f"| {m.replace('_', ' ')} | {per_plot[m].mean():.3f} |")
    L += ["", "`mean iou matched` averages over trees that got a match, the convention "
          "behind the published table; `mean iou all` counts unmatched trees as zero.", ""]

    if "vs_published" in tables:
        L += ["## Against the published methods", "",
              "| Method | Mean IoU | Plots you win |", "|---|---|---|"]
        for _, r in tables["vs_published"].iterrows():
            L.append(f"| {r['method']} | {r['mean_iou']:.3f} | {r['you_win']} |")
        L.append("")

    prof = tables["failure_profile"].iloc[0]
    L += ["## How it fails", "",
          "Per 100 reference trees, computed on point shares, so these do not shift "
          "with voxel size.", "",
          "| Missed | Split | Merged | Clean |", "|---|---|---|---|",
          f"| {prof['missed_per100']:.1f} | {prof['split_per100']:.1f} | "
          f"{prof['merged_per100']:.1f} | {prof['clean_pct']:.0f}% |", ""]

    if "attribute_sensitivity" in tables:
        s = tables["attribute_sensitivity"].dropna(subset=[f"rho_{PRIMARY}"]).head(10)
        L += ["## What makes a plot hard", "", "Spearman ρ against plot mean IoU.", "",
              "| Attribute | Group | ρ | p |", "|---|---|---|---|"]
        for _, r in s.iterrows():
            L.append(f"| {r['label']} | {r['group']} | {r[f'rho_{PRIMARY}']:+.3f} "
                     f"| {r[f'p_{PRIMARY}']:.1e} |")
        L.append("")

    if "easy_vs_hard" in tables:
        ev = tables["easy_vs_hard"]
        L += ["## Easiest vs hardest plots", "",
              f"Best and worst quarter by mean IoU "
              f"({ev.attrs.get('easy_iou', float('nan')):.3f} vs "
              f"{ev.attrs.get('hard_iou', float('nan')):.3f}).", "",
              "| Attribute | Easy | Hard | Std. diff | p |", "|---|---|---|---|---|"]
        for _, r in ev.head(8).iterrows():
            L.append(f"| {r['label']} | {r['easy_mean']:.2f} | {r['hard_mean']:.2f} "
                     f"| {r['std_difference']:+.2f} | {r['mannwhitney_p']:.1e} |")
        L.append("")

    if "stratified_2x2" in tables:
        g = tables["stratified_2x2"]
        L += ["## By canopy structure and diversity", "",
              "Median split on both axes; every plot lands in one cell.", "",
              "| Canopy | Diversity | n | mean IoU |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r['canopy']} | {r['diversity']} | {int(r['n'])} "
                     f"| {r[PRIMARY]:.3f} |")
        L.append("")

    sb = tables["size_breakdown"]
    if len(sb):
        L += ["## Accuracy by tree size", "",
              "| Size | n | Mean IoU | Detection | Missed | Merged |",
              "|---|---|---|---|---|---|"]
        for _, r in sb.iterrows():
            L.append(f"| {r['size_class']} | {int(r['n'])} | {r['mean_iou']:.3f} | "
                     f"{r['detection_rate']:.3f} | {r['missed_per100']:.1f} | "
                     f"{r['merged_per100']:.1f} |")
        L.append("")

    ex = tables["extreme_plots"]
    if len(ex):
        L += ["## Hardest and easiest plots", "", "| | Plot | Trees | Mean IoU |",
              "|---|---|---|---|"]
        for _, r in ex[ex["rank"] == "hardest"].head(5).iterrows():
            L.append(f"| hardest | {r['plot']} | {int(r['n_trees'])} | {r[PRIMARY]:.3f} |")
        for _, r in ex[ex["rank"] == "easiest"].head(5).iterrows():
            L.append(f"| easiest | {r['plot']} | {int(r['n_trees'])} | {r[PRIMARY]:.3f} |")
        L.append("")

    if figs:
        L += ["## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs] + [""]
    (out / "summary.md").write_text("\n".join(L))
