"""The analysis that turns a results table into an assessment.

Answers the question the raw numbers do not: *what makes a plot hard*, and
*how does this method fail*, rather than only how well it scored.

    tlseval report results/ --attributes plot_attributes.csv --out report/

Everything is driven off the per-plot table written by `tlseval batch`, joined
to plot attributes by filename. Nothing is joined by tree ID -- annotation
passes renumber trees, and a tree-ID join silently pairs one tree's score with
another tree's measurements.

Figures are written only if matplotlib is installed; the tables always are.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .core import TlsEvalError, read_results

# Attributes the paper analysed, in the order the results section uses. Any of
# these missing from the attribute table is skipped rather than an error, so a
# partial attribute table still produces a partial report.
CANDIDATE_ATTRIBUTES = [
    "cai_mean_over_occupied", "cai_max", "mean_shared_fraction", "mean_n_competitors",
    "shannon_index", "berger_parker", "Broadleaved_prop", "clarkevans",
    "complexity_score", "n_trees_gt", "dbh_mean", "dbh_sd", "dbh_median",
    "h_mean", "h_sd", "h_median",
]

PRETTY = {
    "cai_mean_over_occupied": "Mean CAI",
    "cai_max": "Max CAI",
    "mean_shared_fraction": "Shared crown fraction",
    "mean_n_competitors": "Crown competitors",
    "shannon_index": "Shannon index",
    "berger_parker": "Berger-Parker dominance",
    "Broadleaved_prop": "Broadleaved proportion",
    "clarkevans": "Clark-Evans index",
    "complexity_score": "Complexity score",
    "n_trees_gt": "Trees per plot",
    "dbh_mean": "Mean DBH", "dbh_sd": "DBH SD", "dbh_median": "Median DBH",
    "h_mean": "Mean height", "h_sd": "Height SD", "h_median": "Median height",
}

METRICS = ["mean_iou", "detection_rate", "mean_precision", "mean_recall"]


# ── loading ──────────────────────────────────────────────────────────────────

def load_run(results_dir):
    """Read a batch output directory. Returns (per_tree, per_plot, config)."""
    d = Path(results_dir)
    per_plot, cfg = read_results(d / "per_plot.csv")
    per_tree, cfg_t = read_results(d / "per_tree.csv")
    if cfg and cfg_t and cfg != cfg_t:
        raise TlsEvalError(
            f"{d}/per_tree.csv and per_plot.csv were produced under "
            f"different settings; they cannot be reported together.\n"
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

    Matching is exact after dropping a .laz/.las extension and, if given, a
    trailing `strip_suffix`. Deliberately not fuzzy: a near-miss that silently
    attaches the wrong plot's attributes is far worse than a failed join, and
    the failure message below tells the caller exactly what to pass.
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
        # A common cause is a method suffix on the prediction filenames.
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
        print(f"warning: attributes found for {hit}/{len(out)} plots "
              f"(missing e.g. {', '.join(missing)})")
    return merged


# ── analyses ─────────────────────────────────────────────────────────────────

def attribute_sensitivity(joined, metrics=None, attributes=None):
    """Spearman correlation of each plot attribute against each metric.

    Spearman rather than Pearson because several attributes are bounded or
    heavily skewed, and the relationships are monotone rather than linear.
    """
    from scipy.stats import spearmanr

    metrics = [m for m in (metrics or METRICS) if m in joined.columns]
    attributes = [a for a in (attributes or CANDIDATE_ATTRIBUTES) if a in joined.columns]
    rows = []
    for a in attributes:
        rec = {"attribute": a, "label": PRETTY.get(a, a),
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
    if f"rho_mean_iou" in out.columns:
        out = out.reindex(out["rho_mean_iou"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def stratify(joined, by, metric="mean_iou", labels=("low", "high")):
    """Split plots at the median of `by` and compare the halves.

    Descriptive: the groups are observed, not assigned, so a difference is not
    an effect of `by` alone.
    """
    from scipy.stats import mannwhitneyu

    sub = joined[[by, metric]].dropna()
    if len(sub) < 20:
        return None
    cut = sub[by].median()
    lo, hi = sub[sub[by] <= cut][metric], sub[sub[by] > cut][metric]
    return {
        "attribute": by, "label": PRETTY.get(by, by), "median_split": cut,
        f"n_{labels[0]}": len(lo), f"n_{labels[1]}": len(hi),
        f"{metric}_{labels[0]}": lo.mean(), f"{metric}_{labels[1]}": hi.mean(),
        "difference": hi.mean() - lo.mean(),
        "mannwhitney_p": mannwhitneyu(lo, hi).pvalue if len(lo) and len(hi) else np.nan,
    }


def stratify_2x2(joined, axis_a, axis_b, metric="mean_iou"):
    """Median split on two attributes at once; every plot lands in one cell."""
    sub = joined[[axis_a, axis_b, metric]].dropna()
    if len(sub) < 40:
        return None
    ca, cb = sub[axis_a].median(), sub[axis_b].median()
    sub = sub.assign(
        _a=np.where(sub[axis_a] <= ca, "low", "high"),
        _b=np.where(sub[axis_b] <= cb, "low", "high"),
    )
    g = sub.groupby(["_a", "_b"])[metric].agg(["size", "mean"]).reset_index()
    g.columns = [PRETTY.get(axis_a, axis_a), PRETTY.get(axis_b, axis_b), "n", metric]
    g.attrs["cuts"] = {axis_a: ca, axis_b: cb}
    return g


def failure_profile(per_tree):
    """Missed / Split / Merged per 100 reference trees.

    Point-based, so unlike IoU these do not move with voxel size. Two methods
    reaching the same IoU by opposite routes -- one fragmenting crowns, one
    fusing them -- separate here and nowhere else.
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
    g = d.groupby("size_class", observed=True).agg(
        n=("iou", "size"), mean_iou=("iou", "mean"),
        detection_rate=("iou", lambda s: (s >= 0.5).mean()),
        missed_per100=("missed", lambda s: 100 * s.mean()),
        merged_per100=("merged", lambda s: 100 * s.mean()),
    ).reset_index()
    return g


def hardest_plots(per_plot, n=10):
    d = per_plot.sort_values("mean_iou")
    cols = [c for c in ("plot", "n_trees", "mean_iou", "detection_rate") if c in d.columns]
    return pd.concat([d.head(n).assign(rank="hardest"),
                      d.tail(n).iloc[::-1].assign(rank="easiest")])[["rank"] + cols]


# ── figures ──────────────────────────────────────────────────────────────────

def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def make_figures(joined, per_tree, sens, out_dir):
    """Write the standard figure set. Silently skipped without matplotlib."""
    plt = _mpl()
    if plt is None:
        print("note: matplotlib not installed, skipping figures")
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. metric distributions
    metrics = [m for m in METRICS if m in joined.columns]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.4))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, metrics):
            ax.hist(joined[m].dropna(), bins=28, color="#2E6F73", edgecolor="white", linewidth=.5)
            ax.axvline(joined[m].mean(), color="#B45309", linewidth=1.6)
            ax.set_title(m.replace("_", " "), fontsize=10)
            ax.set_xlabel(f"mean {joined[m].mean():.3f}", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("plots")
        fig.suptitle("Per-plot metric distributions", fontsize=11)
        fig.tight_layout()
        p = out / "metric_distributions.png"
        fig.savefig(p, dpi=160); plt.close(fig); written.append(p)

    # 2. attribute sensitivity
    if "rho_mean_iou" in sens.columns and sens["rho_mean_iou"].notna().any():
        s = sens.dropna(subset=["rho_mean_iou"]).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.4, .34 * len(s) + 1.4))
        colors = ["#B45309" if v < 0 else "#2E6F73" for v in s["rho_mean_iou"]]
        ax.barh(s["label"], s["rho_mean_iou"], color=colors, height=.68)
        ax.axvline(0, color="#4C5763", linewidth=.9)
        ax.set_xlabel("Spearman ρ against plot mean IoU")
        ax.set_title("What makes a plot hard", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = out / "attribute_sensitivity.png"
        fig.savefig(p, dpi=160); plt.close(fig); written.append(p)

    # 3. scatter against the strongest attribute
    if len(sens) and "rho_mean_iou" in sens.columns and sens["rho_mean_iou"].notna().any():
        top = sens.dropna(subset=["rho_mean_iou"]).iloc[0]
        a = top["attribute"]
        sub = joined[[a, "mean_iou"]].dropna()
        if len(sub) > 10:
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            ax.scatter(sub[a], sub["mean_iou"], s=17, alpha=.62, color="#2E6F73",
                       edgecolor="none")
            z = np.polyfit(sub[a], sub["mean_iou"], 1)
            xs = np.linspace(sub[a].min(), sub[a].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="#B45309", linewidth=1.7)
            ax.set_xlabel(top["label"]); ax.set_ylabel("plot mean IoU")
            ax.set_title(f"{top['label']}   ρ = {top['rho_mean_iou']:.2f}", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            p = out / "strongest_predictor.png"
            fig.savefig(p, dpi=160); plt.close(fig); written.append(p)

    # 4. failure taxonomy
    prof = failure_profile(per_tree).iloc[0]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.bar(["Missed", "Split", "Merged"],
           [prof["missed_per100"], prof["split_per100"], prof["merged_per100"]],
           color=["#9F1239", "#B45309", "#2E6F73"], width=.6)
    ax.set_ylabel("events per 100 reference trees")
    ax.set_title(f"Failure profile   ({prof['clean_pct']:.0f}% clean)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out / "failure_profile.png"
    fig.savefig(p, dpi=160); plt.close(fig); written.append(p)
    return written


# ── entry point ──────────────────────────────────────────────────────────────

def build(results_dir, attributes=None, out_dir="report", figures=True,
          strip_suffix=None):
    """Full report: tables always, figures when matplotlib is available."""
    per_tree, per_plot, cfg = load_run(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = (join_attributes(per_plot, attributes, strip_suffix) if attributes
              else per_plot.copy())

    tables = {
        "per_plot": per_plot,
        "failure_profile": failure_profile(per_tree),
        "size_breakdown": size_breakdown(per_tree),
        "extreme_plots": hardest_plots(per_plot),
    }
    sens = pd.DataFrame()
    if attributes:
        sens = attribute_sensitivity(joined)
        tables["attribute_sensitivity"] = sens
        strat = [s for a in CANDIDATE_ATTRIBUTES if a in joined.columns
                 for s in [stratify(joined, a)] if s]
        if strat:
            tables["median_split"] = pd.DataFrame(strat)
        if {"cai_mean_over_occupied", "shannon_index"} <= set(joined.columns):
            g = stratify_2x2(joined, "cai_mean_over_occupied", "shannon_index")
            if g is not None:
                tables["stratified_2x2"] = g

    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)

    figs = make_figures(joined, per_tree, sens, out) if figures else []
    _write_markdown(out, cfg, per_tree, per_plot, tables, figs)

    print(f"report written to {out}/")
    print(f"  {len(tables)} tables, {len(figs)} figures, summary.md")
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
    prof = tables["failure_profile"].iloc[0]
    L += ["", "## Failure profile", "",
          "Per 100 reference trees. Point-based, so these do not shift with voxel size.", "",
          "| Missed | Split | Merged | Clean |", "|---|---|---|---|",
          f"| {prof['missed_per100']:.1f} | {prof['split_per100']:.1f} | "
          f"{prof['merged_per100']:.1f} | {prof['clean_pct']:.0f}% |", ""]
    if "attribute_sensitivity" in tables:
        s = tables["attribute_sensitivity"].dropna(subset=["rho_mean_iou"]).head(8)
        L += ["## What makes a plot hard", "",
              "Spearman ρ against plot mean IoU.", "",
              "| Attribute | ρ | p |", "|---|---|---|"]
        for _, r in s.iterrows():
            L.append(f"| {r['label']} | {r['rho_mean_iou']:+.3f} | {r['p_mean_iou']:.1e} |")
        L.append("")
    sb = tables["size_breakdown"]
    if len(sb):
        L += ["## Accuracy by tree size", "",
              "| Size | n | Mean IoU | Detection |", "|---|---|---|---|"]
        for _, r in sb.iterrows():
            L.append(f"| {r['size_class']} | {int(r['n'])} | {r['mean_iou']:.3f} "
                     f"| {r['detection_rate']:.3f} |")
        L.append("")
    if figs:
        L += ["## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs] + [""]
    (out / "summary.md").write_text("\n".join(L))
