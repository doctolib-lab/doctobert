"""Plot the distribution of the quality score by domain."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset


def _auto_ylim(values, q_low=0.01, q_high=0.99, pad_frac=0.05):
    """
    Calcule des limites Y robustes via quantiles + un peu de marge.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None

    lo = np.quantile(v, q_low)
    hi = np.quantile(v, q_high)

    # fallback si quantiles identiques
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.min(v)
        hi = np.max(v)

    rng = hi - lo
    pad = rng * pad_frac if rng > 0 else (abs(hi) * pad_frac + 1e-6)

    lo2 = lo - pad
    hi2 = hi + pad

    # souvent on veut commencer à 0 si la métrique est positive
    if np.min(v) >= 0:
        lo2 = max(0.0, lo2)

    return (lo2, hi2)


def violin_medical_entity_density_by_class(
    ds,
    class_col: str = "health_domain_classification_best_class",
    value_col: str = "medical_entity_density",
    split: str | None = None,
    max_groups: int | None = None,
    y_lim=None,  # (0, 1),
    sort: str = "median_desc",  # "median_desc", "median_asc", "count_desc", "label_asc", or None
    figsize=(14, 8),
    rotate_xticks: int = 45,
    cmap_name: str = "plasma",  # close to your example, not black at start
    cut: float = 0,
    bw_adjust: float = 1.0,
    dropna: bool = True,
    # paper-friendly label overrides; None → fall back to column-name default
    # (or, for `title`, drop the auto title since the figure caption owns it)
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    save_path: str | None = None,
):
    import seaborn as sns
    import matplotlib.cm as cm

    # Pick split if needed
    if isinstance(ds, DatasetDict):
        if split is None:
            split = next(iter(ds.keys()))
        ds = ds[split]

    # To pandas
    df = ds.select_columns([class_col, value_col]).to_pandas()

    if dropna:
        df = df.dropna(subset=[class_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if dropna:
        df = df.dropna(subset=[value_col])

    if max_groups is not None:
        top = df[class_col].value_counts().head(max_groups).index
        df = df[df[class_col].isin(top)]

    # Ordering
    if sort == "median_desc":
        order = df.groupby(class_col)[value_col].median().sort_values(ascending=False).index.tolist()
    elif sort == "median_asc":
        order = df.groupby(class_col)[value_col].median().sort_values(ascending=True).index.tolist()
    elif sort == "count_desc":
        order = df[class_col].value_counts().index.tolist()
    elif sort == "label_asc":
        order = sorted(df[class_col].unique().tolist())
    else:
        order = df[class_col].unique().tolist()

    n = len(order)

    # Sequential colors left -> right (not starting from black)
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(x) for x in np.linspace(0.15, 0.95, n)]  # skip darkest end

    # Create 2x1 subplot: top for violin, bottom for document counts
    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]})

    sns.violinplot(
        data=df,
        x=class_col,
        y=value_col,
        order=order,
        inner="quartile",
        cut=cut,
        bw_adjust=bw_adjust,
        linewidth=1,
        palette=colors,
        ax=ax_top,
    )

    if title is not None:
        ax_top.set_title(title)
    ax_top.set_xlabel("")
    ax_top.set_ylabel(ylabel if ylabel is not None else value_col)

    if y_lim == "auto":
        y_lim = _auto_ylim(df[value_col], q_low=0.01, q_high=0.99, pad_frac=0.05)

    if y_lim is not None:
        ax_top.set_ylim(*y_lim)

    ax_top.grid(True, axis="y", alpha=0.25)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Bottom: document counts per class
    counts = df[class_col].value_counts().loc[order]
    x = np.arange(len(order))
    ax_bot.bar(x, counts.values)
    ax_bot.set_ylabel("Documents")
    ax_bot.set_xlabel(xlabel if xlabel is not None else class_col)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(order, rotation=rotate_xticks, ha="right")
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)

    fig.subplots_adjust(hspace=0.08)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Aggregated stats for return
    stats = df.groupby(class_col)[value_col].agg(["mean", "median", "size"]).loc[order]
    stats.columns = [f"mean_{value_col}", f"median_{value_col}", "n_docs"]

    return fig, ax_top, stats


def violin_medical_entity_density_before_after_by_class(
    ds,
    class_col: str = "health_domain_classification_best_class",
    original_col: str = "medical_entity_density",
    rewriting_col: str = "rewriting_medical_entity_density",
    split: str | None = None,
    max_groups: int | None = None,
    sort: str = "median_desc",  # median_desc/median_asc/count_desc/None
    y_lim=None,  # "auto" or (lo,hi) or None (for violin)
    figsize=(16, 7),
    rotate_xticks: int = 45,
    cmap_name: str = "plasma",
    dropna: bool = True,
    # violin geometry
    bw: float = 0.35,
    gap: float = 0.20,
    alpha_original: float = 0.35,
    alpha_rewriting: float = 0.85,
    # percentile bars on violins
    percentiles=(10, 25, 50, 75, 90),
    pct_bar_frac: float = 0.55,
    pct_color: str = "black",
    pct_alpha: float = 0.9,
    # paper-friendly label overrides; None → fall back to column-name default
    # (or, for `title`, drop the auto title since the figure caption owns it)
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    legend_pre: str | None = None,
    legend_post: str | None = None,
    save_path: str | None = None,
):
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    # Pick split if needed
    if isinstance(ds, DatasetDict):
        if split is None:
            split = next(iter(ds.keys()))
        ds = ds[split]

    df = ds.select_columns([class_col, original_col, rewriting_col]).to_pandas()
    df[original_col] = pd.to_numeric(df[original_col], errors="coerce")
    df[rewriting_col] = pd.to_numeric(df[rewriting_col], errors="coerce")

    if dropna:
        df = df.dropna(subset=[class_col])
        df = df.dropna(subset=[original_col, rewriting_col], how="all")

    if max_groups is not None:
        top = df[class_col].value_counts().head(max_groups).index
        df = df[df[class_col].isin(top)]

    # Order (sorted by rewriting median by default)
    if sort == "median_desc":
        order = df.groupby(class_col)[rewriting_col].median().sort_values(ascending=False).index.tolist()
    elif sort == "median_asc":
        order = df.groupby(class_col)[rewriting_col].median().sort_values(ascending=True).index.tolist()
    elif sort == "count_desc":
        order = df[class_col].value_counts().index.tolist()
    elif sort == "label_asc":
        order = sorted(df[class_col].unique().tolist())
    else:
        order = df[class_col].unique().tolist()

    n = len(order)

    # Sequential colors per class (for the violins only)
    cmap = cm.get_cmap(cmap_name)
    class_colors = [cmap(x) for x in np.linspace(0.15, 0.95, n)]

    # Counts (same order)
    counts = df[class_col].value_counts().reindex(order).fillna(0).astype(int)

    fig, (ax_v, ax_c) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    all_vals_for_ylim = []

    def _draw_pct_bars(ax, xpos: float, yvals: np.ndarray, width: float):
        if yvals.size == 0:
            return
        ps = np.percentile(yvals, list(percentiles))
        half = (width * pct_bar_frac) / 2.0
        for p, y in zip(percentiles, ps):
            lw = 2.0 if p == 50 else (1.4 if p in (25, 75) else 1.0)
            ax.hlines(y, xpos - half, xpos + half, colors=pct_color, linewidth=lw, alpha=pct_alpha, zorder=5)

    # --- Top: violins ---
    for i, (cls, color) in enumerate(zip(order, class_colors)):
        sub = df[df[class_col] == cls]

        v_orig = sub[original_col].to_numpy(dtype=float)
        v_orig = v_orig[np.isfinite(v_orig)]

        v_rew = sub[rewriting_col].to_numpy(dtype=float)
        v_rew = v_rew[np.isfinite(v_rew)]

        if v_orig.size > 0:
            all_vals_for_ylim.append(v_orig)
            xpos = i - gap
            parts = ax_v.violinplot(
                [v_orig], positions=[xpos], widths=bw, showmeans=False, showmedians=False, showextrema=False
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(alpha_original)
                body.set_linewidth(0.8)
            _draw_pct_bars(ax_v, xpos, v_orig, bw)

        if v_rew.size > 0:
            all_vals_for_ylim.append(v_rew)
            xpos = i + gap
            parts = ax_v.violinplot(
                [v_rew], positions=[xpos], widths=bw, showmeans=False, showmedians=False, showextrema=False
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(alpha_rewriting)
                body.set_linewidth(0.8)
            _draw_pct_bars(ax_v, xpos, v_rew, bw)

    if title is not None:
        ax_v.set_title(title)
    ax_v.set_ylabel(ylabel if ylabel is not None else "value")
    ax_v.grid(True, axis="y", alpha=0.25)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)

    if y_lim == "auto":
        if len(all_vals_for_ylim) > 0:
            y2 = _auto_ylim(np.concatenate(all_vals_for_ylim), q_low=0.01, q_high=0.99, pad_frac=0.05)
            if y2 is not None:
                ax_v.set_ylim(*y2)
    elif y_lim is not None:
        ax_v.set_ylim(*y_lim)

    # Legend (proxy patches)
    legend_handles = [
        Patch(facecolor="gray", edgecolor="black", alpha=alpha_original,
              label=legend_pre if legend_pre is not None else original_col),
        Patch(facecolor="gray", edgecolor="black", alpha=alpha_rewriting,
              label=legend_post if legend_post is not None else rewriting_col),
    ]
    ax_v.legend(handles=legend_handles, title="", loc="upper right")

    # --- Bottom: counts (plain bars) ---
    x = np.arange(n)
    ax_c.bar(x, counts.values)  # no per-class colors
    ax_c.set_ylabel("count")
    ax_c.set_xlabel(xlabel if xlabel is not None else class_col)
    ax_c.grid(True, axis="y", alpha=0.25)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ---- X ticks/labels: FORCE them to show on bottom axis ----
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(order, rotation=rotate_xticks, ha="right")
    ax_c.tick_params(axis="x", which="both", labelbottom=True)  # <-- key fix

    # Hide x labels on the top plot explicitly
    ax_v.tick_params(axis="x", which="both", labelbottom=False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax_v, ax_c), df, order, counts


def cdf_medical_entity_density_by_subset(
    ds,
    subset_col: str = "subset",
    value_col: str = "medical_entity_density",
    thresholds=(0.10, 0.20),
    mode: str = "retention",  # "retention" => P(X >= x); "cdf" => P(X <= x)
    log_x: bool = False,
    x_lim=None,
    figsize=(10, 6),
    cmap_name: str = "tab10",
    dropna: bool = True,
    save_path: str | None = None,
):
    import matplotlib.cm as cm

    if isinstance(ds, DatasetDict):
        ds = ds[next(iter(ds.keys()))]

    df = ds.select_columns([subset_col, value_col]).to_pandas()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if dropna:
        df = df.dropna(subset=[subset_col, value_col])

    # Preserve dataset insertion order for legibility of the legend
    subsets = list(dict.fromkeys(df[subset_col].tolist()))
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(subsets))]

    fig, ax = plt.subplots(figsize=figsize)

    retention = {}
    for sub, color in zip(subsets, colors):
        v = np.sort(df.loc[df[subset_col] == sub, value_col].to_numpy(dtype=float))
        v = v[np.isfinite(v)]
        n = v.size
        if n == 0:
            continue
        if mode == "retention":
            y = 1 - np.arange(1, n + 1) / n
        else:
            y = np.arange(1, n + 1) / n
        ax.plot(v, y, label=f"{sub} (n={n:,d})", color=color, linewidth=1.6)
        retention[sub] = {t: float((v >= t).mean()) for t in thresholds}

    for t in thresholds:
        ax.axvline(t, linestyle="--", color="0.4", linewidth=0.9, alpha=0.8)
        ax.text(t, 1.01, f"{t:.2f}", ha="center", va="bottom", fontsize=8, color="0.3")

    ax.set_xlabel(value_col)
    ax.set_ylabel("P(X ≥ x)" if mode == "retention" else "P(X ≤ x)")
    ax.set_title(
        f"{'Retention curve' if mode == 'retention' else 'CDF'} of {value_col} by {subset_col}"
    )
    ax.set_ylim(0, 1.02)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    stats = pd.DataFrame(retention).T
    stats.columns = [f"P(X≥{t:.2f})" for t in stats.columns]
    return fig, ax, stats


def print_density_bin_distribution(
    ds_table,
    value_col: str = "medical_entity_density",
    bins=(0, 0.05, 0.10, 0.20, 0.30, 1.0),
) -> None:
    """Print overall density-bin distribution as a markdown table.

    Bins are left-closed, right-open: [0, 0.05) / [0.05, 0.10) / … / [0.30, 1.0].
    """
    import pyarrow.compute as pc

    arr = ds_table[value_col].drop_null().to_pylist()
    total = len(arr)
    if total == 0:
        print(f"{value_col}: no non-null values.")
        return

    # Bin counts
    bins_list = list(bins)
    counts = [0] * (len(bins_list) - 1)
    for v in arr:
        for i in range(len(bins_list) - 1):
            if bins_list[i] <= v < bins_list[i + 1]:
                counts[i] += 1
                break

    print(f"\n{value_col} distribution (n={total:,}):")
    header = f"| {'Bin':<14} | {'Count':>10} | {'%':>7} | {'cumulative %':>13} |"
    sep = f"|{'-' * 16}|{'-' * 12}|{'-' * 9}|{'-' * 15}|"
    print(header)
    print(sep)
    cumulative = 0
    for i in range(len(bins_list) - 1):
        lo, hi = bins_list[i], bins_list[i + 1]
        n = counts[i]
        cumulative += n
        label = f"[{lo:.2f}, {hi:.2f})"
        pct = 100 * n / total
        cpct = 100 * cumulative / total
        print(f"| {label:<14} | {n:>10,} | {pct:>6.2f}% | {cpct:>12.2f}% |")


def print_density_bin_by_class(
    ds_table,
    class_col: str = "health_domain_classification_best_class",
    value_col: str = "medical_entity_density",
    bins=(0, 0.05, 0.10, 0.20, 0.30, 1.0),
) -> None:
    """Print density-bin × class stacked proportions as a markdown table.

    Mirrors the edu-quality stacked table in plot_edu_quality_distribution.py:
    rows = classes (sorted by share of highest bin, descending),
    columns = density bins.
    """
    if class_col not in ds_table.column_names:
        print(f"{class_col} not found in table — skipping density-bin-by-class table.")
        return

    import pandas as pd

    df = ds_table.select([class_col, value_col]).to_pandas()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[class_col, value_col])

    bins_list = list(bins)
    labels = [f"[{bins_list[i]:.2f},{bins_list[i+1]:.2f})" for i in range(len(bins_list) - 1)]

    df["bin"] = pd.cut(df[value_col], bins=bins_list, labels=labels, right=False)
    df = df.dropna(subset=["bin"])

    counts = (
        df.groupby([class_col, "bin"], observed=True).size()
        .unstack(fill_value=0)
    )
    # Ensure all bin columns are present in order
    for lab in labels:
        if lab not in counts.columns:
            counts[lab] = 0
    counts = counts[labels]

    # Sort by share of the second-to-last bin (≥0.10) — most discriminative
    high_bin = labels[-2]
    order = counts[high_bin].sort_values(ascending=False).index
    counts = counts.loc[order]

    props = counts.div(counts.sum(axis=1).replace(0, float("nan")), axis=0).fillna(0)
    totals = counts.sum(axis=1)

    print(f"\n{value_col} by {class_col} — proportions per bin (sorted by [{bins_list[-2]:.2f},{bins_list[-1]:.2f}) share):")
    col_header = " | ".join(f"{l:>14}" for l in labels)
    print(f"| {'class':<42} | {'n_docs':>8} | {col_header} |")
    sep = "|" + "-" * 44 + "|" + "-" * 10 + "|" + "|".join("-" * 16 for _ in labels) + "|"
    print(sep)
    for cls in order:
        n = int(totals[cls])
        prop_cells = " | ".join(f"{100 * props.loc[cls, l]:>13.1f}%" for l in labels)
        print(f"| {str(cls):<42} | {n:>8,} | {prop_cells} |")


def plot_density_bin_by_class(
    ds,
    class_col: str = "health_domain_classification_best_class",
    value_col: str = "medical_entity_density",
    bins=(0, 0.05, 0.10, 0.20, 0.30, 1.0),
    weight: str = "count",  # "count" or "num_words"
    num_words_col: str = "num_words",
    figsize=(14, 8),
    rotate_xticks: int = 45,
    title: str | None = None,
    xlabel: str = "Medical subdomain",
    legend_title: str = "Medical-term density bin",
    save_path: str | None = None,
):
    """Stacked-bar plot of density-bin proportions by class — mirrors
    `plot_edu_quality_distribution.plot_distribution_stacked`.

    Top panel: proportion of each density bin per class (stacked bars, sorted
    by share of the highest bin descending).
    Bottom panel: document (or word) counts per class.
    """
    bins_list = list(bins)
    labels = [f"[{bins_list[i]:.2f},{bins_list[i+1]:.2f})" for i in range(len(bins_list) - 1)]
    n_bins = len(labels)

    # Pull only what we need
    need = [class_col, value_col]
    if weight == "num_words":
        need.append(num_words_col)
    try:
        ds_small = ds.select_columns(need)
    except Exception:
        ds_small = ds

    df = ds_small.to_pandas()[need].dropna(subset=[class_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    df[class_col] = df[class_col].astype("category")

    df["bin"] = pd.cut(df[value_col], bins=bins_list, labels=labels, right=False)
    df = df.dropna(subset=["bin"])

    if weight == "count":
        df["w"] = 1.0
    else:
        df[num_words_col] = pd.to_numeric(df[num_words_col], errors="coerce").fillna(0.0)
        df["w"] = df[num_words_col].astype(float)

    counts = (
        df.groupby([class_col, "bin"], observed=True, sort=False)["w"]
        .sum()
        .unstack(fill_value=0.0)
    )
    for lab in labels:
        if lab not in counts.columns:
            counts[lab] = 0.0
    counts = counts[labels]

    # Sort by share of highest bin (last label)
    props = counts.div(counts.sum(axis=1).replace(0, float("nan")), axis=0).fillna(0)
    order = props[labels[-1]].sort_values(ascending=False).index
    props = props.loc[order]
    totals = counts.sum(axis=1).loc[order]

    x = np.arange(len(props))
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # Viridis reversed: low bin (bottom) = light yellow, high bin (top) = dark purple.
    cmap = plt.cm.get_cmap("viridis_r")
    colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    bottoms = np.zeros(len(props))
    for i, lab in enumerate(labels):
        ax_top.bar(x, props[lab].values, bottom=bottoms, label=lab, color=colors[i])
        bottoms += props[lab].values

    ax_top.set_ylabel("Proportion of documents" if weight == "count" else "Proportion of words")
    ax_top.set_ylim(0, 1)
    if title is not None:
        ax_top.set_title(title)
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)
    handles, hlabels = ax_top.get_legend_handles_labels()
    handles, hlabels = handles[::-1], hlabels[::-1]
    ax_top.legend(
        handles, hlabels,
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    ax_bot.bar(x, totals.values)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)
    ax_bot.set_ylabel("Documents" if weight == "count" else "Words")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(order, rotation=rotate_xticks, ha="right")

    fig.subplots_adjust(right=0.82, hspace=0.08)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    return props.round(4), counts.round(0).astype("int64")


# ----------------------------------------------------------------------------
# Entry point 1: per-dataset raw profiling
#   Compare medical_entity_density distributions across raw (un-rewritten) source
#   datasets, grouped by health subdomain / edu score / dataset subset.
#   Usage: python plot_term_density_distribution.py per_dataset --save_dir ...
# ----------------------------------------------------------------------------


def main_per_dataset(num_workers: int = 4, save_dir: str | None = None, max_samples: int | None = None):
    """Profile raw medical-entity-density across all source corpora."""
    datasets = []
    # fmt: off
    # raw data (un-rewritten source corpora)
    dataset_paths = [
        ("fineweb-2", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner"),
        ("finepdfs", "/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner"),
        ("finewiki", "/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner"),
        # ("nachos", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner"),
        # ("transcorpus-bio-fr", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified_extracted_gliner"),
        # ("mmc", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/multilingual_medical_corpus/health_domain_classified_edu_quality_scored_extracted_gliner"),
        # ("e3c", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/E3C/layer3/health_domain_classified_edu_quality_scored_extracted_gliner"),
        # ("synthesized", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/final/v2_extracted_gliner"),
    ]
    # fmt: on

    class_col = "health_domain_classification_best_class"
    edu_class_col = "edu_quality_normalized_score"
    value_col = "medical_entity_density"

    for name, dataset_path in dataset_paths:
        ds = load_dataset(dataset_path, split="train", num_proc=num_workers)

        if max_samples is not None:
            ds = ds.shuffle(seed=42).select(range(min(max_samples, ds.num_rows)))
            print(f"Sampled the first {ds.num_rows:,d} examples")

        # Keep only the columns plotted in this entry point
        ds = ds.remove_columns(
            list(set(ds.column_names) - {value_col, class_col, edu_class_col})
        )
        ds = ds.add_column("subset", [name] * ds.num_rows)

        print("=" * 100)
        print(f"Plotting {name} with {ds.num_rows:,d} examples...")
        print("=" * 100)

        # tmp: skip synthesized when comparing real source datasets
        if name == "synthesized":
            continue

        # _, _, stats = violin_medical_entity_density_by_class(
        #     ds,
        #     class_col=class_col,
        #     value_col=value_col,
        #     y_lim="auto",
        #     bw_adjust=0.9,
        #     xlabel="Medical subdomain",
        #     ylabel="Medical-term density",
        #     save_path=f"{save_dir}/{name}_medical_entity_density_by_domain.png" if save_dir else None,
        # )
        # print(stats.to_markdown())

        # _, _, stats = violin_medical_entity_density_by_class(
        #     ds,
        #     class_col=edu_class_col,
        #     value_col=value_col,
        #     sort="label_asc",
        #     y_lim="auto",
        #     bw_adjust=0.9,
        #     xlabel="Educational quality",
        #     ylabel="Medical-term density",
        #     save_path=f"{save_dir}/{name}_medical_entity_density_by_edu_quality.png" if save_dir else None,
        # )
        # print(stats.to_markdown())

        ds_table = ds.with_format("arrow")[:].select(
            [c for c in [value_col, class_col] if c in ds.column_names]
        )
        print_density_bin_distribution(ds_table, value_col=value_col)
        print_density_bin_by_class(ds_table, class_col=class_col, value_col=value_col)
        props, counts = plot_density_bin_by_class(
            ds,
            class_col=class_col,
            value_col=value_col,
            xlabel="Medical subdomain",
            save_path=f"{save_dir}/{name}_density_bin_by_domain.png" if save_dir else None,
        )
        print(counts.to_markdown())
        print(props.to_markdown())

        datasets.append(ds)

    dataset = concatenate_datasets(datasets)
    print(dataset)

    # print("=" * 100)
    # print("Plotting all (by domain)...")
    # print("=" * 100)
    # _, _, stats = violin_medical_entity_density_by_class(
    #     dataset, class_col=class_col, value_col=value_col,
    #     y_lim="auto", bw_adjust=0.9,
    #     xlabel="Medical subdomain",
    #     ylabel="Medical-term density",
    #     save_path=f"{save_dir}/all_medical_entity_density_by_domain.png" if save_dir else None,
    # )
    # print(stats.to_markdown())

    # print("=" * 100)
    # print("Plotting all (by edu quality)...")
    # print("=" * 100)
    # _, _, stats = violin_medical_entity_density_by_class(
    #     dataset, class_col=edu_class_col, value_col=value_col,
    #     sort="label_asc", y_lim="auto", bw_adjust=0.9,
    #     xlabel="Educational quality",
    #     ylabel="Medical-term density",
    #     save_path=f"{save_dir}/all_medical_entity_density_by_edu_quality.png" if save_dir else None,
    # )
    # print(stats.to_markdown())

    # print("=" * 100)
    # print("Plotting all (medterm retention CDF by subset)...")
    # print("=" * 100)
    # _, _, stats = cdf_medical_entity_density_by_subset(
    #     dataset,
    #     subset_col="subset",
    #     value_col=value_col,
    #     thresholds=(0.10, 0.20),
    #     x_lim=(0, 0.6),
    #     save_path=f"{save_dir}/all_medical_entity_density_cdf_by_subset.png" if save_dir else None,
    # )
    # print(stats.to_markdown())

    # # tmp: add synthesized dataset
    # dataset = concatenate_datasets([dataset, ds])

    # print("=" * 100)
    # print("Plotting all (by subset)...")
    # print("=" * 100)
    # _, _, stats = violin_medical_entity_density_by_class(
    #     dataset, class_col="subset", value_col=value_col,
    #     y_lim="auto", bw_adjust=0.9,
    #     xlabel="Source dataset",
    #     ylabel="Medical-term density",
    #     save_path=f"{save_dir}/all_medical_entity_density_by_subset.png" if save_dir else None,
    # )
    # print(stats.to_markdown())

    print("=" * 100)
    print("Density bin distribution (all corpora combined)...")
    print("=" * 100)
    all_table = dataset.with_format("arrow")[:].select(
        [c for c in [value_col, class_col] if c in dataset.column_names]
    )
    print_density_bin_distribution(all_table, value_col=value_col)
    print_density_bin_by_class(all_table, class_col=class_col, value_col=value_col)
    props, counts = plot_density_bin_by_class(
        dataset,
        class_col=class_col,
        value_col=value_col,
        xlabel="Medical subdomain",
        save_path=f"{save_dir}/all_density_bin_by_domain.png" if save_dir else None,
    )
    print(counts.to_markdown())
    print(props.to_markdown())


# ----------------------------------------------------------------------------
# Entry point 2: before/after rewriting analysis
#   Compare source-doc extraction (`original_medical_entity_density`) vs rewritten-doc
#   extraction (`medical_entity_density`), plus char_compression_ratio (rewritten/source length).
#   Operates on the postprocessed V4.x rewritten datasets.
#   Usage: python plot_term_density_distribution.py rewriting --save_dir ...
# ----------------------------------------------------------------------------


def main_rewriting(num_workers: int = 4, save_dir: str | None = None, max_samples: int | None = None):
    """Before/after-rewriting profiling on V4.x postprocessed + edu-rescored datasets.

    Loads the `_edu_quality_scored/` outputs of `run_dataset_classifier_array` (waves
    868063 / 1044323 / 868064 — see docs/18_final_dataset.md), which carry BOTH
    sets of original_* columns:
      - `original_medical_entity_density`  (renamed by postprocess_extract.py)
      - `original_edu_quality_normalized_score`  (renamed by run_dataset_classifier.py)
    so we can do a paired before/after on med density AND edu score in one pass.
    """
    datasets = []
    # fmt: off
    # V4.x rewritten + edu-rescored data (with original_* columns from BOTH
    # postprocess_extract.py and run_dataset_classifier.py renames)
    dataset_paths = [
        ("fineweb-2", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_edu_quality_scored"),
        ("finepdfs", "/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_10shards_edu_quality_scored"),
        ("finewiki", "/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_edu_quality_scored"),
    ]
    # fmt: on

    class_col = "health_domain_classification_best_class"
    edu_class_col = "edu_quality_normalized_score"
    value_col = "medical_entity_density"
    original_value_col = "original_medical_entity_density"
    # Use the unnormalized regression score for the before/after violin: the
    # normalized version is bucketed into 0-5 integers and the violin collapses
    # into horizontal stripes. The continuous raw score is much more readable.
    edu_value_col = "edu_quality_score"
    original_edu_value_col = "original_edu_quality_score"

    for name, dataset_path in dataset_paths:
        ds = load_dataset(dataset_path, split="train", num_proc=num_workers)

        if max_samples is not None:
            ds = ds.shuffle(seed=42).select(range(min(max_samples, ds.num_rows)))
            print(f"Sampled the first {ds.num_rows:,d} examples")

        # tmp: derive before/after metrics; flatten rewriting model name from rewriting_config struct
        def _extract_rewritting_style(x):
            text, orig_text = x["text"], x.get("original_text")
            new_d, orig_d = x["medical_entity_density"], x["original_medical_entity_density"]
            new_e, orig_e = x.get(edu_value_col), x.get(original_edu_value_col)
            num_w, orig_num_w = x.get("num_words"), x.get("original_num_words")

            char_compression_ratio = (len(text) / len(orig_text)) if text and orig_text else None
            word_compression_ratio = (num_w / orig_num_w) if num_w and orig_num_w else None
            medical_entity_density_diff = (new_d - orig_d) if new_d is not None and orig_d is not None else None
            # multiplicative complement; helpful when interpreting "rewriting doubled density"
            medical_entity_density_ratio = (new_d / orig_d) if new_d is not None and orig_d not in (None, 0) else None
            edu_quality_score_diff = (
                (new_e - orig_e) if new_e is not None and orig_e is not None else None
            )
            rewriting_model = (x.get("rewriting_config") or {}).get("model")
            return {
                "char_compression_ratio": char_compression_ratio,
                "word_compression_ratio": word_compression_ratio,
                "medical_entity_density_diff": medical_entity_density_diff,
                "medical_entity_density_ratio": medical_entity_density_ratio,
                "edu_quality_score_diff": edu_quality_score_diff,
                "rewriting_model": rewriting_model,
            }

        ds = ds.map(_extract_rewritting_style, num_proc=num_workers)

        # Keep only the columns plotted in this entry point
        ds = ds.remove_columns(
            list(
                set(ds.column_names)
                - {
                    value_col,
                    class_col,
                    edu_class_col,
                    original_value_col,
                    edu_value_col,
                    original_edu_value_col,
                    "char_compression_ratio",
                    "word_compression_ratio",
                    "medical_entity_density_diff",
                    "medical_entity_density_ratio",
                    "edu_quality_score_diff",
                    "rewriting_model",
                }
            )
        )
        ds = ds.add_column("subset", [name] * ds.num_rows)

        print("=" * 100)
        print(f"Plotting {name} with {ds.num_rows:,d} examples...")
        print("=" * 100)

        # tmp: rewriting stats — mean / std / quantiles / frac-positive (for diff-style cols)
        import pyarrow.compute as pc

        arrow_table = ds.with_format("arrow")[:]
        quantiles = [0.25, 0.5, 0.75, 0.9]
        diff_cols = {  # cols where sign is meaningful — also emit frac>0
            "medical_entity_density_diff",
            "edu_quality_score_diff",
        }
        for col in [
            "char_compression_ratio",
            "word_compression_ratio",
            "medical_entity_density",
            original_value_col,
            "medical_entity_density_diff",
            "medical_entity_density_ratio",
            edu_value_col,
            original_edu_value_col,
            "edu_quality_score_diff",
        ]:
            if col not in arrow_table.column_names:
                continue
            arr = arrow_table[col]
            mean = pc.mean(arr).as_py()
            stddev = pc.stddev(arr).as_py()
            n = arr.null_count
            n_valid = len(arr) - n
            qs = pc.quantile(arr, q=quantiles).to_pylist()
            extra = ""
            if col in diff_cols:
                # fraction of docs where rewriting strictly raised the value (sign>0)
                frac_pos = pc.mean(pc.greater(arr, 0)).as_py()
                extra = f" | frac>0={frac_pos:.3f}" if frac_pos is not None else ""
            print(
                f"{col}: mean={mean:.4f} std={stddev:.4f} n={n_valid:,}"
                + " | "
                + " ".join(f"q{int(q*100)}={round(v, 4)}" for q, v in zip(quantiles, qs))
                + extra
            )

        _, _, stats = violin_medical_entity_density_by_class(
            ds,
            class_col=class_col,
            value_col="char_compression_ratio",
            y_lim="auto",
            bw_adjust=0.9,
            xlabel="Medical subdomain",
            ylabel="Character compression ratio (post / pre)",
            save_path=f"{save_dir}/{name}_char_compression_ratio_by_domain.png" if save_dir else None,
        )
        print(stats.to_markdown())

        # _, _, stats = violin_medical_entity_density_by_class(
        #     ds,
        #     class_col="rewriting_model",
        #     value_col="char_compression_ratio",
        #     y_lim="auto",
        #     bw_adjust=0.9,
        #     save_path=f"{save_dir}/{name}_char_compression_ratio_by_rewriting_model.png" if save_dir else None,
        # )
        # print(stats.to_markdown())

        # tmp: med-density before/after already plotted in a prior run — disable
        # to keep this invocation focused on the edu re-score comparison only.
        violin_medical_entity_density_before_after_by_class(
            ds,
            class_col=class_col,
            original_col=original_value_col,
            rewriting_col=value_col,
            y_lim="auto",
            percentiles=(50,),
            legend_pre="Pre-rephrasing",
            legend_post="Post-rephrasing",
            xlabel="Medical subdomain",
            ylabel="Medical-term density",
            save_path=f"{save_dir}/{name}_medical_entity_density_before_after_by_domain.png" if save_dir else None,
        )

        # Mirror the med-density before/after for edu_quality_score (unnormalized
        # regression output; normalized 0-5 violin collapses into stripes).
        # Function is named for med density but accepts arbitrary numeric cols.
        violin_medical_entity_density_before_after_by_class(
            ds,
            class_col=class_col,
            original_col=original_edu_value_col,
            rewriting_col=edu_value_col,
            y_lim="auto",
            percentiles=(50,),
            legend_pre="Pre-rephrasing",
            legend_post="Post-rephrasing",
            xlabel="Medical subdomain",
            ylabel="Educational quality",
            save_path=f"{save_dir}/{name}_edu_quality_score_before_after_by_domain.png" if save_dir else None,
        )

        datasets.append(ds)

    if not datasets:
        print("No datasets loaded; exiting.")
        return

    dataset = concatenate_datasets(datasets)
    print(dataset)

    print("=" * 100)
    print("Plotting all (before/after medical_entity_density by domain)...")
    print("=" * 100)
    violin_medical_entity_density_before_after_by_class(
        dataset,
        class_col=class_col,
        original_col=original_value_col,
        rewriting_col=value_col,
        y_lim="auto",
        percentiles=(50,),
        legend_pre="Pre-rephrasing",
        legend_post="Post-rephrasing",
        xlabel="Medical subdomain",
        ylabel="Medical-term density",
        save_path=f"{save_dir}/all_medical_entity_density_before_after_by_domain.png" if save_dir else None,
    )

    print("=" * 100)
    print("Plotting all (before/after edu_quality_score by domain)...")
    print("=" * 100)
    violin_medical_entity_density_before_after_by_class(
        dataset,
        class_col=class_col,
        original_col=original_edu_value_col,
        rewriting_col=edu_value_col,
        y_lim="auto",
        percentiles=(50,),
        legend_pre="Pre-rephrasing",
        legend_post="Post-rephrasing",
        xlabel="Medical subdomain",
        ylabel="Educational quality",
        save_path=f"{save_dir}/all_edu_quality_score_before_after_by_domain.png" if save_dir else None,
    )


# ----------------------------------------------------------------------------
# Legacy single-entry interface (kept for slurm/CLI compatibility — delegates
# to `main_per_dataset` by default; flip the call to switch entry points).
# ----------------------------------------------------------------------------


def main(dataset_path: str | None = None, num_workers: int = 4, save_dir: str | None = None, max_samples: int | None = None):
    main_per_dataset(num_workers=num_workers, save_dir=save_dir, max_samples=max_samples)
    # main_rewriting(num_workers=num_workers, save_dir=save_dir, max_samples=max_samples)


if __name__ == "__main__":
    import fire

    fire.Fire({
        "main": main,
        "per_dataset": main_per_dataset,
        "rewriting": main_rewriting,
    })
