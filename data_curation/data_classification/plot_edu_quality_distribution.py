"""Plot the distribution of the quality score by domain."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset


def plot_distribution_stacked(
    ds,
    domain_column: str,
    quality_column: str,
    num_words_column: str,
    weight: Literal["count", "num_words"] = "count",
    figsize=(14, 8),
    rotate_xticks=45,
    # paper-friendly label overrides; defaults pick the terms used in the paper.
    # Pass `title=None` (the default) to drop the auto title — the figure caption
    # owns it. Override any to customise.
    title: str | None = None,
    xlabel: str = "Medical subdomain",
    legend_title: str = "Educational quality",
    save_path: str | None = None,
):
    """Plot the distribution of the quality score by domain.

    Creates a 2x1 figure:
      - Top: stacked proportions by score.
      - Bottom: total volume by domain (documents or words, matching `weight`).
    """
    # Minimize columns at the datasets level to save RAM before converting to pandas
    need = [domain_column, quality_column]
    if weight == "num_words":
        need.append(num_words_column)
    try:
        ds_small = ds.select_columns(need)
    except Exception:
        ds_small = ds

    # Convert only required columns to pandas
    df = ds_small.to_pandas()[need].dropna()

    # Quality: coerce to 0..5 integers and make categorical with fixed categories
    q = pd.to_numeric(df[quality_column], errors="coerce").round().clip(0, 5).astype("Int64")
    valid_mask = q.notna()
    if not bool(valid_mask.all()):
        df = df.loc[valid_mask].copy()
        q = q.loc[valid_mask]
    df[quality_column] = pd.Categorical(q.astype("int8"), categories=list(range(6)), ordered=True)

    # Domain as categorical to reduce memory and speed up groupby
    df[domain_column] = df[domain_column].astype("category")

    # Weight vector
    if weight == "count":
        df["w"] = 1.0
    else:
        df[num_words_column] = pd.to_numeric(df[num_words_column], errors="coerce").fillna(0.0)
        df["w"] = df[num_words_column].astype(float)

    # Fast aggregation: observed combinations only, no sort
    counts = (
        df.groupby([domain_column, quality_column], observed=True, sort=False)["w"]
        .sum()
        .unstack(level=quality_column, fill_value=0.0)
    )
    # Ensure all columns 0..5 exist and are in order
    for s in range(6):
        if s not in counts.columns:
            counts[s] = 0.0
    counts = counts[[0, 1, 2, 3, 4, 5]]
    # count order
    order = counts[5].sort_values(ascending=False).index
    counts = counts.loc[order]

    # Proportions for top panel
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Sort by share of 4–5
    # order = props.loc[:, 4:].sum(axis=1).sort_values(ascending=False).index
    # Single consistent sort order by absolute number of score-5 items
    order = props[5].sort_values(ascending=False).index
    props = props.loc[order]
    totals = counts.sum(axis=1).loc[order]  # volume for bottom panel

    # Plot
    x = np.arange(len(props))
    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]})

    # Color palette for stacks
    cmap = plt.cm.get_cmap("Set2")
    colors = [cmap(i) for i in range(6)]
    colors = colors[::-1]

    # Top: stacked proportions
    bottoms = np.zeros(len(props))
    for score in range(6):
        ax_top.bar(
            x,
            props[score].values,
            bottom=bottoms,
            label=f"Score {score}",
            color=colors[score],
        )
        bottoms += props[score].values

    ax_top.set_ylabel("Proportion of documents" if weight == "count" else "Proportion of words")
    ax_top.set_ylim(0, 1)
    if title is not None:
        ax_top.set_title(title)
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)
    # Legend (sorted so 5 is at top)
    handles, labels = ax_top.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    ax_top.legend(
        handles,
        labels,
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    # Bottom: totals per domain
    ax_bot.bar(x, totals.values)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)
    ax_bot.set_ylabel("Documents" if weight == "count" else "Words")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(props.index, rotation=rotate_xticks, ha="right")

    # Make space for the legend on the right and a little space between rows
    fig.subplots_adjust(right=0.82, hspace=0.08)
    plt.tight_layout()
    plt.show()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=True,
        )

    # return props rounded to 4 decimals
    return props.round(4), counts.round(0).astype("int64").applymap(lambda n: f"{n:,}")


def print_edu_stats_by_domain(dataset, domain_column: str, quality_column: str):
    """Print mean + median edu quality score per subdomain as markdown."""
    df = dataset.select_columns([domain_column, quality_column]).to_pandas()
    df[quality_column] = pd.to_numeric(df[quality_column], errors="coerce")
    df = df.dropna(subset=[domain_column, quality_column])
    stats = df.groupby(domain_column, observed=True)[quality_column].agg(["mean", "median", "size"])
    stats.columns = ["mean_edu", "median_edu", "n_docs"]
    stats = stats.sort_values("mean_edu", ascending=False)
    print(stats.to_markdown())


def plot_all(
    dataset,
    name: str,
    domain_column: str,
    domain_label: str,
    save_dir: str | None,
    max_samples: int | None = None,
    domain_display_name: str | None = None,
):
    """Plot edu-quality distribution grouped by `domain_column` for a given dataset.

    `domain_label` is a filename-safe slug (used in the saved PNG path).
    `domain_display_name` is the paper-friendly xlabel shown on the plot
    (e.g. "Medical subdomain", "Source dataset"); defaults to `domain_label`.
    """
    if domain_display_name is None:
        domain_display_name = domain_label

    if max_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, dataset.num_rows)))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    print("=" * 100)
    print(f"Plotting {name} by {domain_label}...")
    print("=" * 100)

    quality_column = "edu_quality_normalized_score"

    print_edu_stats_by_domain(dataset, domain_column, quality_column)

    for weight in ("count", "num_words"):
        props, counts = plot_distribution_stacked(
            dataset,
            domain_column=domain_column,
            quality_column=quality_column,
            num_words_column="num_words",
            weight=weight,
            xlabel=domain_display_name,
            save_path=f"{save_dir}/{name}_edu_quality_by_{domain_label}_{weight}.png" if save_dir else None,
        )
        print(counts.to_markdown())
        print(props.to_markdown())

    # tmp: subset x edu quality score
    # domain_column = "subset"
    # props, counts = plot_distribution_stacked(
    #     dataset,
    #     domain_column=domain_column,
    #     quality_column=quality_column,
    #     num_words_column="num_words",
    #     weight="count",
    #     save_path="images/edu_quality_by_subset_count.png",
    # )
    # print(counts.to_markdown())
    # print(props.to_markdown())

    # props, counts = plot_distribution_stacked(
    #     dataset,
    #     domain_column=domain_column,
    #     quality_column=quality_column,
    #     num_words_column="num_words",
    #     weight="num_words",
    #     save_path="images/edu_quality_by_subset_num_words.png",
    # )
    # print(counts.to_markdown())
    # print(props.to_markdown())


def main(dataset_path: str | None = None, num_workers: int = 4, save_dir: str | None = None, max_samples: int | None = None):
    """Edu-quality stacked-bar profiling across the source corpora.

    `save_dir` is a CLI argument (no on-disk save when omitted) — matches
    the entry-point convention in `plot_term_density_distribution.py`.
    """
    # dataset = load_dataset(
    #     dataset_path,
    #     split="train",
    #     num_proc=num_workers,
    # )
    # print(dataset)

    # tmp
    # datasets = []
    # fmt: off
    dataset_paths = [
        ("fineweb-2", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored"),
        ("finepdfs", "/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored"),
        ("finewiki", "/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored"),
        # ("nachos", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored"),
        # ("mmc", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/multilingual_medical_corpus/health_domain_classified_edu_quality_scored"),
        # ("e3c", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/E3C/layer3/health_domain_classified_edu_quality_scored"),
        # ("transcorpus_bio_fr", "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified"),
    ]
    # fmt: on

    subtopic_column = "health_domain_classification_best_class"

    datasets = []
    for name, dataset_path in dataset_paths:
        dataset = load_dataset(dataset_path, split="train", num_proc=num_workers)
        dataset = dataset.add_column("subset", [name] * dataset.num_rows)

        plot_all(
            dataset, name, subtopic_column, "subtopic", save_dir, max_samples,
            domain_display_name="Medical subdomain",
        )
        datasets.append(dataset)

    all_dataset = concatenate_datasets(datasets)
    plot_all(
        all_dataset, "all", subtopic_column, "subtopic", save_dir, max_samples,
        domain_display_name="Medical subdomain",
    )
    plot_all(
        all_dataset, "all", "subset", "subset", save_dir, max_samples,
        domain_display_name="Source dataset",
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
