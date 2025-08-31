from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal
from datasets import load_dataset


def plot_distribution_stacked(
    ds,
    domain_column: str,
    quality_column: str,
    num_words_column: str,
    weight: Literal["count", "num_words"] = "count",
    figsize=(14, 6),
    rotate_xticks=30,
    save_path: str | None = None,
):
    """Plot the distribution of the quality score by domain."""
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

    # proportions
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # sort by share of 4–5 (optional; comment out if you prefer natural order)
    order = props.loc[:, 4:].sum(axis=1).sort_values(ascending=False).index
    props = props.loc[order]

    # Plot
    x = np.arange(len(props))
    bottoms = np.zeros(len(props))
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    cmap = plt.cm.get_cmap("Set2")
    colors = [cmap(i) for i in range(6)]
    colors = colors[::-1]

    for score in range(6):
        ax.bar(
            x,
            props[score].values,
            bottom=bottoms,
            label=f"Score {score}",
            color=colors[score],
        )
        bottoms += props[score].values

    # Label
    ax.set_xticks(x)
    ax.set_xticklabels(props.index, rotation=rotate_xticks, ha="right")
    ax.set_xlabel("Health topics")
    ylabel = "Proportion of documents" if weight == "count" else "Proportion of words"
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(f"Edu quality score (0-5) by health topic ({weight})")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    ax.legend(
        handles,
        labels,
        title="Edu quality score",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.82)  # room for legend
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
    return props.round(4)


def main(dataset_path: str, num_workers: int = 4, max_samples: int | None = None):
    dataset = load_dataset(
        dataset_path,
        split="train",
        num_proc=num_workers,
    )
    print(dataset)

    if max_samples is not None:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    props = plot_distribution_stacked(
        dataset,
        domain_column="health_domain_classification_best_class",
        quality_column="edu_quality_normalized_score",
        num_words_column="num_words",
        weight="count",
        save_path="images/edu_quality_by_domain_count.png",
    )
    print(props.to_markdown())

    props = plot_distribution_stacked(
        dataset,
        domain_column="health_domain_classification_best_class",
        quality_column="edu_quality_normalized_score",
        num_words_column="num_words",
        weight="num_words",
        save_path="images/edu_quality_by_domain_num_words.png",
    )
    print(props.to_markdown())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
