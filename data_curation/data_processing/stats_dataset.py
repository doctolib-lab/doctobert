"""Compute statistics for a dataset."""

import os
import math
from tqdm import tqdm

import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, concatenate_datasets, load_dataset


def load_local_dataset(input_path: str, num_proc: int = 1) -> Dataset:
    """Load Hugging Face dataset from local file."""
    if input_path.endswith(".txt"):
        return load_dataset("text", data_files=input_path, split="train")
    elif input_path.endswith(".jsonl") or input_path.endswith(".json"):
        return load_dataset("json", data_files=input_path, split="train")
    elif input_path.endswith(".csv") or input_path.endswith(".csv.gz"):
        return load_dataset("csv", data_files=input_path, split="train")
    elif input_path.endswith(".parquet"):
        return load_dataset("parquet", data_files=input_path, split="train")
    elif os.path.isdir(input_path):
        return load_dataset(input_path, split="train", num_proc=num_proc)
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    # return f"{root}_{idx:09d}{ext}"
    os.makedirs(os.path.dirname(root), exist_ok=True)
    return f"{root}/{idx:09d}{ext}"


def save_hf_dataset(ds: Dataset, output_path: str, output_shard_size: int | None = None):
    """Save Hugging Face dataset to local file."""
    if output_path.endswith(".txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="Writing"):
                f.write(row["text"] + "\n")
    elif output_path.endswith(".jsonl"):
        ds.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif output_path.endswith(".parquet"):
        if output_shard_size is not None:
            num_shards = math.ceil(len(ds) / output_shard_size)
            for shard_idx in range(num_shards):
                shard = ds.shard(index=shard_idx, num_shards=num_shards)
                shard.to_parquet(_make_shard_path(output_path, shard_idx))
        else:
            ds.to_parquet(output_path)
    else:
        ds.save_to_disk(output_path)


def get_num_words(ds_table: pa.Table, num_words_column: str = "num_words"):
    """Return brief information and descriptive stats using Arrow aggregations."""

    # Pull the column as an Arrow array without materializing to NumPy
    arrow_column = ds_table[num_words_column]

    total_words = pc.sum(arrow_column).as_py()
    mean_words = pc.mean(arrow_column).as_py()
    quantiles = pc.quantile(arrow_column, q=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_pylist()
    std_words = pc.stddev(arrow_column).as_py()
    min_words = pc.min(arrow_column).as_py()
    max_words = pc.max(arrow_column).as_py()

    print("Words statistics:")
    print(f"#docs: {ds_table.num_rows:,}")
    print(f"Sum: {total_words:,}")
    print(f"Mean: {mean_words:,.2f}")
    print(f"Std: {std_words:,.2f}")
    print(f"Min: {min_words:,}")
    print(f"P25: {quantiles[0]:,.2f}")
    print(f"Median: {quantiles[1]:,.2f}")
    print(f"P75: {quantiles[2]:,.2f}")
    print(f"P90: {quantiles[3]:,.2f}")
    print(f"P95: {quantiles[4]:,.2f}")
    print(f"P99: {quantiles[5]:,.2f}")
    print(f"Max: {max_words:,}")


def summarize_numeric(ds_table: pa.Table, column: str) -> None:
    """Descriptive stats for a numeric column (mean, std, percentiles)."""
    if column not in ds_table.column_names:
        return
    arr = ds_table[column].drop_null()
    if len(arr) == 0:
        print(f"\n{column}: column present but all null — skipping.")
        return
    mean = pc.mean(arr).as_py()
    std = pc.stddev(arr).as_py()
    quantiles = pc.quantile(arr, q=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_pylist()
    mn = pc.min(arr).as_py()
    mx = pc.max(arr).as_py()
    print(f"\n{column} statistics (n={len(arr):,}):")
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Min: {mn:.4f}")
    print(f"P25: {quantiles[0]:.4f}")
    print(f"Median: {quantiles[1]:.4f}")
    print(f"P75: {quantiles[2]:.4f}")
    print(f"P90: {quantiles[3]:.4f}")
    print(f"P95: {quantiles[4]:.4f}")
    print(f"P99: {quantiles[5]:.4f}")
    print(f"Max: {mx:.4f}")


def get_num_words_by_class(
    ds_table: pa.Table,
    class_column: str,
    numeric_columns: list[str] | None = None,
) -> None:
    """Per-class doc count, word sum, share, and optional mean/median of numeric columns."""

    numeric_columns = [c for c in (numeric_columns or []) if c in ds_table.column_names and c != class_column]

    aggs: list[tuple[str, str]] = [("num_words", "count"), ("num_words", "sum")]
    for c in numeric_columns:
        aggs.append((c, "mean"))
        aggs.append((c, "approximate_median"))

    aggregated = ds_table.group_by(class_column).aggregate(aggs)

    classes = aggregated[class_column].to_pylist()
    doc_counts = aggregated["num_words_count"].to_pylist()
    word_sums = aggregated["num_words_sum"].to_pylist()
    total_docs = sum(doc_counts)
    total_words = sum(word_sums)

    rows = []
    for i, cls in enumerate(classes):
        row = {
            "class": cls,
            "n_docs": doc_counts[i],
            "doc_pct": 100 * doc_counts[i] / total_docs if total_docs else 0,
            "n_words": word_sums[i],
            "word_pct": 100 * word_sums[i] / total_words if total_words else 0,
        }
        for c in numeric_columns:
            row[f"{c}__mean"] = aggregated[f"{c}_mean"][i].as_py()
            row[f"{c}__median"] = aggregated[f"{c}_approximate_median"][i].as_py()
        rows.append(row)

    rows.sort(key=lambda r: r["n_docs"], reverse=True)

    print(f"\n\nDistribution by {class_column}:")
    print("=" * 100)
    header = f"{'class':<45} {'n_docs':>14} {'doc_%':>7}   {'n_words':>16} {'word_%':>7}"
    for c in numeric_columns:
        header += f"   {c + '_mean':>28} {c + '_median':>28}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{str(r['class']):<45} {r['n_docs']:>14,} {r['doc_pct']:>6.2f}%   "
            f"{r['n_words']:>16,} {r['word_pct']:>6.2f}%"
        )
        for c in numeric_columns:
            m = r[f"{c}__mean"]
            md = r[f"{c}__median"]
            m_str = "n/a" if m is None else f"{m:.4f}"
            md_str = "n/a" if md is None else f"{md:.4f}"
            line += f"   {m_str:>28} {md_str:>28}"
        print(line)


def main(
    input_path: str,
    text_column_name: str = "text",
    num_workers: int = 16,
    max_samples: int | None = None,
    apply_med01_edu4_filter: bool = False,
):
    """Compute statistics on a single corpus or a concatenation of corpora.

    `input_path` may be one path or a comma-separated list of paths. When
    multiple paths are passed, datasets are concatenated (on the *intersection*
    of column names — so per-corpus extras don't trip Arrow's schema check)
    before stats are computed. Useful for family aggregates like
    "FineMed = fw2 + finepdfs + finewiki".

    `apply_med01_edu4_filter` applies the canonical Stage-1 filter in-memory:
    `medical_entity_density >= 0.1` AND `edu_quality_normalized_score >= 4`.
    Run it on the pre-8k-split source corpora (`…_extracted_gliner/`) so the
    filtered stats are doc-level — matching the population that downstream
    rewriting eventually expanded into the rephrased corpus.
    """
    paths = [p.strip() for p in input_path.split(",") if p.strip()]

    if len(paths) == 1:
        ds = load_local_dataset(paths[0])
    else:
        parts: list[Dataset] = []
        for p in paths:
            sub = load_local_dataset(p)
            print(f"  + {p}: {sub.num_rows:,d} documents")
            parts.append(sub)
        # Concat on the intersection of column names — drops per-corpus extras
        # like `rewriting_config` that only the rephrased shards carry.
        common_cols = set(parts[0].column_names)
        for part in parts[1:]:
            common_cols &= set(part.column_names)
        parts = [
            part.remove_columns([c for c in part.column_names if c not in common_cols])
            for part in parts
        ]
        ds = concatenate_datasets(parts)

    print(f"Loaded {ds.num_rows:,d} documents")

    if max_samples is not None:
        ds = ds.select(range(max_samples))

    if apply_med01_edu4_filter:
        ds = ds.filter(
            lambda m, e: m >= 0.1 and e >= 4,
            input_columns=["medical_entity_density", "edu_quality_normalized_score"],
            num_proc=num_workers,
            desc="Filtering med01 & edu4",
        )
        print(f"After med01 & edu4 filter: {ds.num_rows:,d} documents")

    class_columns = [
        # "domain_classification_best_class",
        "health_domain_classification_best_class",
        "edu_quality_normalized_score",
    ]
    numeric_columns = [
        "edu_quality_normalized_score",
        "medical_entity_density",
    ]
    keep_columns = list(set(class_columns) | set(numeric_columns))

    if "num_words" not in ds.column_names:
        ds = ds.map(
            lambda x: {"num_words": len(x[text_column_name].split())},
            num_proc=num_workers,
            remove_columns=list(set(ds.column_names) - set(keep_columns)),
            desc="Counting words",
        )

    selected_columns = list((set(keep_columns) | {"num_words"}) & set(ds.column_names))
    # Build a minimal Arrow table with the required columns
    ds_table = ds.with_format("arrow")[:].select(selected_columns)

    get_num_words(ds_table)

    for col in numeric_columns:
        summarize_numeric(ds_table, col)

    available_numeric = [c for c in numeric_columns if c in ds_table.column_names]
    for class_column in class_columns:
        if class_column not in ds_table.column_names:
            continue
        per_class_numeric = [c for c in available_numeric if c != class_column]
        get_num_words_by_class(ds_table, class_column, numeric_columns=per_class_numeric)

    # tmp: save results
    # output_path = "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr.parquet"
    # output_shard_size = 1_000_000
    # save_hf_dataset(ds, output_path, output_shard_size=output_shard_size)
    # print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import fire

    fire.Fire(main)
