"""
Convert a dataset to MDS format.

pip install mosaicml-streaming
"""

import hashlib
import inspect
import json
import multiprocessing as mp
import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, concatenate_datasets, load_dataset
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm
from transformers import AutoTokenizer


def load_local_dataset(input_path: str) -> Dataset:
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
        return load_dataset(input_path, split="train")
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


def get_dataset_info(ds: Dataset, count_columns: list[str] = ["num_words"]):
    """Print information about a dataset."""
    # Build a minimal Arrow table with the required columns
    ds_table = ds.with_format("arrow")[:].select(count_columns)

    # res = f"{len(ds):,} documents"
    res = {"num_documents": len(ds)}

    for col in count_columns:
        # Total numbers (fast C++ kernels)
        total_count = pc.sum(ds_table[col]).as_py()
        # res += f" and {total_count:,} {col}"
        res[col] = total_count

    return res


_BIOCLI5_SUBTOPICS = frozenset([
    "Biomedical & mechanistic science",
    "Clinical cases & vignettes",
    "Clinical guidelines & pathways",
    "Drugs, trials & regulation",
    "Medical devices, diagnostics & imaging",
])

# Substring-matched filter knobs for `_filter_func_p1_quality`. Each entry is
# (mode_substring, predicate(example) -> bool). Predicates return True when
# the row passes the knob; missing-column rows are treated as passing so the
# filter is a no-op on inputs that don't carry the relevant signal.
# To add a new knob: append one tuple here — no other changes needed.
_FILTER_KNOBS = [
    ("subtopic_biocli5", lambda x: x.get("health_domain_classification_best_class") in _BIOCLI5_SUBTOPICS),
    ("edu2",  lambda x: x.get("edu_quality_normalized_score") is None or x["edu_quality_normalized_score"] >= 2),
    ("edu4",  lambda x: x.get("edu_quality_normalized_score") is None or x["edu_quality_normalized_score"] >= 4),
    ("med02", lambda x: x.get("medical_entity_density") is None or x["medical_entity_density"] >= 0.2),
    ("med01", lambda x: x.get("medical_entity_density") is None or x["medical_entity_density"] >= 0.1),
]


def write_to_mds(dataset: Dataset, output_dir: str, size_limit_mb: int = 512):
    """Write a dataset to MDS (default size limit is 512MB).

    Wipes any pre-existing partial output: MDSWriter refuses to write into a
    non-empty directory, so a re-run that hits this function (i.e., the worker's
    marker-skip did NOT short-circuit, meaning the previous run never completed
    this parquet) would otherwise crash. Safe to wipe because reaching this
    point implies no completion marker → any on-disk files are partial.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    size_limit = int(size_limit_mb) * (1 << 20)
    os.makedirs(output_dir, exist_ok=True)
    with MDSWriter(out=output_dir, columns={"text": "str"}, size_limit=size_limit) as writer:
        for record in tqdm(dataset, total=len(dataset), desc="Writing to MDS", disable=True):  # not sys.stderr.isatty()):
            writer.write(record)


def worker_process_one_parquet(
    parquet_path: str,
    name: str,
    parquet_idx: int,
    output_dir: str,
    tokenizer_name_or_path: str | None,
    mode: str,
    test_size: int | float | None = None,
    seed: int = 42,
    p2_long_threshold: int = 2700,
    p2_shrink_alpha: float = 0.2,
    p2_short_floor: int = 256,
    p2_max_long_frac: float = 0.8,
):
    """Process a single parquet file and return stats (no shared file writes).

    `mode` is the filter knob string (e.g. "edu4_med01_subtopic_biocli5"); substring
    matching drives _filter_func_p1_quality and the dedup / rwkeep transforms.
    If `mode` contains "p2down", a length-aware downsample is applied after the
    quality filter (see the p2-down block below for details).
    `tokenizer_name_or_path` is only loaded if the `_process_func` map is enabled below;
    pass None to skip the load entirely.
    """

    # Skip-if-done: marker written at end of a successful worker run.
    # Marker only carries the small scalar stats (data_stats / filtered_stats).
    subdir_name = f"{parquet_idx:09d}_{name}"
    marker_dir = os.path.join(output_dir, "_done_markers")
    marker_json = os.path.join(marker_dir, f"{subdir_name}.json")
    if os.path.isfile(marker_json):
        with open(marker_json, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"SKIP {subdir_name}: marker present in {marker_dir}")
        return cached.get("data_stats"), cached.get("filtered_stats"), {}

    # Load tokenizer inside worker only when tokenization is enabled (currently disabled —
    # see commented `_process_func` map below). Skip the load when we don't need it.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path) if tokenizer_name_or_path else None

    def _process_func(examples):
        texts = examples["text"]
        # # tmp: truncate long texts
        # texts = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in tokenizer(texts, truncation=True, max_length=1024)["input_ids"]]
        # examples["text"] = texts
        examples["num_tokens"] = [len(token_ids) for token_ids in tokenizer(texts)["input_ids"]]
        if "num_words" not in examples:
            examples["num_words"] = [len(text.split()) for text in texts]
        return examples

    def _filter_func_p1_universal(example, num_word_column: str = "num_words"):
        """Phase-1 universal hygiene filter — applies to all data (raw AND rewritten)."""

        # Remove documents by word/token count
        if example.get(num_word_column) is not None and example[num_word_column] < 10:
            return False

        # Remove documents by health domain classification
        if "health_domain_classification_best_class" in example and example["health_domain_classification_best_class"] in [
            "Others",
            # "Commercial & promotional",
        ]:
            # print(f"Filtering out {example['health_domain_classification_best_class']}")
            return False

        return True

    # Split mode on the `_plus_rewritten` delimiter into per-side knob strings.
    # Examples:
    #   "raw_edu4_med01_subtopic_biocli5_plus_rewritten_subtopic_biocli5"
    #       → raw_mode="raw_edu4_med01_subtopic_biocli5", rw_mode="subtopic_biocli5"
    #   "raw_edu4_med01_plus_rewritten"
    #       → raw_mode="raw_edu4_med01",                 rw_mode=""  (rewritten passes through)
    #   "raw_edu4_med01_subtopic_biocli5"  (no rewritten in pipe)
    #       → raw_mode="raw_edu4_med01_subtopic_biocli5", rw_mode=""
    if "_plus_rewritten" in mode:
        _raw_mode, _rw_mode = mode.split("_plus_rewritten", 1)
        _rw_mode = _rw_mode.lstrip("_")
    else:
        _raw_mode, _rw_mode = mode, ""

    def _filter_func_p1_quality(example, side_mode: str):
        """Phase-1 quality filter — substring-matched edu / med / subtopic gates against `side_mode`."""

        is_union = "union" in side_mode
        # Identity element: True for AND (intersection), False for OR (union)
        passed = not is_union
        for key, predicate in _FILTER_KNOBS:
            if key in side_mode:
                check = predicate(example)
                passed = (passed or check) if is_union else (passed and check)

        return passed

    def _filter_func_p1(example):
        """Phase-1 orchestrator: universal hygiene + per-side quality gates.

        Picks the knob string based on the row-name tag: rewritten rows use the
        chunk after `_plus_rewritten`, raw rows use the chunk before. An empty
        side string applies no quality gates (universal hygiene still runs).
        """
        if not _filter_func_p1_universal(example):
            return False
        side_mode = _rw_mode if "rewritten" in name else _raw_mode
        if not side_mode:
            return True
        return _filter_func_p1_quality(example, side_mode)

    # Load dataset
    ds = load_local_dataset(parquet_path)

    # debug
    # ds = ds.select(range(100))

    # Keep only what we need before the heavy ops
    keep_cols = {"text", "num_words", "health_domain_classification_best_class", "edu_quality_normalized_score", "medical_entity_density"}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    # don't count tokens for now
    # def _stable_fingerprint() -> str:
    #     """Create stable fingerprint that auto-detects function changes."""
    #     func_source = inspect.getsource(_process_func)
    #     combined = ds._fingerprint + func_source + tokenizer_name_or_path
    #     return hashlib.md5(combined.encode()).hexdigest()

    # ds = ds.map(
    #     _process_func,
    #     batched=True,
    #     # num_proc=num_workers,
    #     new_fingerprint=_stable_fingerprint(),
    #     desc="Tokenizing dataset",
    # )

    # Collect stats before filtering
    data_stats = {"name": name, "parquet_idx": parquet_idx, **get_dataset_info(ds, count_columns=["num_words"])}

    ds = ds.filter(
        _filter_func_p1,
        # _filter_func_p1_quality,
        # fn_kwargs={"num_word_column": "num_word"},
        # num_proc=num_workers,
    )

    # p2 length-aware downsample (gated): drop very short chunks, then proportionally
    # shrink each parquet while keeping (most) long docs. Preserves source mixture
    # by shrinking each source uniformly. The max_long_frac cap prevents long-rich
    # sources (finepdfs, NACHOS) from collapsing to 100% long, which would erase
    # short-context training signal and cause continual-pretraining regressions.
    # No row duplication → no train/test leak risk.
    if "p2down" in mode:
        pre_n = len(ds)
        if "num_words" in ds.column_names:
            nw = ds["num_words"]
        else:
            nw = [len(t.split()) for t in ds["text"]]
        kept = [i for i, w in enumerate(nw) if w >= p2_short_floor]
        if len(kept) < pre_n:
            ds = ds.select(kept)
            nw = [nw[i] for i in kept]
        long_indices = [i for i, w in enumerate(nw) if w >= p2_long_threshold]
        short_indices = [i for i, w in enumerate(nw) if w < p2_long_threshold]
        target_n = round(len(ds) * p2_shrink_alpha)
        n_long_max = round(target_n * p2_max_long_frac)
        n_long = min(len(long_indices), n_long_max)
        n_short = max(0, target_n - n_long)
        rng = random.Random(seed + parquet_idx)
        # Random-sample longs when we have more than the cap (avoids the
        # systematic "drop the later parquet rows" bias from a slice).
        if len(long_indices) > n_long:
            sampled_long = rng.sample(long_indices, n_long)
        else:
            sampled_long = list(long_indices)
        if n_short >= len(short_indices):
            sampled_short = list(short_indices)
        else:
            sampled_short = rng.sample(short_indices, n_short)
        selected = sampled_long + sampled_short
        rng.shuffle(selected)
        ds = ds.select(selected)
        print(
            f"p2down {name}/{parquet_idx}: {pre_n:,} -> {len(ds):,} "
            f"(n_long={len(sampled_long):,}, n_short={len(sampled_short):,}, "
            f"floor={p2_short_floor}, L_target={p2_long_threshold}, "
            f"alpha={p2_shrink_alpha}, max_long_frac={p2_max_long_frac})"
        )

    # Collect stats after filtering (and p2 downsample if active).
    # Only basic per-parquet counts; distribution aggregations are out of scope
    # (kept buggy at scale; do offline from MDS shards if needed).
    filtered_stats = {"name": name, "parquet_idx": parquet_idx, **get_dataset_info(ds, count_columns=["num_words"])}
    stats_data: dict = {}  # legacy slot in worker return tuple, intentionally empty

    # Save to mds (subdir_name was computed at the top for the skip-marker check)
    if test_size is not None:
        # test_size = min(test_size, int(ds.num_rows * 0.05))  # fallback to 5%
        test_size = min(test_size, 0.05)  # fallback to 5%
        if test_size > 0:
            ds = ds.train_test_split(test_size=test_size, seed=seed)

            # Save train split to MDS
            print(f"Saving {subdir_name} train set to mds format...")
            train_output_path = os.path.join(output_dir, "train", subdir_name)
            write_to_mds(ds["train"], train_output_path)

            # Save test split to MDS
            print(f"Saving {subdir_name} validation set to mds format...")
            val_output_path = os.path.join(output_dir, "validation", subdir_name)
            write_to_mds(ds["test"], val_output_path)
        else:
            print(f"Saving {subdir_name} train set to mds format...")
            train_output_path = os.path.join(output_dir, "train", subdir_name)
            write_to_mds(ds, train_output_path)
    else:
        print(f"Saving {subdir_name} dataset to mds format...")
        output_path = os.path.join(output_dir, subdir_name)
        write_to_mds(ds, output_path)
    print(f"Finished processing parquet {subdir_name}")

    # Persist the done-marker so a re-run can skip this parquet.
    os.makedirs(marker_dir, exist_ok=True)
    with open(marker_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_stats": data_stats,
                "filtered_stats": filtered_stats,
            },
            f,
            ensure_ascii=False,
        )

    return data_stats, filtered_stats, stats_data


def main(
    output_dir: str,
    input_paths: list[str] | str,
    tokenizer_name_or_path: str | None = None,
    name: list[str] | str = "fineweb2",
    mode: str = "default",
    num_workers: int = 4,
    test_size: int | float | None = None,
    seed: int = 42,
    p2_long_threshold: int = 2700,
    p2_shrink_alpha: float = 0.2,
    p2_short_floor: int = 128,
    p2_max_long_frac: float = 0.8,
):
    """Convert one or more parquet dirs to a single MDS dataset.

    Args:
        input_paths: a single directory (containing `*.parquet`) or a comma-separated list
            of directories. Each directory's parquets are scanned and registered.
        name: row-name tag used by _filter_func_p1; tags containing "_rewritten"
            short-circuit the quality gates (universal hygiene still applies),
            otherwise _filter_func_p1_quality is applied. Can be a single string
            (broadcast to all input_paths) or a comma-separated list of tags
            parallel to input_paths (one tag per dir).
        mode: filter-knob string, substring-matched. Quality knobs go through
            _filter_func_p1_quality (subtopic_biocli5 / edu2 / edu4 / med01 / med02 / union).
            Stage gate: include "p2down" to apply per-parquet length-aware
            downsampling AFTER the quality filter (keeps all long docs, drops
            chunks below p2_short_floor, samples shorts to shrink to alpha × N).
            Default "default" → no filters, no p2 downsample.
        p2_long_threshold: word count at/above which a chunk counts as "long"
            for p2 downsampling. Default 2700 ≈ 4096 tokens.
        p2_shrink_alpha: per-parquet shrink factor for p2 (output_size ≈ alpha
            × post-filter size). Source mixture preserved by uniform shrink.
            Default 0.2 (aggressive — assumes p2 dataset is much smaller than p1).
        p2_short_floor: drop chunks with num_words below this before p2.
        p2_max_long_frac: ceiling on output long-fraction per parquet (default 0.8).
            Prevents long-rich sources (finepdfs, NACHOS) from collapsing to 100%
            long, which would erase short-context signal. Set to 1.0 to disable
            the cap (current behavior pre-knob).
    """
    if isinstance(input_paths, str):
        input_paths = [p.strip() for p in input_paths.split(",") if p.strip()]
    if isinstance(name, str):
        names = [n.strip() for n in name.split(",") if n.strip()]
    else:
        names = list(name)
    if len(names) == 1:
        names = names * len(input_paths)
    if len(names) != len(input_paths):
        raise ValueError(
            f"name must be a single tag or a list of length == len(input_paths); "
            f"got {len(names)} name(s) for {len(input_paths)} input_path(s)"
        )

    parquet_paths = []
    for tag, input_path in zip(names, input_paths):
        parquet_paths.extend([(tag, str(p)) for p in Path(input_path).glob("*.parquet")])
    print(f"Found {len(parquet_paths)} parquet files across {len(input_paths)} input dir(s)")

    os.makedirs(output_dir, exist_ok=True)

    ctx = mp.get_context("spawn")  # avoid fork+Arrow issues
    args = [
        (
            parquet_path,
            name,
            idx,
            output_dir,
            tokenizer_name_or_path,
            mode,
            test_size,
            seed,
            p2_long_threshold,
            p2_shrink_alpha,
            p2_short_floor,
            p2_max_long_frac,
        )
        for idx, (name, parquet_path) in enumerate(parquet_paths)
    ]
    with ctx.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        results = pool.starmap(worker_process_one_parquet, args, chunksize=1)

    # Merge mosaic index files. Remove any stale top-level index.json from a
    # previous failed run — streaming.base.util.merge_index does shutil.move(..)
    # into that path and errors with "Destination path already exists" otherwise.
    def _remove_stale_index(dir_path: str) -> None:
        idx = os.path.join(dir_path, "index.json")
        if os.path.isfile(idx):
            os.remove(idx)

    if test_size is not None:
        _remove_stale_index(os.path.join(output_dir, "train"))
        _remove_stale_index(os.path.join(output_dir, "validation"))
        merge_index(os.path.join(output_dir, "train"), keep_local=True)
        merge_index(os.path.join(output_dir, "validation"), keep_local=True)
    else:
        _remove_stale_index(output_dir)
        merge_index(output_dir, keep_local=True)

    # Write data summary files
    data_summary_file = os.path.join(output_dir, "data_summary.jsonl")
    filtered_data_summary_file = os.path.join(output_dir, "data_summary_filtered.jsonl")

    # Defensive None filter — older markers (from runs where stats were disabled)
    # carry `data_stats: null` / `filtered_stats: null`. Writing those as "null"
    # lines breaks pandas.read_json downstream; skip them.
    n_skipped_stats = 0
    with open(data_summary_file, "w", encoding="utf-8") as f_data, \
         open(filtered_data_summary_file, "w", encoding="utf-8") as f_filtered:
        for data_stats, filtered_stats, _ in results:
            if data_stats is None or filtered_stats is None:
                n_skipped_stats += 1
                continue
            f_data.write(json.dumps(data_stats, ensure_ascii=False) + "\n")
            f_filtered.write(json.dumps(filtered_stats, ensure_ascii=False) + "\n")
    if n_skipped_stats:
        print(f"data_summary*.jsonl: skipped {n_skipped_stats} parquet(s) with null stats (legacy/disabled-stats markers).")

    # Append group-by-name lines then a total line to each summary file
    def _append_group_then_total(file_path: str) -> None:
        if not os.path.isfile(file_path):
            return

        df = pd.read_json(file_path, lines=True)
        if df.empty:
            return

        grouped = df.groupby("name").sum().reset_index()

        with open(file_path, "a", encoding="utf-8") as f:
            # Append per-name grouped rows first
            for _, row in grouped.iterrows():
                rec = {
                    "name": row["name"],
                    "parquet_idx": -1,
                }
                for col in ["num_documents", "num_words", "num_tokens"]:
                    if col in grouped.columns and pd.notna(row[col]):
                        rec[col] = int(row[col])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Then append TOTAL row
            total_rec = {
                "name": "total",
                "parquet_idx": -1,
            }
            for col in ["num_documents", "num_words", "num_tokens"]:
                if col in grouped.columns:
                    total_rec[col] = int(grouped[col].sum())
            f.write(json.dumps(total_rec, ensure_ascii=False) + "\n")

    _append_group_then_total(data_summary_file)
    _append_group_then_total(filtered_data_summary_file)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
