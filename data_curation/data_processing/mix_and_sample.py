"""
Load multiple datasets, shuffle and sample them, and save the result as a new dataset.
"""

import glob
import math
import os
import random

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:09d}{ext}"
    # os.makedirs(os.path.dirname(root), exist_ok=True)
    # return f"{root}/{idx:09d}{ext}"


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
        parquet_files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        if parquet_files:
            # Treat directory of parquet files as a parquet dataset
            return load_dataset("parquet", data_files=parquet_files, split="train")
        dataset_info_path = os.path.join(input_path, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            # Load dataset saved with `Dataset.save_to_disk`
            return load_from_disk(input_path)
        raise ValueError(f"Directory has no parquet files or HF dataset metadata: {input_path}")
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


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


def main(
    input_paths: list[str],
    output_path: str,
    max_samples: int | None = None,
    output_shard_size: int | None = None,
):
    rng = random.Random(42)

    # Fire may pass a space- or comma-separated string; normalize to list
    if isinstance(input_paths, str):
        input_paths = input_paths.split(",") if "," in input_paths else input_paths.split()
    else:
        input_paths = list(input_paths)

    max_samples_per_dataset = None
    if max_samples is not None:
        # tmp: some subsets has less than max_samples, so we multiply by 2 to ensure we have enough samples
        max_samples_per_dataset = max_samples // len(input_paths) * 2
        print(f"Max samples per dataset: {max_samples_per_dataset}")

    # Load datasets
    datasets = []
    for input_path in input_paths:
        dataset = load_local_dataset(input_path)
        # print(dataset)
        if max_samples_per_dataset is not None and max_samples_per_dataset < len(dataset):
            # Sample indices directly to avoid shuffling the whole dataset
            random_indices = rng.sample(range(len(dataset)), max_samples_per_dataset)
            dataset = dataset.select(random_indices)

        dataset = dataset.remove_columns(
            list(
                set(dataset.column_names)
                - {
                    "text",
                    "num_words",
                    "health_domain_classification_scores",
                    "health_domain_classification_best_class",
                    "health_domain_classification_best_score",
                    "edu_quality_score",
                    "edu_quality_normalized_score",
                }
            )
        )
        dataset = dataset.add_column("subset", [input_path] * len(dataset))
        datasets.append(dataset)

    # concatenate datasets
    processed_dataset = concatenate_datasets(datasets)
    print(f"Processed dataset: {processed_dataset.num_rows:,d} examples")
    
    # random sample
    if max_samples is not None and max_samples < len(processed_dataset):
        random_indices = rng.sample(range(len(processed_dataset)), max_samples)
        processed_dataset = processed_dataset.select(random_indices)
    print(f"Filtered dataset: {processed_dataset.num_rows:,d} examples")

    # Save results
    save_hf_dataset(processed_dataset, output_path, output_shard_size=output_shard_size)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
