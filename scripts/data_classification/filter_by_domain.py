"""Filter a dataset by domain classification output by nemo-domain-classifier."""

import math
import os
from collections import Counter

import pyarrow.compute as pc
from datasets import Dataset, load_dataset
from tqdm import tqdm


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    # return f"{root}_{idx:05d}{ext}"
    os.makedirs(os.path.dirname(root), exist_ok=True)
    return f"{root}/{idx:05d}{ext}"


def load_hf_dataset(input_path: str) -> Dataset:
    """Load Hugging Face dataset from local file."""
    if os.path.isfile(input_path):
        if input_path.endswith(".txt"):
            ds = load_dataset("text", data_files=input_path, split="train")
        elif input_path.endswith(".jsonl"):
            ds = load_dataset("json", data_files=input_path, split="train")
        elif input_path.endswith(".parquet"):
            ds = load_dataset("parquet", data_files=input_path, split="train")
        else:
            raise ValueError(f"Unsupported file extension: {input_path}")
    else:
        ds = load_dataset(input_path, split="train")
    return ds


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


def main(dataset_path: str, output_dir: str, target_labels: list[str] | str | None = None, num_workers: int = 32):
    dataset = load_hf_dataset(dataset_path)
    print(dataset)

    value_counts = Counter()

    column_name = "domain_classification_best_class"

    # def count_values(batch):
    #     value_counts.update(batch[column_name])
    #     return {}

    # dataset.map(
    #     count_values,
    #     batched=True,
    #     batch_size=1000,
    #     remove_columns=dataset.column_names,
    #     load_from_cache_file=False,
    # )

    # # print(value_counts.most_common())
    # print(value_counts)

    # HuggingFace Dataset -> Arrow Table -> ChunkedArray
    arr = dataset.data[column_name]
    vc_struct = pc.value_counts(arr)
    # Convert the small result to a normal Python dict
    values = vc_struct.field("values").to_pylist()
    counts = vc_struct.field("counts").to_pylist()
    value_counts = dict(zip(values, counts))
    print(value_counts)

    if target_labels is not None:
        if isinstance(target_labels, str):
            target_labels = [target_labels]

        for target_label in target_labels:
            sampled_dataset = dataset.filter(lambda x: x[column_name] == target_label, num_proc=num_workers)
            output_path = f"{output_dir}/{target_label.lower().replace(' ', '_')}.parquet"
            save_hf_dataset(sampled_dataset, output_path, output_shard_size=1_000_000)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
