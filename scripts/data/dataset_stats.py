"""Compute statistics for a dataset."""

import os

import numpy as np
from datasets import Dataset, load_dataset


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


def get_num_words(sample: dict) -> int:
    return {"num_words": len(sample["text"].split())}


def main(input_path: str, num_proc: int = 16, max_samples: int | None = None):
    ds = load_hf_dataset(input_path)
    print(ds)

    if max_samples is not None:
        ds = ds.select(range(max_samples))

    ds = ds.map(get_num_words, num_proc=num_proc)
    num_words = np.array(ds["num_words"])
    stat = {
        "sum": int(num_words.sum()),
        "mean": float(num_words.mean()),
        "median": float(np.median(num_words)),
        "std": float(num_words.std()),
        "min": int(num_words.min()),
        "max": int(num_words.max()),
    }
    # print(stat)

    # Print with comma formatting
    print("Dataset statistics:")
    print(f"Sum: {stat['sum']:,}")
    print(f"Mean: {stat['mean']:,.2f}")
    print(f"Median: {stat['median']:,.2f}")
    print(f"Std: {stat['std']:,.2f}")
    print(f"Min: {stat['min']:,}")
    print(f"w: {stat['max']:,}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
