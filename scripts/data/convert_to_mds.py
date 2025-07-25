"""Convert a dataset to MDS format."""
# pip install mosaicml-streaming

import os

from datasets import Dataset, load_dataset
from streaming import MDSWriter


def load_hf_dataset(input_path: str) -> Dataset:
    """Load Hugging Face dataset from local file."""
    if os.path.isfile(input_path):
        if input_path.endswith(".txt"):
            ds = load_dataset("text", data_files={"train": input_path}, split="train")
        elif input_path.endswith(".jsonl"):
            ds = load_dataset("json", data_files={"train": input_path}, split="train")
        elif input_path.endswith(".parquet"):
            ds = load_dataset("parquet", data_files={"train": input_path}, split="train")
        else:
            raise ValueError(f"Unsupported file extension: {input_path}")
    else:
        ds = load_dataset(input_path, split="train")
    return ds


def convert_to_mds(dataset, output_path):
    with MDSWriter(out=output_path, columns={"text": "str"}) as writer:
        for record in dataset:
            writer.write(record)


def main(input_path: str, output_dir: str, test_size: float | None = None, seed: int = 42):
    # Load the dataset
    dataset = load_hf_dataset(input_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create train/test split
    if test_size is not None:
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)

        # Convert train split to MDS
        train_output_path = os.path.join(output_dir, "train")
        print(f"Converting train set ({len(dataset['train']):,} samples) to MDS format...")
        convert_to_mds(dataset["train"], train_output_path)

        # Convert test split to MDS
        val_output_path = os.path.join(output_dir, "validation")
        print(f"Converting val set ({len(dataset['test']):,} samples) to MDS format...")
        convert_to_mds(dataset["test"], val_output_path)
    else:
        print(f"Converting dataset ({len(dataset):,} samples) to MDS format...")
        convert_to_mds(dataset, output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
