"""Postprocess dataset saved in datatrove format."""

import json
import math
import os

from datasets import Dataset, Sequence, Value, load_dataset
from tqdm import tqdm

# Keep in sync with MEDICAL_ENTITIES in postprocess_extract.py (canonical 8 entity types).
# Used to parse `original_medical_entities` JSON strings back into uniform struct columns
# (postprocess_extract.py serializes to JSON to dodge datatrove's per-worker Parquet
# schema inference; we deserialize here for downstream consumers).
MEDICAL_ENTITIES: tuple[str, ...] = (
    "disease", "drug", "body_part", "medical_procedure",
    "molecular_marker", "clinical_device", "vital_function", "living_beings",
)


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:09d}{ext}"
    # os.makedirs(os.path.dirname(root), exist_ok=True)
    # return f"{root}/{idx:09d}{ext}"


def load_local_dataset(input_path: str) -> Dataset:
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


def save_hf_dataset(ds: Dataset, output_path: str, output_shard_size: int | None = None):
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
    input_path: str,
    output_path: str,
    num_workers: int = 8,
    shuffle: bool = False,
    seed: int = 42,
    output_shard_size: int | None = None,
    max_samples: int | None = None,
):
    dataset = load_local_dataset(input_path)
    print(f"Loaded {dataset.num_rows:,d} examples")

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
        print("Shuffled dataset")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, dataset.num_rows)))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    def process_example(example):
        new_example = {}
        for k, v in example["metadata"].items():
            # ignore dataset column
            if k in ["dataset"]:
                continue
            # Parse `original_medical_entities` JSON string back to struct dict (it was
            # JSON-serialized in postprocess_extract.py to dodge datatrove's per-worker
            # parquet schema inference). Falls back to `{ek: []}` for any
            # malformed/None/non-dict row to keep the resulting schema uniform.
            elif k == "original_medical_entities":
                try:
                    parsed = json.loads(v) if isinstance(v, str) and v else None
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if not isinstance(parsed, dict):
                    parsed = {}
                new_example[k] = {ek: (parsed.get(ek) or []) for ek in MEDICAL_ENTITIES}
            elif k in example:
                new_example[f"{k}_old"] = v
            else:
                new_example[k] = v
        # tmp: update num_words
        # new_example["num_words"] = len(example["text"].split())
        return new_example

    dataset = dataset.map(
        process_example,
        remove_columns=["metadata"],
        num_proc=num_workers,
        desc="Processing examples...",
    )
    print(f"Processed {dataset.num_rows:,d} examples")

    # Explicitly type the deserialized struct column so any per-worker shard that saw only
    # empty lists doesn't infer `list<null>`. Same fix as in postprocess_extract.py but
    # done via cast (HF Dataset.cast_column propagates through `to_parquet` cleanly,
    # unlike datatrove's per-worker writer where it doesn't).
    if "original_medical_entities" in dataset.column_names:
        dataset = dataset.cast_column(
            "original_medical_entities",
            {k: Sequence(Value("string")) for k in MEDICAL_ENTITIES},
        )

    save_hf_dataset(dataset, output_path, output_shard_size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
