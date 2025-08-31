"""
Postprocess LLM topic/quality annotations.

For topic annotations:
1. Normalize the topics to the canonical categories.
2. Filter out the topics that are not in the canonical categories.
3. Save the dataset to parquet shards.

For quality annotations:
1. Parse the quality annotation and filter out invalid scores.
2. Save the dataset to parquet shards.
"""

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Literal

from datasets import Dataset, load_dataset

CANONICAL = [
    "Clinical cases & vignettes",
    "Clinical guidelines & pathways",
    "Patient education & lifestyle",
    "Wellness, supplements & CAM",
    "Public health, policy & programs",
    "Commercial & promotional",
    "Drugs, trials & regulation",
    "Biomedical & mechanistic science",
    "Medical devices, diagnostics & imaging",
    "Health IT, telemedicine & operations",
    "Occupational health & safety",
    "Health workforce education & training",
    "Health services & facilities",
    "Other health",
    "Others",
]

# Lowercased lookup for exact canonical names
_CANON_LUT = {c.lower(): c for c in CANONICAL}

# Hand-tuned mappings for your off-list labels + common variants
_DIRECT_MAP = {
    # trials/research variants
    "clinical trials & regulation": "Drugs, trials & regulation",
    "clinical trials & research": "Drugs, trials & regulation",
    "clinical trials & pathways": "Drugs, trials & regulation",
    "clinical research & trials": "Drugs, trials & regulation",
    "clinical research & studies": "Drugs, trials & regulation",
    "clinical research": "Drugs, trials & regulation",
    # diagnostics/imaging variants
    "clinical diagnostics & imaging": "Medical devices, diagnostics & imaging",
    "clinical diagnostics & testing": "Medical devices, diagnostics & imaging",
    # telemedicine variants
    "telemedicine & operations": "Health IT, telemedicine & operations",
    "telemedicine & remote care": "Health IT, telemedicine & operations",
    "telemedicine & digital health": "Health IT, telemedicine & operations",
    "telemedicine & health it": "Health IT, telemedicine & operations",
    "telemedicine & health services": "Health IT, telemedicine & operations",
    "telemedicine & virtual care": "Health IT, telemedicine & operations",
    # services/facilities
    "clinical services & facilities": "Health services & facilities",
    # procedures
    "clinical procedures & interventions": "Clinical guidelines & pathways",
    "clinical procedures & techniques": "Clinical guidelines & pathways",
    # education-ish
    "clinical definitions & terminology": "Health workforce education & training",
    "clinical anatomy & morphology": "Health workforce education & training",
    # french terms
    "clauses de conscience": "Public health, policy & programs",
    "radioprotection": "Occupational health & safety",
    "addictions comportementales": "Other health",
    # null-ish
    # "none": "Others",
    # "": "Others",
}


def save_to_parquet(ds: Dataset, output_dir: str, output_shard_size: int | None = None):
    """Save a dataset to parquet shards."""
    if output_shard_size is not None:
        num_shards = math.ceil(len(ds) / output_shard_size)
        for shard_idx in range(num_shards):
            shard = ds.shard(index=shard_idx, num_shards=num_shards)
            shard.to_parquet(f"{output_dir}/{shard_idx:09d}.parquet")
    else:
        ds.to_parquet(f"{output_dir}/000000000.parquet")


def _normalize_text(s: str) -> str:
    """Lower, trim, collapse spaces, strip accents, normalize &/and."""
    s = s or ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" and ", " & ")
    return s


def normalize_topic(label: str | None) -> str | None:
    """Normalize a topic label to the canonical categories."""
    if label is None:
        return None

    norm = _normalize_text(label)
    if norm in _CANON_LUT:
        return _CANON_LUT[norm]

    # Direct mapping of known variants
    if norm in _DIRECT_MAP:
        return _DIRECT_MAP[norm]

    return label


def postprocess_topic(ds: Dataset, num_workers: int = 8) -> Dataset:
    """Parse the topic annotation and normalize it to the canonical categories."""

    def process_func(example):
        try:
            json_output = json.loads(example["output"])
        except json.JSONDecodeError:
            json_output = {}

        topic = json_output.get("topic")
        normalized_topic = normalize_topic(topic)

        return {"topic": normalized_topic}

    ds = ds.map(process_func, num_proc=num_workers)

    # filter out invalid topics
    ds = ds.filter(
        lambda x: x["topic"] in CANONICAL,
        num_proc=num_workers,
    )
    print(f"Filtered to {ds.num_rows:,d} examples")

    return ds


def postprocess_quality(ds: Dataset, num_workers: int = 8) -> Dataset:
    """Parse the quality annotation and filter out invalid scores."""

    def process_func(example):
        try:
            json_output = json.loads(example["output"])
        except json.JSONDecodeError:
            json_output = {}

        score = json_output.get("score")
        if score is not None:
            score = int(score)

        return {"score": score}

    ds = ds.map(process_func, num_proc=num_workers)

    # score counts
    score_counts = Counter(ds["score"])
    print(score_counts)

    # filter out invalid scores
    allowed_scores = set(range(6))
    ds = ds.filter(
        lambda x: x["score"] in allowed_scores,
        num_proc=num_workers,
    )
    print(f"Filtered to {ds.num_rows:,d} examples")

    return ds


def main(
    input_dir: str,
    output_dir: str,
    task: Literal["topic", "quality"],
    num_workers: int = 8,
    output_shard_size: int | None = None,
    test_split_size: int | None = None,
):
    """Main function."""
    ds = load_dataset(input_dir, split="train")
    print(f"Loaded {ds.num_rows:,d} examples")

    if task == "topic":
        ds = postprocess_topic(ds, num_workers)
    elif task == "quality":
        ds = postprocess_quality(ds, num_workers)
    else:
        raise ValueError(f"Invalid task: {task}")

    if test_split_size is not None:
        ds = ds.train_test_split(test_size=test_split_size)
        save_to_parquet(ds["train"], f"{output_dir}/train", output_shard_size)
        save_to_parquet(ds["test"], f"{output_dir}/test", output_shard_size)
    else:
        save_to_parquet(ds, output_dir, output_shard_size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
