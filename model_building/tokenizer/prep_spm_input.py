"""Prepare SPM input dataset."""

import os

# import re
import regex as re
from multiprocessing import Pool

import pyarrow.compute as pc
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

# from scripts.data.split_document import build_tokenizer, split_documents

# Script detectors
UNWANTED_SCRIPTS_RE = re.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}\p{Cyrillic}]")
# Allowed characters / symbols
ASCII_PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~"""
# EXTRA_SYMS = "…–—-•·°µ×±≤≥‰′″§«»“”‘’€£¥©®™"
WHITESPACE_OK = "\t\n\r\u00a0\u202f"  # tab, LF/CR, NBSP, NNBSP


# A quick per-char validator
def char_ok(ch: str) -> bool:
    """Check if a character is allowed."""
    # Always allow basic whitespace
    if ch in WHITESPACE_OK or ch == " ":
        return True
    # Latin or Greek letters (with diacritics)
    if re.match(r"\p{Latin}|\p{Greek}", ch):
        return True
    # Digits
    if re.match(r"\p{Nd}", ch):
        return True
    # ASCII punctuation
    if ch in ASCII_PUNCT:
        return True
    # Extra curated symbols
    # if ch in EXTRA_SYMS:
    #     return True
    return False


def quality_gate(
    text: str | None,
    max_unwanted_ratio: float = 0.01,  # ≤1% unwanted-script chars
    max_disallowed_ratio: float = 0.02,  # ≤2% disallowed chars overall
    # min_len: int = 20,
    # max_len: int = 10000,
) -> bool:
    """Quality gate for a text."""
    if text is None:
        return False

    # Normalize newlines; keep structure
    t = text.replace("\r\n", "\n")

    # # Length gate
    # if not (min_len <= len(t) <= max_len):
    #     return False

    # Quick kill-switch: lots of unwanted scripts?
    bad_script_hits = UNWANTED_SCRIPTS_RE.findall(t)
    if bad_script_hits and (len(bad_script_hits) / len(t) > max_unwanted_ratio):
        return False

    # Character-level allowlist
    disallowed = sum(0 if char_ok(c) else 1 for c in t)
    if disallowed / max(1, len(t)) > max_disallowed_ratio:
        return False

    # Discard lines that are almost all punctuation or URL noise
    # letters = len(re.findall(r"\p{L}", t))
    # if letters / max(1, len(t)) < 0.2 and "http" in t:
    #     return False

    return True


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


def get_dataset_info(ds: Dataset, num_words_column: str = "num_words"):
    """Print information about a dataset."""
    # total_num_words = pc.sum(dataset.data["num_words"]).as_py()
    # Filter is index-based: After filter, the dataset keeps an index mapping to the original table
    total_num_words = pc.sum(ds.with_format("arrow")[num_words_column]).as_py()
    return f"{len(ds):,} documents and {total_num_words:,} words"


def write_dataset_to_txt(ds: Dataset, output_path: str):
    """Write a dataset to a txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Writing"):
            f.write(row["text"] + "\n")


def filter_func(
    example,
    edu_quality_threshold: int | None = None,
    med_density_threshold: float | None = None,
    select_domains: list[str] | None = None,
):
    """Filter function for a dataset."""
    if example.get("num_words") is not None and example["num_words"] < 10:
        return False

    if example.get("health_domain_classification_best_class") in {"Others"}:
        return False

    if (
        edu_quality_threshold is not None
        and example.get("edu_quality_normalized_score") is not None
        and example["edu_quality_normalized_score"] < edu_quality_threshold
    ):
        return False

    if (
        med_density_threshold is not None
        and example.get("medical_entity_density") is not None
        and example["medical_entity_density"] < med_density_threshold
    ):
        return False

    if (
        select_domains is not None
        and example.get("health_domain_classification_best_class") is not None
        and example["health_domain_classification_best_class"] not in select_domains
    ):
        return False

    return True


def sample_dataset(ds: Dataset, num_samples: int, seed: int = 42):
    """Randomly sample a dataset."""
    if num_samples >= ds.num_rows:
        print(f"Warning: num_samples {num_samples} is greater than the number of rows in the dataset {ds.num_rows}")
        return ds
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(num_samples))
    return ds


def main(output_dir: str, dataset: str, num_workers: int = 8, max_samples: int | None = None):
    # Full 7-subtopic select list (kept for reference; current active list below)
    # select_domains = [
    #     "Biomedical & mechanistic science",
    #     "Clinical cases & vignettes",
    #     "Clinical guidelines & pathways",
    #     "Drugs, trials & regulation",
    #     "Health IT, telemedicine & operations",
    #     "Health workforce education & training",
    #     "Medical devices, diagnostics & imaging",
    # ]
    # Tight bio+clinical subset (matches `subtopic_biocli5` in convert_to_mds_parallel.py)
    select_domains = [
        "Biomedical & mechanistic science",
        "Clinical cases & vignettes",
        "Clinical guidelines & pathways",
        "Drugs, trials & regulation",
        "Medical devices, diagnostics & imaging",
    ]
    # other_domains = [
    #     # "Commercial & promotional",
    #     "Health services & facilities",
    #     "Occupational health & safety",
    #     "Other health",
    #     "Patient education & lifestyle",
    #     "Public health, policy & programs",
    #     "Wellness, supplements & CAM",
    # ]
    edu_quality_threshold = 4
    med_density_threshold = 0.1

    # Each entry: name -> {path, tasks: list of filter tasks}.
    # Per-task keys:
    #   select_domains             : passed to filter_func (per-task)
    #   downsample_ratio           : keep ratio * own_rows
    #   downsample_ratio_of_target : keep ratio * target_domain_data_size
    # Tasks without any downsample contribute to target_domain_data_size.
    # edu_quality_threshold and med_density_threshold are uniform across all datasets.
    # Comment out an entry / task to disable it.
    dataset_configs = {
        # "nachos": {
        #     "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified",
        #     "tasks": [{}],
        # },
        "fineweb-2": {
            # "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored",
            "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified",
            "tasks": [
                {},
                # {"select_domains": other_domains, "downsample_ratio": 0.5},  # tried 0.35, 0.24, 0.5
            ],
        },
        "finepdfs": {
            # "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored",
            "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified",
            "tasks": [
                {},
                # {"select_domains": other_domains, "downsample_ratio": 0.5},  # tried 0.35, 0.24, 0.5
            ],
        },
        "finewiki": {
            "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified",
            "tasks": [{}],
        },
        # "transcorpus_bio_fr": {
        #     "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified_extracted_gliner",
        #     "tasks": [{}],
        # },
        # "mmc": {
        #     "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/multilingual_medical_corpus/health_domain_classified_edu_quality_scored",
        #     "tasks": [{}],
        # },
        # "e3c": {
        #     "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/E3C/layer3/health_domain_classified_edu_quality_scored",
        #     "tasks": [{}],
        # },
        # "synthesized_v2": {
        #     "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/final/v2",
        #     "tasks": [{}],
        # },
        # "science": {
        #     "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_scored",
        #     "tasks": [
        #         {"downsample_ratio_of_target": 0.075},  # historical: edu>=3
        #     ],
        # },
        # "cs": {
        #     "path": "/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_scored",
        #     "tasks": [
        #         {"downsample_ratio_of_target": 0.075},  # historical: edu>=3
        #     ],
        # },
    }

    # Resolve `dataset`: either a known name (key in dataset_configs) or a raw path.
    if dataset in dataset_configs:
        name = dataset
        cfg = dataset_configs[dataset]
    else:
        name = os.path.basename(dataset.rstrip("/"))
        cfg = {"path": dataset, "tasks": [{}]}

    datasets = []
    target_domain_data_size = 0
    # (tag, filtered_ds, ratio) — deferred until target_domain_data_size is known
    pending_target_relative = []

    ds = load_local_dataset(cfg["path"])
    for i, task in enumerate(cfg["tasks"]):
        tag = name if len(cfg["tasks"]) == 1 else f"{name}[{i}]"
        filtered_ds = ds.filter(
            filter_func,
            num_proc=num_workers,
            fn_kwargs={
                "edu_quality_threshold": edu_quality_threshold,
                "med_density_threshold": med_density_threshold,
                "select_domains": task.get("select_domains", select_domains),
            },
        )
        print(f"{tag}: {get_dataset_info(filtered_ds)}")

        if "downsample_ratio_of_target" in task:
            pending_target_relative.append((tag, filtered_ds, task["downsample_ratio_of_target"]))
        elif "downsample_ratio" in task:
            size = int(filtered_ds.num_rows * task["downsample_ratio"])
            filtered_ds = sample_dataset(filtered_ds, size)
            print(f"{tag} after downsampling: {get_dataset_info(filtered_ds)}")
            datasets.append(filtered_ds)
        else:
            target_domain_data_size += filtered_ds.num_rows
            datasets.append(filtered_ds)

    for tag, ds, ratio in pending_target_relative:
        size = int(target_domain_data_size * ratio)
        ds = sample_dataset(ds, size)
        print(f"{tag} after downsampling (vs target): {get_dataset_info(ds)}")
        datasets.append(ds)

    # concatenate datasets
    for ds in datasets:
        ds = ds.remove_columns(list(set(ds.column_names) - {"text"}))
    dataset = concatenate_datasets(datasets)
    print(f"Concatenated dataset: {get_dataset_info(dataset)}")

    # random sampling
    if max_samples is not None:
        dataset = sample_dataset(dataset, max_samples)
        print(f"Sampled dataset: {get_dataset_info(dataset)}")

    # Leave the corpus as normal multi-line text to keep structure
    # split documents
    # dataset = dataset.map(
    #     split_documents,
    #     batched=True,
    #     batch_size=1,
    #     num_proc=num_workers,
    #     # writer_batch_size=1000,
    #     # remove_columns=dataset.column_names,
    #     fn_kwargs={"max_tokens": 4196, "tokenizer": build_tokenizer()},
    #     desc="Splitting documents",
    # )
    # print(f"Split dataset: {get_dataset_info(dataset)}")

    # normalize text
    # pattern_newline = re.compile(r"[\r\n]+")

    # def _normalize_text(s):
    #     return pattern_newline.sub(" ", s).strip()

    # dataset = dataset.map(
    #     lambda x: {"text": _normalize_text(x["text"])},
    #     num_proc=num_workers,
    #     desc="Replacing newlines",
    # )

    # quality gate
    # dataset = dataset.filter(lambda x: quality_gate(x["text"]), num_proc=num_workers)
    # print(f"Quality-filtered dataset: {get_dataset_info(dataset)}")

    # save to txt
    os.makedirs(output_dir, exist_ok=True)
    parts = []
    for shard_idx in range(num_workers):
        ds = dataset.shard(index=shard_idx, num_shards=num_workers)
        parts.append((ds, f"{output_dir}/shard_{shard_idx}.txt"))
    with Pool(processes=num_workers) as pool:
        pool.starmap(write_dataset_to_txt, parts)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
