"""Evaluate tokenizer fertility on a dataset."""

import json
import os
import re
from typing import Any, Callable

import pandas as pd
import pyarrow.compute as pc
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

# CLS_TOKEN = "<cls>"
# SEP_TOKEN = "<sep>"
# EOS_TOKEN = "<eos>"


def get_text_builder(data_path: str) -> Callable:
    def _normalize_text(s: str) -> str:
        # Remove space before a period
        s = re.sub(r"\s+\.", ".", s)
        # Remove spaces around apostrophes
        s = re.sub(r"\s*'\s*", "'", s)
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _ner_tokens(e: dict[str, Any]) -> list[str]:
        return _normalize_text(" ".join(e["tokens"]))

    def _two_segment_with_specials(left: str, right: str) -> str:
        # Matches DEFT2020 task_1 and CLISTER formatting (double space after SEP)
        # return f"{CLS_TOKEN} {left} {SEP_TOKEN}  {right} {EOS_TOKEN}"
        return f"{left}   {right}"

    def _question_with_choices(question: str, choices: list[str], enumerate_prefixes: list[str] | None = None) -> str:
        if enumerate_prefixes:
            choices = [f"{p} {c}" for p, c in zip(enumerate_prefixes, choices)]
        # concatenated = f" {SEP_TOKEN} ".join(choices)
        # return f"{CLS_TOKEN} {question} {SEP_TOKEN} {concatenated} {EOS_TOKEN}"
        concatenated = "  ".join(choices)
        return f" {question}  {concatenated} "

    def _join_tokens(e: dict[str, Any]) -> str:
        return " ".join(e["tokens"]) if isinstance(e.get("tokens"), list) else str(e.get("tokens", ""))

    mapping = {
        # DEFT2020
        "deft2020/data/local_hf_task_1": lambda e: _two_segment_with_specials(e["source"], e["cible"]),
        "deft2020/data/local_hf_task_2": lambda e: _question_with_choices(
            e["source"], [e["cible_1"], e["cible_2"], e["cible_3"]], enumerate_prefixes=["(1)", "(2)", "(3)"]
        ),
        # FrenchMedMCQA (task_1 and task_2 use the same text template)
        "frenchmedmcqa/data/local_hf_None": lambda e: _question_with_choices(
            e["question"], [e[f"answer_{letter}"] for letter in ["a", "b", "c", "d", "e"]]
        ),
        # CLISTER (two segments)
        "clister/data/local_hf_None": lambda e: _two_segment_with_specials(e["text_1"], e["text_2"]),
        # CAS, PxCorpus, ESSAI (join tokens)
        "pxcorpus/data/local_hf_None": lambda e: _join_tokens(e),
        "cas/data/local_hf_pos": lambda e: _join_tokens(e),
        "essai/data/local_hf_pos": lambda e: _join_tokens(e),
        # DEFT2021
        "deft2021/data/local_hf_ner": lambda e: _ner_tokens(e),
        "deft2021/data/local_hf_cls": lambda e: e["text"],
        # DiaMED (lowercased clinical_case; no stopwords filtering)
        "diamed/data/local_hf_None": lambda e: e["clinical_case"].lower(),
        # MANTRAGSC (NER)
        "mantragsc/data/local_hf_fr_medline": lambda e: _ner_tokens(e),
        "mantragsc/data/local_hf_fr_patents": lambda e: _ner_tokens(e),
        "mantragsc/data/local_hf_fr_emea": lambda e: _ner_tokens(e),
        # MORFITT (CLS)
        "morfitt/data/local_hf_source": lambda e: e["abstract"],
        # E3C (NER)
        "e3c/data/local_hf_French_clinical": lambda e: _ner_tokens(e),
        "e3c/data/local_hf_French_temporal": lambda e: _ner_tokens(e),
        # QUAERO (NER)
        "quaero/data/local_hf_medline": lambda e: _ner_tokens(e),
        "quaero/data/local_hf_emea": lambda e: _ner_tokens(e),
    }

    for key, builder in mapping.items():
        if key in data_path:
            return builder
    raise ValueError(f"No text builder found for data_path: {data_path}")


def build_text(data_path: str, example: dict[str, Any]) -> str:
    return get_text_builder(data_path)(example)


def load_hf_dataset(
    dataset_name: str,
    dataset_config_name: str | None = None,
    split: str = "train",
) -> Dataset:
    dataset_args = {
        "path": dataset_name,
        "split": split,
        "trust_remote_code": True,
    }
    if dataset_config_name:
        dataset_args["name"] = dataset_config_name
    return load_dataset(**dataset_args)


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


def eval_tokenizer(
    tokenizer_name_or_path: str,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    dataset_config_name: str | None = None,
    split: str = "train",
    # batch_size: int = 1000,
    num_workers: int = 4,
    stat_long_examples: bool = False,
    stat_detailed_tokens: bool = False,
    output_counts_file: str | None = None,
):
    # Load dataset
    if dataset_path is not None:
        # dataset = load_local_dataset(dataset_path)
        dataset = load_from_disk(f"{dataset_path}/{split}")
    elif dataset_name is not None:
        dataset = load_hf_dataset(dataset_name, dataset_config_name, split)
    else:
        raise ValueError("Either dataset_path or dataset_name must be provided.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def tokenize_function(example):
        # text = example["text"]
        text = build_text(dataset_path, example)

        # Tokenize the batch of texts (no special tokens so we only count the real pieces).
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        result = {
            # "tokens": input_ids,
            # "char_count": [len(t) for t in texts],
            "token_count": len(input_ids),
            "word_count": len(text.split()),
        }
        if stat_detailed_tokens:
            result["tokens"] = input_ids
        return result

    tokenized_dataset = dataset.map(
        tokenize_function,
        # batched=True,
        # batch_size=batch_size,
        num_proc=num_workers,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset...",
    )

    print("Collecting token statistics...")
    # Convert the tokenized dataset to an Arrow table once (zero-copy / memory-mapped)
    arrow_table = tokenized_dataset.with_format("arrow")[:]

    # Total numbers (fast C++ kernels)
    total_tokens = pc.sum(arrow_table["token_count"]).as_py()
    total_words = pc.sum(arrow_table["word_count"]).as_py()
    # Calculate compression ratio (fertility)
    fertility = total_tokens / total_words if total_words > 0 else 0

    dataset_stats = {
        "total_examples": len(dataset),
        "total_tokens": total_tokens,
        "total_words": total_words,
        "avg_tokens_per_example": total_tokens / len(dataset),
        "avg_words_per_example": total_words / len(dataset),
        "fertility": fertility,
    }
    print("Dataset stats:", json.dumps(dataset_stats, indent=4, ensure_ascii=False))

    long_examples_stats = None
    if stat_long_examples:
        sampled_tokenized_dataset = tokenized_dataset.filter(
            lambda x: x["token_count"] > tokenizer.model_max_length,
            num_proc=num_workers,
            desc="Checking long examples...",
        )
        print(f"Found {len(sampled_tokenized_dataset):,} long examples.")

        # Compute how many tokens exceed the model's maximum length across the dataset
        examples_exceeding_max_length = len(sampled_tokenized_dataset)
        if examples_exceeding_max_length > 0:
            long_table = sampled_tokenized_dataset.with_format("arrow")[:]
            # For each long example, calculate (token_count - model_max_length) and sum them.
            excess_array = pc.subtract(long_table["token_count"], tokenizer.model_max_length)
            tokens_exceeding_max_length = pc.sum(excess_array).as_py()
        else:
            tokens_exceeding_max_length = 0

        long_examples_stats = {
            "examples_exceeding_max_length": examples_exceeding_max_length,
            "tokens_exceeding_max_length": tokens_exceeding_max_length,
        }
        print("Long examples stats:", json.dumps(long_examples_stats, indent=4, ensure_ascii=False))

    vocabulary_stats = None
    if stat_detailed_tokens:
        print("Collecting detailed token statistics...")
        # Flatten list<item: int32> column → one big 1-D array of token IDs
        flat_tokens = pc.list_flatten(arrow_table["tokens"])

        # Count occurrences of each token ID (returns struct of values & counts)
        counts_struct = pc.value_counts(flat_tokens)
        token_ids = counts_struct.field("values").to_pylist()
        token_freqs = counts_struct.field("counts").to_pylist()
        # Build list of (token_id, count) tuples and sort descending by count
        token_counts_list = list(zip(token_ids, token_freqs))
        token_counts_list.sort(key=lambda x: x[1], reverse=True)
        # For compatibility with previous JSON output
        token_counts_dict = {tid: cnt for tid, cnt in token_counts_list}

        # Calculate fertility metrics
        vocab_size = len(tokenizer.vocab)
        unique_tokens_used = len(token_counts_list)
        vocab_coverage = unique_tokens_used / vocab_size

        # Token frequency analysis
        most_common_tokens = token_counts_list[:20]
        least_common_tokens = token_counts_list[-20:]

        # Convert token IDs to actual tokens for readability
        def token_id_to_str(token_id):
            try:
                return tokenizer.decode([token_id])
            except Exception:
                return f"<UNK:{token_id}>"

        most_common_readable = [(token_id_to_str(token_id), count) for token_id, count in most_common_tokens]
        least_common_readable = [(token_id_to_str(token_id), count) for token_id, count in least_common_tokens]

        vocabulary_stats = {
            "vocab_size": vocab_size,
            "vocab_coverage": vocab_coverage,
            "num_unique_tokens_used": unique_tokens_used,
            "num_unused_tokens": vocab_size - unique_tokens_used,
            "most_common": most_common_readable,
            "least_common": least_common_readable,
        }
        print("Vocabulary stats:", json.dumps(vocabulary_stats, indent=4, ensure_ascii=False))

        if output_counts_file is not None:
            # Save token frequency dictionary as JSON
            with open(output_counts_file, "w", encoding="utf-8") as f:
                json.dump(token_counts_dict, f, indent=4, ensure_ascii=False)
            print(f"Token counts saved to: {output_counts_file}")

    return dataset_stats, long_examples_stats, vocabulary_stats


def main(
    tokenizer_name_or_path: str,
    output_file: str | None = None,
    # batch_size: int = 1000,
    num_workers: int = 4,
):
    # tasks = [
    #     # CAS & ESSAI (we download the POS task used in the benchmark)
    #     {"path": "DrBenchmark/CAS", "name": "pos"},
    #     {"path": "DrBenchmark/ESSAI", "name": "pos"},
    #     # QUAERO (two NER sources)
    #     {"path": "DrBenchmark/QUAERO", "name": "emea"},
    #     {"path": "DrBenchmark/QUAERO", "name": "medline"},
    #     # E3C (French Clinical + Temporal)
    #     {"path": "DrBenchmark/E3C", "name": "French_clinical"},
    #     {"path": "DrBenchmark/E3C", "name": "French_temporal"},
    #     # MorFITT (multi-label CLS)
    #     {"path": "DrBenchmark/MORFITT"},
    #     # FrenchMedMCQA (MCQA; CLS use is derived from the same dataset)
    #     {"path": "DrBenchmark/FrenchMedMCQA"},
    #     # Mantra-GSC (French: EMEA / Medline / Patents)
    #     {"path": "DrBenchmark/MANTRAGSC", "name": "fr_emea"},
    #     {"path": "DrBenchmark/MANTRAGSC", "name": "fr_medline"},
    #     {"path": "DrBenchmark/MANTRAGSC", "name": "fr_patents"},
    #     # CLISTER (STS)
    #     {"path": "DrBenchmark/CLISTER"},
    #     # DEFT-2020 (Task 1 = STS, Task 2 = CLS)
    #     {"path": "DrBenchmark/DEFT2020", "name": "task_1"},
    #     {"path": "DrBenchmark/DEFT2020", "name": "task_2"},
    #     # DEFT-2021 (Task 1 = multi-label CLS, Task 2 = NER)
    #     {"path": "DrBenchmark/DEFT2021", "name": "cls"},
    #     {"path": "DrBenchmark/DEFT2021", "name": "ner"},
    #     # DiaMed (CLS)
    #     {"path": "DrBenchmark/DiaMED"},
    #     # PxCorpus (NER + CLS)
    #     {"path": "DrBenchmark/PxCorpus", "name": "default"},
    #     # {"path": "DrBenchmark/PxCorpus", "name": "ner"},
    #     # {"path": "DrBenchmark/PxCorpus", "name": "cls"},
    # ]

    root_path = "/lustre/fswork/projects/rech/ilr/commun/DrBenchmark/recipes"
    tasks = [
        {"path": "deft2021/data/local_hf_ner"},
        {"path": "deft2021/data/local_hf_cls"},
        # {"path": "cas/data/local_hf_ner_neg"},
        {"path": "cas/data/local_hf_pos"},
        # {"path": "cas/data/local_hf_cls"},
        # {"path": "cas/data/local_hf_ner_spec"},
        {"path": "diamed/data/local_hf_None"},
        {"path": "mantragsc/data/local_hf_fr_medline"},
        {"path": "mantragsc/data/local_hf_fr_patents"},
        {"path": "mantragsc/data/local_hf_fr_emea"},
        # {"path": "essai/data/local_hf_ner_neg"},
        {"path": "essai/data/local_hf_pos"},
        # {"path": "essai/data/local_hf_cls"},
        # {"path": "essai/data/local_hf_ner_spec"},
        {"path": "frenchmedmcqa/data/local_hf_None"},
        {"path": "deft2020/data/local_hf_task_1"},
        {"path": "deft2020/data/local_hf_task_2"},
        {"path": "pxcorpus/data/local_hf_None"},
        {"path": "morfitt/data/local_hf_source"},
        {"path": "clister/data/local_hf_None"},
        {"path": "e3c/data/local_hf_French_clinical"},
        {"path": "e3c/data/local_hf_French_temporal"},
        {"path": "quaero/data/local_hf_medline"},
        {"path": "quaero/data/local_hf_emea"},
    ]

    data = []
    for entry in tasks:
        entry["full_path"] = f"{root_path}/{entry['path']}"
        path_split = entry["path"].split("/")
        entry["path"] = path_split[0]
        entry["name"] = path_split[-1].replace("local_hf_", "")
        dataset_stats, _, _ = eval_tokenizer(
            tokenizer_name_or_path,
            # dataset_name=entry["path"],
            # dataset_config_name=entry["name"],
            dataset_path=entry["full_path"],
            split="train",
            # batch_size=batch_size,
            num_workers=num_workers,
        )
        data.append(
            {
                "dataset_name": entry["path"],
                "dataset_config": entry["name"],
                **dataset_stats,
            }
        )

    df = pd.DataFrame(data)
    # Add a concise average row across numeric columns
    avg = df.mean(numeric_only=True)
    avg["dataset_name"] = "avg"
    avg["dataset_config"] = ""
    df = pd.concat([df, pd.DataFrame([avg.reindex(df.columns)])], ignore_index=True)

    print(df.to_markdown(index=False))

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Dataset stats saved to: {output_file}")

    # tmp
    tmp_output_file= "/lustre/fswork/projects/rech/ilr/commun/doctobert/model_building/tokenizer/tmp_results/avg_fertility.txt"
    with open(tmp_output_file, "a") as f:
        f.write(f"\n{tokenizer_name_or_path.split('/')[-1]}\n")
        f.write(f"{avg['fertility']:.4f}\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
