"""
Token-length splitter with greedy packing.

- Splits each document into chunks that fit within ``max_tokens`` for a given HF tokenizer.
- Split priority: paragraph breaks (``\n{2,}``), then single newlines, then simple sentence boundaries,
  then generic whitespace; final fallback slices the token stream directly.
- Performs greedy adjacent re-packing to merge neighboring pieces while keeping each packed chunk
  ``<= max_tokens`` to reduce padding waste at train time.
"""

import math
import os
import re
from collections import defaultdict
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

SEPARATORS: list[str] = [
    r"\n{2,}",  # paragraph break
    r"\n",  # line break
    r"(?<=[.!?;])\s+",  # sentence boundary (simple heuristic)
    r"\s+",  # fallback = any whitespace between words
]


class FakeWordTokenizer:
    """Minimal word-level tokenizer with an HF-like interface."""

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
        self._word_to_id = {}
        self._id_to_word = {}
        self.word_split_pattern = re.compile(r"\S+")

    def _tokenize(self, text: str) -> list[str]:
        text_proc = text.lower() if self.lowercase else text
        # Split on whitespace; keep punctuation attached to tokens similar to simple "by word"
        return self.word_split_pattern.findall(text_proc)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002 (parity with HF sig)
        tokens = self._tokenize(text)
        ids = []
        for tok in tokens:
            if tok not in self._word_to_id:
                new_id = len(self._word_to_id) + 1  # start ids at 1
                self._word_to_id[tok] = new_id
                self._id_to_word[new_id] = tok
            ids.append(self._word_to_id[tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        # Reconstruct with single spaces; original spacing is not preserved.
        words = [self._id_to_word.get(i, "") for i in ids]
        return " ".join(w for w in words if w)


def build_tokenizer(tokenizer_name_or_path: str | None = None) -> PreTrainedTokenizer | FakeWordTokenizer:
    """Return either a real HF tokenizer or a simple word-level fake tokenizer."""
    if tokenizer_name_or_path is not None:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    return FakeWordTokenizer()


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    # return f"{root}_{idx:09d}{ext}"
    os.makedirs(os.path.dirname(root), exist_ok=True)
    return f"{root}/{idx:09d}{ext}"


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


def get_token_len(text: str, tokenizer: PreTrainedTokenizer) -> int:
    "Fast token count with special tokens disabled."
    return len(tokenizer.encode(text, add_special_tokens=False))


def fits_within_max_tokens(text: str, max_tokens: int, tokenizer: PreTrainedTokenizer) -> bool:
    """Check if text has less than max_tokens tokens."""
    # heurstic to first check num of words to avoid tokenizing long text
    fertility = 1  # lower than real fertility
    if len(text.split()) > max_tokens / fertility:
        return False
    return get_token_len(text, tokenizer) <= max_tokens


def split_keep_sep(text: str, pattern: str) -> list[str]:
    """Split text on regex pattern but keep each delimiter at the end of the preceding segment."""
    pieces, last = [], 0
    for match in re.finditer(pattern, text):
        end = match.end()
        pieces.append(text[last:end])
        last = end
    pieces.append(text[last:])  # add tail
    return pieces


def recursive_split(
    text: str,
    max_tokens: int = 1024,
    tokenizer: PreTrainedTokenizer = None,
    seps: list[str] = SEPARATORS,
) -> list[str]:
    """Recursively split text until every chunk <= max_tokens tokens."""
    # First, check if the entire text is short enough, if so, return the text as is
    # if get_token_len(text, tokenizer) <= max_tokens:
    if fits_within_max_tokens(text, max_tokens, tokenizer):
        return [text]

    # Final fallback: no splitters left -> slice the tok stream itself
    if not seps:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]

    # If the text is too long, split it recursively using the separators
    pattern = seps[0]
    pieces = []
    for piece in split_keep_sep(text, pattern):
        # if get_token_len(piece, tokenizer) <= max_tokens:
        if fits_within_max_tokens(piece, max_tokens, tokenizer):
            pieces.append(piece)
        else:  # still too big -> recurse with the next splitter
            pieces.extend(recursive_split(piece, max_tokens, tokenizer, seps[1:]))

    # After splitting, pack the pieces back to <= max_tokens tokens
    # Keep more context by packing the pieces back together
    packed = []
    buffer = ""
    for piece in pieces:
        if not buffer:
            buffer = piece
            continue
        # if get_token_len(buffer + piece, tokenizer) <= max_tokens:
        if fits_within_max_tokens(buffer + piece, max_tokens, tokenizer):
            buffer += piece
        else:
            packed.append(buffer)
            buffer = piece
    if buffer:
        packed.append(buffer)

    return packed


def split_documents(
    examples: dict[str, Any],
    max_tokens: int,
    tokenizer: PreTrainedTokenizer | FakeWordTokenizer,
) -> dict[str, Any]:
    """Split documents into chunks where each chunk is <= max_tokens tokens."""
    outputs = defaultdict(list)
    for i, text in enumerate(examples["text"]):
        chunks = recursive_split(text, max_tokens=max_tokens, tokenizer=tokenizer)
        outputs["text"].extend(chunks)
        outputs["num_words"].extend([len(chunk.split()) for chunk in chunks])
        # repeat other columns
        for k, v in examples.items():
            if k == "text":
                # don't add original text
                continue
            # tmp: remove num_words
            # elif k == "num_words":
            #     outputs["original_num_words"].extend([v[i]] * len(chunks))
            elif k == "num_words":
                # skip original num_words since we already compute new values above
                continue
            else:
                outputs[k].extend([v[i]] * len(chunks))
    return outputs


def main(
    input_path: str,
    output_path: str,
    tokenizer_name_or_path: str | None = None,
    max_tokens: int = 1024,
    num_proc: int | None = None,
    shuffle: bool = False,
    output_shard_size: int | None = None,
):
    # Load dataset
    dataset = load_local_dataset(input_path)

    # Shuffle, since some long documents are at the end of the dataset
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    ## tmp: keep only med01 ∧ edu4 (medical_entity_density >= 0.1 AND edu_quality_normalized_score >= 4)
    ## Disabled for the rewritten-corpora pass (no pretraining-quality filter on rewriting output).
    # dataset = dataset.filter(
    #     lambda m, e: m >= 0.1 and e >= 4,
    #     input_columns=["medical_entity_density", "edu_quality_normalized_score"],
    #     num_proc=num_proc,
    #     desc="Filtering med01 & edu4",
    # )
    # print(f"After med01 & edu4 filter: {len(dataset):,} documents")

    original_count = len(dataset)
    print(f"Loaded {original_count:,} documents")

    # Load tokenizer
    tokenizer = build_tokenizer(tokenizer_name_or_path)

    # Split documents
    processed_dataset = dataset.map(
        split_documents,
        batched=True,
        batch_size=1,
        num_proc=num_proc,
        # writer_batch_size=1000,
        # remove_columns=dataset.column_names,
        fn_kwargs={"max_tokens": max_tokens, "tokenizer": tokenizer},
        desc="Splitting documents",
    )

    new_doc_count = len(processed_dataset)
    print(f"Processed {original_count:,} original documents -> {new_doc_count:,} chunks")

    # Save results
    save_hf_dataset(processed_dataset, output_path, output_shard_size=output_shard_size)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
