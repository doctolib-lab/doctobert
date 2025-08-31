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


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    # return f"{root}_{idx:09d}{ext}"
    os.makedirs(os.path.dirname(root), exist_ok=True)
    return f"{root}/{idx:09d}{ext}"


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


def main(
    input_path: str,
    output_path: str,
    tokenizer_name_or_path: str,
    max_tokens: int = 1024,
    num_proc: int | None = None,
    shuffle: bool = False,
    output_shard_size: int | None = None,
):
    # Load dataset
    dataset = load_hf_dataset(input_path)

    # Shuffle, since some long documents are at the end of the dataset
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    original_count = len(dataset)
    print(f"Loaded {original_count:,} documents")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # Split documents
    def _split_examples(examples: dict[str, Any]) -> dict[str, Any]:
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
                elif k == "num_words":
                    outputs["original_num_words"].extend([v[i]] * len(chunks))
                else:
                    outputs[k].extend([v[i]] * len(chunks))
        return outputs

    processed_dataset = dataset.map(
        _split_examples,
        batched=True,
        batch_size=1,
        num_proc=num_proc,
        # writer_batch_size=1000,
        # remove_columns=dataset.column_names,
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
