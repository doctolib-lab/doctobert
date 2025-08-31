"""
Paragraph-first splitter with sentence/word fallback (HF Datasets I/O).

- Preserves original document boundaries; avoids crossing paragraphs when possible.
- For paragraphs longer than ``max_words``, splits into sentences and packs sentences into chunks
  up to ``max_words`` words. For sentences longer than ``max_words``, either split by words into
  windows or keep intact based on flags.
"""

import os
import re
from typing import Generator

from datasets import load_dataset
from tqdm import tqdm

sentence_split_pattern = re.compile(r"([.!?;]+)")
paragraph_split_pattern = re.compile(r"\n\s*\n")


def split_document_by_sentences(
    document: str,
    max_words: int,
    split_long_paragraph: bool,
    split_long_sentence: bool,
) -> Generator[str, None, None]:
    """Yield chunks of *document* not exceeding *max_words* words."""

    # Iterate over paragraphs first, preventing crossing paragraph boundaries
    for paragraph in paragraph_split_pattern.split(document):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Fast-path: paragraph short enough ‑> keep as-is
        paragraph_word_count = len(paragraph.split())
        if paragraph_word_count <= max_words:
            yield paragraph
            continue

        # Paragraph too long – either keep it intact or fall back to sentence-level splitting
        if not split_long_paragraph:
            print(f"Paragraph longer than max_words ({paragraph_word_count} > {max_words}) encountered")
            yield paragraph
            continue

        # Split into sentences
        parts = sentence_split_pattern.split(paragraph)

        # Reconstruct sentences with their punctuation
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and sentence_split_pattern.match(parts[i + 1]):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i])
                i += 1

        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Split into words
            sentence_words = sentence.split()

            # If adding this sentence would exceed limit, yield current chunk
            if current_word_count + len(sentence_words) > max_words and current_chunk:
                yield " ".join(current_chunk)
                current_chunk = []
                current_word_count = 0

            # If single sentence is too long, split it by words
            if len(sentence_words) > max_words:
                # First yield any existing chunk
                if current_chunk:
                    yield " ".join(current_chunk)
                    current_chunk = []
                    current_word_count = 0

                print(f"Sentence longer than max_words ({len(sentence_words)} > {max_words}) encountered")

                if split_long_sentence:
                    # Split long sentence into word chunks
                    for j in range(0, len(sentence_words), max_words):
                        chunk = " ".join(sentence_words[j : j + max_words])
                        yield chunk
                else:
                    # Keep the long sentence intact
                    yield sentence
            else:
                current_chunk.extend(sentence_words)
                current_word_count += len(sentence_words)

        # Yield remaining words if any (per paragraph)
        if current_chunk:
            yield " ".join(current_chunk)


def main(
    input_path: str,
    output_path: str,
    max_words: int = 1024,
    split_long_paragraph: bool = True,
    split_long_sentence: bool = True,
    num_proc: int | None = None,
):
    # Load dataset
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
        # ds = load_from_disk(input_path)

    original_count = len(ds)
    print(f"Loaded {original_count:,} documents")

    def _split_examples(examples):
        all_chunks = []

        for doc in examples["text"]:
            all_chunks.extend(
                split_document_by_sentences(
                    doc,
                    max_words,
                    split_long_paragraph,
                    split_long_sentence,
                )
            )

        return {"text": all_chunks}

    processed_ds = ds.map(
        _split_examples,
        batched=True,
        batch_size=1,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Splitting documents",
    )

    new_doc_count = len(processed_ds)
    print(f"Processed {original_count:,} original documents -> {new_doc_count:,} chunks")

    # Save results
    if output_path.endswith(".txt"):
        # Stream writing avoids loading everything into memory
        with open(output_path, "w", encoding="utf-8") as f:
            for row in tqdm(processed_ds, desc="Writing"):
                f.write(row["text"] + "\n")
    elif output_path.endswith(".jsonl"):
        processed_ds.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif output_path.endswith(".parquet"):
        processed_ds.to_parquet(output_path)
    else:
        # Directory – Arrow + metadata (HF native format)
        processed_ds.save_to_disk(output_path)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
