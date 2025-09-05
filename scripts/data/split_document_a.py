"""
Word-count splitter with sentence-aware grouping (single-file line input/output).

- Emits chunks that are at most ``max_words`` words, preferring to keep full sentences together.
- If a sentence exceeds ``max_words``, it can either be kept intact or split into word windows
  depending on ``--no-split-long-sentence``.
"""

import os
import re
from typing import Generator
from tqdm import tqdm

split_pattern = re.compile(r"([.!?;]+)")


def split_document_by_sentences(document: str, max_words: int, split_long_sentence: bool) -> Generator[str, None, None]:
    # Split into sentences - this will now include the separators
    parts = split_pattern.split(document)

    # Reconstruct sentences with their punctuation
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and split_pattern.match(parts[i + 1]):
            # Combine sentence with its punctuation
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        else:
            # Handle case where there's no punctuation after
            if parts[i].strip():
                sentences.append(parts[i])
            i += 1

    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

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

            print(
                f"Sentence longer than max_words ({len(sentence_words)} > {max_words}) encountered"
            )

            # tmp
            # with open("tmp_6000.txt", "a", encoding="utf-8") as f:
            #     f.write(sentence + "\n")

            if split_long_sentence:
                # Split long sentence into word chunks
                for i in range(0, len(sentence_words), max_words):
                    chunk = " ".join(sentence_words[i : i + max_words])
                    yield chunk
            else:
                # Keep the long sentence intact
                yield sentence
        else:
            current_chunk.extend(sentence_words)
            current_word_count += len(sentence_words)

    # Yield remaining words if any
    if current_chunk:
        yield " ".join(current_chunk)


def main(
    input_file,
    output_file,
    max_words: int = 1024,
    buffer_size: int = 8192,
    show_progress: bool = True,
    split_long_sentence: bool = True,
):
    # Get file size for progress tracking
    file_size = os.path.getsize(input_file) if show_progress else 0
    processed_bytes = 0

    original_count = 0
    new_doc_count = 0

    # Open both files - read input line by line, write output incrementally
    with open(input_file, "r", encoding="utf-8", buffering=buffer_size) as infile, open(
        output_file, "w", encoding="utf-8", buffering=buffer_size
    ) as outfile:

        pbar = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc="Splitting",
            disable=not show_progress,
        )

        for line_num, line in enumerate(infile, 1):
            line_bytes = len(line.encode("utf-8"))
            processed_bytes += line_bytes
            pbar.update(line_bytes)

            document = line.strip()
            if not document:
                continue

            original_count += 1
            words = document.split()

            if len(words) <= max_words:
                outfile.write(document + "\n")
                new_doc_count += 1
            else:
                # Process this document and write chunks immediately
                chunks = split_document_by_sentences(
                    document, max_words, split_long_sentence
                )
                for chunk in chunks:
                    outfile.write(chunk + "\n")
                    new_doc_count += 1

            # Flush output buffer periodically for very large files
            if line_num % 5000 == 0:
                outfile.flush()

        pbar.close()

    print(f"Processed {original_count:,} original documents")
    print(f"Created {new_doc_count:,} documents after splitting")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--max-words", type=int, default=1024)
    parser.add_argument("--buffer-size", type=int, default=8192)
    parser.add_argument(
        "--no-progress", action="store_true", help="Hide tqdm progress bar"
    )
    parser.add_argument(
        "--no-split-long-sentence",
        action="store_true",
        help="Do not split individual sentences that exceed --max-words; output them intact instead.",
    )
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        max_words=args.max_words,
        buffer_size=args.buffer_size,
        show_progress=not args.no_progress,
        split_long_sentence=not args.no_split_long_sentence,
    )
