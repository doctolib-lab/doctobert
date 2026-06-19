"""
Train SPM tokenizer.

Docs:
https://github.com/google/sentencepiece/blob/master/python/README.md
https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
https://chromium.googlesource.com/chromium/src/+/main/third_party/sentencepiece/src/doc/options.md
"""

import sentencepiece as spm


def main(
    corpus: str,
    model_prefix: str = "tokenizer",
    model_type: str = "bpe",
    vocab_size: int = 32_000,
    byte_fallback: bool = True,
    character_coverage: float = 0.9995,
    num_threads: int = 16,
    max_lines: int | None = None,
    for_roberta: bool = False,
):
    """Train SPM BPE tokenizer.

    `for_roberta` flag controls special-token placement:
      - False (default, ModernBERT path): pad/bos/eos disabled at SPM level. Only
        `<unk>` at id 0. Downstream `convert_spm_to_hf.py` APPENDS [UNK]/[CLS]/[SEP]/
        [PAD]/[MASK] at the tail of the vocab. Works fine for ModernBERT (RoPE,
        no position-table sizing).
      - True (RoBERTa path): reserves [UNK]/[PAD]/[CLS]/[SEP] at SPM ids 0–3 via the
        unk_id/pad_id/bos_id/eos_id slots, plus [MASK] at id 4 via user_defined_symbols.
        Required so that HF RoBERTa's position formula (cumsum + pad_id) doesn't
        overflow the position table: needs pad_id ≤ max_position_embeddings - max_seq_len.
    """
    user_defined_symbols = []
    # Place [MASK] FIRST when for_roberta=True so it lands at id 4 (right after the
    # 4 reserved SPM slots), keeping all 5 specials in the low-id range.
    if for_roberta:
        user_defined_symbols.append("[MASK]")
    for k in range(2, 9):
        user_defined_symbols.append("▁" * k)  # ▁
    for k in range(1, 9):
        user_defined_symbols.append("\n" * k)  # LF
    for k in range(1, 9):
        user_defined_symbols.append("\t" * k)  # TAB

    # other options:
    # --pad_id=0 --pad_piece="[PAD]" \         # BERT-style specials
    # --unk_id=1 --unk_piece="[UNK]" \
    # --bos_id=2 --bos_piece="[BOS]" \
    # --eos_id=3 --eos_piece="[EOS]" \
    # --user_defined_symbols='[CLS],[SEP],[MASK]'

    # --split_by_whitespace=false + --normalization_rule_name=identity = no space collapse

    if for_roberta:
        # HF RoBERTa convention: PAD at id 1 (so cumsum + pad_id fits in position table).
        # SPM's bos/eos slots reused as CLS/SEP (HF wrapping registers them by name).
        special_args = {
            "unk_id": 0, "unk_piece": "[UNK]",
            "pad_id": 1, "pad_piece": "[PAD]",
            "bos_id": 2, "bos_piece": "[CLS]",
            "eos_id": 3, "eos_piece": "[SEP]",
        }
    else:
        # ModernBERT (existing) layout: specials disabled at SPM level, appended at tail
        # by convert_spm_to_hf.py.
        special_args = {
            "unk_id": 0, "unk_piece": "<unk>",  # keep SPM's internal <unk> (not used in HF)
            "bos_id": -1,
            "eos_id": -1,
            "pad_id": -1,
        }

    train_args = {
        # I/O
        "input": corpus,  # input files (comma-separated)
        "model_prefix": model_prefix,  # output prefix
        # Model / vocab
        "model_type": model_type,  # unigram or bpe
        "vocab_size": vocab_size,  # try 30k-50k
        "byte_fallback": byte_fallback,  # guarantees zero UNKs via byte backoff
        "character_coverage": character_coverage,  # by default: 0.9995, or 0.9999, 1.0
        "hard_vocab_limit": False,  # soft-limit so it won't crash on small corpora
        # Normalization & whitespace behavior
        # normalization_rule_name = "identity",          # nmt_nfkc or identity; default nmt_nfkc maps all Unicode whitespace to a space and collapses repeats
        "remove_extra_whitespaces": False,  # removes leading, trailing, and duplicate internal whitespace
        "allow_whitespace_only_pieces": True,  # allows whitespace-only pieces to be added to the vocabulary
        "add_dummy_prefix": True,  # standard "▁" word boundary marker
        # split_by_whitespace=False,               # whitespace is not used as a delimiter but is encoded into tokens via the meta-symbol
        # split_by_unicode_script=False,           # let BPE learn across Latin + Greek letters
        # Number / digit splitting
        "split_by_number": True,  # "120 mg" → ["120", " mg"]
        "split_digits": True,  # "2025" → ["2", "0", "2", "5"]
        # required_chars = "0123456789",               # ensure digits are covered
        # Limits & performance
        "max_sentence_length": 8192,  # for long lines
        "num_threads": num_threads,  # align with allocated CPU cores
        # Special tokens (for_roberta-aware)
        **special_args,
        # "user_defined_symbols": user_defined_symbols,
    }

    if user_defined_symbols:
        train_args["user_defined_symbols"] = user_defined_symbols  # literal backslash-n and backslash-t

    if max_lines is not None:
        train_args.update(
            {
                "input_sentence_size": max_lines,  # sample 5M lines
                "shuffle_input_sentence": True,  # uniform sampling boosts coverage
            }
        )

    spm.SentencePieceTrainer.train(**train_args)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
