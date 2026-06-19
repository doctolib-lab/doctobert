"""
Convert SPM BPE tokenizer to HuggingFace pretrained format.
"""

import sentencepiece as spm
from tokenizers import AddedToken, decoders, normalizers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, convert_slow_tokenizer
import json
from tokenizers import Tokenizer
from tokenizers import Regex

def remove_replace_normalizers_in_tokenizer(tok: Tokenizer) -> Tokenizer:
    """
    Remove any normalizer node with type == 'Replace' from the tokenizer graph.
    Works recursively inside Sequence normalizers.
    """
    conf = json.loads(tok.to_str())
    conf["normalizer"] = _prune_replace_nodes(conf.get("normalizer"))
    # if the normalizer becomes empty, drop it
    if conf["normalizer"] is None:
        conf.pop("normalizer", None)
    return Tokenizer.from_str(json.dumps(conf))

def _prune_replace_nodes(node):
    if not node:
        return None
    t = node.get("type")
    if t == "Replace":
        return None
    if t == "Sequence":
        kept = []
        for sub in node.get("normalizers", []):
            pruned = _prune_replace_nodes(sub)
            if pruned is not None:
                kept.append(pruned)
        # if nothing left in the sequence, drop it
        if not kept:
            return None
        node = dict(node)  # shallow copy
        node["normalizers"] = kept
        return node
    # keep any other normalizer as-is (Strip, Precompiled, NFKC, Lowercase, etc.)
    return node

def main(
    spm_path: str,
    output_dir: str,
    model_max_length: int = 8192,
    target_vocab_size: int | None = None,
    round_to: int = 64,
    for_roberta: bool = False,
):
    """Convert SPM tokenizer to HF.

    `for_roberta` flag must match what was passed to train_spm.py:
      - False (default): SPM has only `<unk>` at id 0; this script APPENDS [UNK]/[CLS]/
        [SEP]/[PAD]/[MASK] at the tail. Works for ModernBERT (RoPE, no position
        table). The resulting pad_token_id is high (~vocab_size), which breaks HF
        RoBERTa's position formula.
      - True: SPM already has [UNK]=0, [PAD]=1, [CLS]=2, [SEP]=3, [MASK]=4 reserved
        at training time. This script REGISTERS them by name (no re-adding), keeping
        pad_token_id=1 — matches HF for_roberta-base convention.
    """
    # 1) Load SPM and save tokenizer.json
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    sp.vocab_file = spm_path  # field read by the converter
    print(f"SPM vocab size: {sp.get_piece_size():,}")

    # convert
    converter = convert_slow_tokenizer.SpmConverter(sp)
    base_tokenizer = converter.converted()

    # mistral-7b-style
    # base_tokenizer.normalizer = None
    # Strip default collapsing spaces
    base_tokenizer = remove_replace_normalizers_in_tokenizer(base_tokenizer)
    base_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="first",  # SPM dummy prefix, but only once at BOS
        split=False,             # don't split on spaces; keep counts intact
    )

    # gemma3-style
    # base_tokenizer.normalizer = normalizers.Replace(" ", "▁")  # todo: add rather than replace
    # base_tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(" "), behavior="merged_with_previous")

    base_tokenizer.model.byte_fallback = True

    base_tokenizer.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),   # turn metaspace back into real space
        decoders.ByteFallback(),      # keep byte safety
        decoders.Fuse(),              # fuse pieces that should be merged
        decoders.Strip(" ", 1, 0),    # drop the single leading dummy space
    ])

    # save
    base_tokenizer_path = f"{output_dir}/tokenizer.json"
    base_tokenizer.save(base_tokenizer_path)

    # 2) Wrap as HF fast tokenizer.
    if for_roberta:
        # Specials already at SPM ids 0-4 (set by train_spm.py with for_roberta=True).
        # Pass them as plain strings → HF resolves them in the existing vocab (no re-add).
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=base_tokenizer_path,
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            # MASK keeps the lstrip=True flag from the original; pass as AddedToken so HF
            # carries the flag forward. Since "[MASK]" is already in the vocab at id 4,
            # HF will associate the metadata rather than re-add.
            mask_token=AddedToken("[MASK]", special=True, normalized=False, lstrip=True),
            model_max_length=model_max_length,
            clean_up_tokenization_spaces=False,
        )
    else:
        # ModernBERT (existing) path: add specials at the tail of the vocab.
        unk = AddedToken("[UNK]", special=True, normalized=False)
        cls_ = AddedToken("[CLS]", special=True, normalized=False)
        sep = AddedToken("[SEP]", special=True, normalized=False)
        pad = AddedToken("[PAD]", special=True, normalized=False)
        mask = AddedToken("[MASK]", special=True, normalized=False, lstrip=True)  # ModernBERT-like lstrip

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=base_tokenizer_path,
            unk_token=unk,
            cls_token=cls_,
            sep_token=sep,
            pad_token=pad,
            mask_token=mask,
            model_max_length=model_max_length,  # set your context length
            clean_up_tokenization_spaces=False,  # whitespace/punctuation auto-fixes on decode
        )
    # hf_tokenizer.model_max_length = model_max_length
    print(f"HF vocab size: {len(hf_tokenizer):,}")

    # bos and eos tokens
    hf_tokenizer.bos_token = hf_tokenizer.cls_token or "[CLS]"
    hf_tokenizer.eos_token = hf_tokenizer.sep_token or "[SEP]"

    # post-processor to insert when add_special_tokens=True
    hf_tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", hf_tokenizer.convert_tokens_to_ids("[CLS]")),
            ("[SEP]", hf_tokenizer.convert_tokens_to_ids("[SEP]")),
        ],
    )

    # add format tokens
    format_tokens = []
    # for k in range(2, 9):
    #     format_tokens.append(AddedToken("▁" * k, special=False, normalized=False, lstrip=False, rstrip=False, single_word=False))
    # for k in range(1, 9):
    #     format_tokens.append(AddedToken("\n" * k, special=False, normalized=False, lstrip=False, rstrip=False, single_word=False))
    # for k in range(1, 9):
    #     format_tokens.append(AddedToken("\t" * k, special=False, normalized=False, lstrip=False, rstrip=False, single_word=False))
    for k in range(1, 7):
        format_tokens.append(AddedToken("#" * k, special=False, normalized=False, lstrip=False, rstrip=False, single_word=False))
    hf_tokenizer.add_tokens(format_tokens)

    # Optional: ModernBERT-style placeholders
    normal_placeholders = [
        "|||PATIENT_NAME|||",
        "|||AGE|||",
        "|||SEX|||",
        "|||DATE|||",
        "|||TIME|||",
        "|||ADDRESS|||",
        "|||PHONE_NUMBER|||",
        "|||EMAIL_ADDRESS|||",
        "|||HOSPITAL|||",
        "|||SERVICE|||",
        "|||PHYSICIAN_NAME|||",
    ]

    special_placeholders = [
        AddedToken("<|endoftext|>", special=True, normalized=False),
        AddedToken("<|padding|>", special=True, normalized=False),
    ]
    hf_tokenizer.add_tokens(normal_placeholders)
    hf_tokenizer.add_special_tokens({"additional_special_tokens": special_placeholders})
    print(f"HF vocab size: {len(hf_tokenizer):,}")

    # 3) Adjust vocab size
    # Prefer exact target_vocab_size if provided and feasible
    # Fallback to rounding up to the nearest multiple of `round_to`.
    current_size = len(hf_tokenizer)
    if target_vocab_size is not None and target_vocab_size > 0:
        if current_size < target_vocab_size:
            # pad to exact target size
            num_to_add = target_vocab_size - current_size
            print(f"Adding {num_to_add} tokens to reach target_vocab_size {target_vocab_size}")
            to_add = [f"[unused{i}]" for i in range(num_to_add)]
            added = hf_tokenizer.add_tokens(to_add)
            assert added == num_to_add, f"Expected to add {num_to_add} tokens, but added {added}"
        elif current_size > target_vocab_size:
            # cannot shrink safely; fallback to round_to behavior below
            print(f"Warning: current vocab ({current_size}) exceeds target_vocab_size ({target_vocab_size}). ")
    # If no target specified, round up to multiple of `round_to`.
    elif round_to is not None and round_to > 0:
        if remainder := current_size % round_to:
            num_to_add = round_to - remainder
            print(f"Adding {num_to_add} tokens to reach multiple of {round_to}")
            to_add = [f"[unused{i}]" for i in range(num_to_add)]
            added = hf_tokenizer.add_tokens(to_add)
            assert added == num_to_add, f"Expected to add {num_to_add} tokens, but added {added}"

    # 4) Long context (optional) + save
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Final HF vocab size: {len(hf_tokenizer):,}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
