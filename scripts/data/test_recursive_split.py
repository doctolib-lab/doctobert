"""
Tests for the recursive_split function in split_document.py.
Usage: pytest test_recursive_split.py
"""

import importlib.util
from pathlib import Path


class FakeCharTokenizer:
    """
    Minimal tokenizer for tests:
    - Each Unicode character is a token
    - decode reconstructs the exact substring
    """

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _load_split_module():
    this_dir = Path(__file__).resolve().parent
    module_path = this_dir / "split_document.py"
    spec = importlib.util.spec_from_file_location("split_document", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_identity_when_under_limit():
    mod = _load_split_module()
    tokenizer = FakeCharTokenizer()

    text = "Hello.\n\nWorld!\nNext line."
    out = mod.recursive_split(text, max_tokens=10_000, tokenizer=tokenizer)

    # Should return the original text as a single chunk
    assert out == [text]


def test_preserves_separators_roundtrip_join():
    mod = _load_split_module()
    tokenizer = FakeCharTokenizer()

    text = "Para1 line1.\nPara1 line2.\n\nPara2 line1.\nPara2 line2."
    out = mod.recursive_split(text, max_tokens=25, tokenizer=tokenizer)

    # Rejoining all chunks must yield the original string (separators preserved)
    assert "".join(out) == text

    # Each chunk must respect the token budget
    for chunk in out:
        assert len(tokenizer.encode(chunk, add_special_tokens=False)) <= 25


def test_final_fallback_token_slicing_no_separators():
    mod = _load_split_module()
    tokenizer = FakeCharTokenizer()

    # No whitespace or punctuation, forces fallback to token-slicing path
    text = "x" * 53
    out = mod.recursive_split(text, max_tokens=10, tokenizer=tokenizer)

    # Expect exact slicing by tokens (characters)
    assert out == [
        "x" * 10,
        "x" * 10,
        "x" * 10,
        "x" * 10,
        "x" * 10,
        "x" * 3,
    ]
