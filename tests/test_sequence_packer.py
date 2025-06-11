# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import os, sys, random, itertools

# import pytest
import torch
from torch import tensor

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence_packer import GreedyBestFitSequencePacker, split_packed_batch

max_seq_len = 10


def generate_sequences(n, max_seq_len=10):
    "generate n seqs of random length"
    return [[i] * random.randint(0, max_seq_len) for i in range(n)]


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


xs = [
    [0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3],
    [4, 4, 4, 4, 4, 4],
    [5, 5, 5, 5, 5, 5],
    [6, 6, 6, 6, 6, 6, 6],
    [7, 7, 7],
    [8, 8],
    [],
    [10, 10, 10, 10, 10, 10, 10],
    [11, 11, 11, 11],
    [12, 12, 12, 12],
    [13, 13, 13, 13, 13, 13, 13, 13],
    [14, 14, 14],
    [15, 15, 15, 15, 15, 15, 15],
    [16, 16, 16, 16, 16, 16, 16],
    [17, 17, 17, 17, 17, 17, 17, 17],
    [18],
    [19, 19, 19, 19],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [21, 21, 21, 21],
    [22, 22, 22, 22, 22, 22, 22, 22],
    [23, 23],
    [24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
    [25, 25, 25, 25, 25, 25, 25],
]


def compare_structures(s1, s2, rtol=1e-5, atol=1e-8, path="", message_prefix=""):
    """Recursively compare nested structures (lists, dicts, tensors). Raise AssertionError on mismatch."""
    prefix = f"{message_prefix} Mismatch at {path if path else 'root'}:"
    s1_type = type(s1)
    s2_type = type(s2)
    if s1_type is not s2_type:
        raise AssertionError(f"{prefix} Type mismatch: {s1_type} != {s2_type}. Values: {s1} vs {s2}")

    if isinstance(s1, dict):
        s1_keys = set(s1.keys())
        s2_keys = set(s2.keys())
        if s1_keys != s2_keys:
            diff = s1_keys.symmetric_difference(s2_keys)
            raise AssertionError(f"{prefix} Dict key mismatch: {diff}. Keys: {s1_keys} vs {s2_keys}")
        for k in s1_keys:
            compare_structures(s1[k], s2[k], rtol, atol, path=f"{path}[{k!r}]", message_prefix=message_prefix)
    elif isinstance(s1, list):
        len1 = len(s1)
        len2 = len(s2)
        if len1 != len2:
            raise AssertionError(f"{prefix} List length mismatch: {len1} != {len2}. Lists: {s1} vs {s2}")
        for i, (i1, i2) in enumerate(zip(s1, s2)):
            compare_structures(i1, i2, rtol, atol, path=f"{path}[{i}]", message_prefix=message_prefix)
    elif isinstance(s1, torch.Tensor):
        if not torch.allclose(s1, s2, rtol=rtol, atol=atol):
            # Find first mismatch index for better debugging
            mismatch_indices = torch.where(~torch.isclose(s1, s2, rtol=rtol, atol=atol))
            first_mismatch_idx = (
                tuple(idx[0].item() for idx in mismatch_indices) if len(mismatch_indices[0]) > 0 else None
            )
            err_msg = f"{prefix} Tensor mismatch.\nExpected:\n{s1}\nActual:\n{s2}"
            if first_mismatch_idx:
                err_msg += f"\nFirst mismatch at index: {first_mismatch_idx}, values: {s1[first_mismatch_idx]} vs {s2[first_mismatch_idx]}"
            raise AssertionError(err_msg)
    elif s1 != s2:
        raise AssertionError(f"{prefix} Value mismatch: {s1} != {s2}")

    # If we reach here without raising, structures are equal at this level
    return  # No need to return True


def test_packer():
    input_batch_size = 4
    xds = [{"input_ids": seq} for seq in xs]
    input_batches = list(batched(xds, input_batch_size))
    d = {
        "src_iterable": input_batches,
        "src_batch_size": input_batch_size,
        "src_max_seq_len": max_seq_len,
        "out_batch_size": 5,
        "out_pseq_len": 15,
        "buffer_size": 5,
        "pad_token_id": -1,
        "mask_token_id": -2,
        "ignore_token_id": -3,
        "mask_prob": 0.0,
        "seed": 42,
        "suppress_masking": True,
    }

    out_batches = list(GreedyBestFitSequencePacker(**d))

    out_expected = [
        {
            "input_ids": tensor(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 8, 8],
                    [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, -1, -1],
                    [7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 18],
                    [12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14],
                ]
            ),
            "labels": None,
            "cu_seqlens": [
                tensor([0, 4, 9, 15], dtype=torch.int32),
                tensor([0, 10, 13, 15], dtype=torch.int32),
                tensor([0, 6, 13, 15], dtype=torch.int32),
                tensor([0, 3, 10, 14, 15], dtype=torch.int32),
                tensor([0, 4, 12, 15], dtype=torch.int32),
            ],
            "max_seqlen": [6, 10, 7, 7, 8],
        },
        {
            "input_ids": tensor(
                [
                    [19, 19, 19, 19, 17, 17, 17, 17, 17, 17, 17, 17, 23, 23, -1],
                    [16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, -1],
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, -1],
                    [22, 22, 22, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25, 25, 25],
                    [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, -1, -1, -1, -1, -1],
                ]
            ),
            "labels": None,
            "cu_seqlens": [
                tensor([0, 4, 12, 14, 15], dtype=torch.int32),
                tensor([0, 7, 14, 15], dtype=torch.int32),
                tensor([0, 10, 14, 15], dtype=torch.int32),
                tensor([0, 8, 15], dtype=torch.int32),
                tensor([0, 10, 15], dtype=torch.int32),
            ],
            "max_seqlen": [8, 7, 10, 8, 10],
        },
    ]

    compare_structures(out_expected, out_batches, message_prefix="test_packer:")
    pass


def test_packer_masking():
    input_batch_size = 4
    xds = [{"input_ids": seq} for seq in xs]
    input_batches = list(batched(xds, input_batch_size))
    d = {
        "src_iterable": input_batches,
        "src_batch_size": input_batch_size,
        "src_max_seq_len": max_seq_len,
        "out_batch_size": 5,
        "out_pseq_len": 15,
        "buffer_size": 5,
        "pad_token_id": -1,
        "mask_token_id": -2,
        "ignore_token_id": -3,
        "mask_prob": 0.0,  # 0 mask probability, like no masking
        "seed": 42,
        "suppress_masking": False,
    }

    out_batches = list(GreedyBestFitSequencePacker(**d))

    # expected given a seed of 42!
    out_expected = [
        {
            "input_ids": tensor(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 8, 8],
                    [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, -1, -1],
                    [7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 18],
                    [12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14],
                ]
            ),
            "labels": tensor(
                [
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 4, 9, 15], dtype=torch.int32),
                tensor([0, 10, 13, 15], dtype=torch.int32),
                tensor([0, 6, 13, 15], dtype=torch.int32),
                tensor([0, 3, 10, 14, 15], dtype=torch.int32),
                tensor([0, 4, 12, 15], dtype=torch.int32),
            ],
            "max_seqlen": [6, 10, 7, 7, 8],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        {
            "input_ids": tensor(
                [
                    [19, 19, 19, 19, 17, 17, 17, 17, 17, 17, 17, 17, 23, 23, -1],
                    [16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, -1],
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, -1],
                    [22, 22, 22, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25, 25, 25],
                    [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, -1, -1, -1, -1, -1],
                ]
            ),
            "labels": tensor(
                [
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 4, 12, 14, 15], dtype=torch.int32),
                tensor([0, 7, 14, 15], dtype=torch.int32),
                tensor([0, 10, 14, 15], dtype=torch.int32),
                tensor([0, 8, 15], dtype=torch.int32),
                tensor([0, 10, 15], dtype=torch.int32),
            ],
            "max_seqlen": [8, 7, 10, 8, 10],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ]
            ),
        },
    ]

    compare_structures(out_expected, out_batches, message_prefix="test_packer_masking:")
    pass


def test_packer_masking_randomly():
    input_batch_size = 4
    xds = [{"input_ids": seq} for seq in xs]
    input_batches = list(batched(xds, input_batch_size))
    d = {
        "src_iterable": input_batches,
        "src_batch_size": input_batch_size,
        "src_max_seq_len": max_seq_len,
        "out_batch_size": 5,
        "out_pseq_len": 15,
        "buffer_size": 5,
        "pad_token_id": -1,
        "mask_token_id": -2,
        "ignore_token_id": -3,
        "mask_prob": 0.5,  # <== extensive masking and word randomization
        "seed": 42,
        "suppress_masking": False,
    }

    out_batches = list(GreedyBestFitSequencePacker(**d))

    out_expected = [
        {
            "input_ids": tensor(
                [
                    [0, 4, 0, 0, -2, 1, 1, 1, -2, 4, -2, 4, 4, 4, 6],
                    [-2, 2, -2, 2, 2, 2, -2, 2, 2, 2, -2, 3, -2, -2, 8],
                    [5, 5, -2, -2, 5, -2, -2, 6, -2, 6, 10, 6, 6, -2, -1],
                    [7, -2, -2, 10, -2, -2, -2, 10, 10, 10, 11, 11, 11, -2, -2],
                    [12, 12, 12, 12, 13, 13, 13, -2, -2, 13, -2, 6, 14, -2, -2],
                ]
            ),
            "labels": tensor(
                [
                    [-3, 0, -3, -3, 1, -3, -3, -3, 1, 4, 4, -3, -3, -3, 4],
                    [2, -3, 2, -3, -3, -3, 2, -3, -3, -3, 3, 3, 3, 8, -3],
                    [-3, -3, 5, 5, 5, 5, 6, 6, 6, -3, 6, -3, -3, -3, -3],
                    [-3, 7, 7, -3, 10, 10, 10, -3, -3, -3, -3, 11, -3, 11, 18],
                    [-3, 12, -3, -3, -3, -3, -3, 13, 13, 13, 13, 13, -3, 14, 14],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 4, 9, 15], dtype=torch.int32),
                tensor([0, 10, 13, 15], dtype=torch.int32),
                tensor([0, 6, 13, 15], dtype=torch.int32),
                tensor([0, 3, 10, 14, 15], dtype=torch.int32),
                tensor([0, 4, 12, 15], dtype=torch.int32),
            ],
            "max_seqlen": [6, 10, 7, 7, 8],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        {
            "input_ids": tensor(
                [
                    [19, -2, 19, 19, 17, 0, 17, 17, 17, -2, 3, -2, 23, -2, -2],
                    [-2, 16, -2, 16, 16, -2, 16, -2, 15, 15, 15, -2, 15, 15, -1],
                    [20, 20, -2, -2, 20, 4, -2, 20, 20, -2, 21, -2, 21, -2, -1],
                    [22, 22, -2, 22, -2, -2, 22, -2, 25, 25, -2, 25, -2, 25, -2],
                    [24, -2, -2, 2, 24, 24, 24, 24, 24, 24, -2, -1, -1, -2, -2],
                ]
            ),
            "labels": tensor(
                [
                    [-3, 19, -3, -3, -3, 17, -3, -3, -3, 17, 17, 17, 23, 23, -3],
                    [16, -3, 16, -3, -3, 16, -3, 15, -3, 15, -3, 15, 15, 15, -3],
                    [-3, 20, 20, 20, -3, 20, 20, -3, -3, 20, -3, 21, -3, 21, -3],
                    [-3, -3, 22, -3, 22, 22, -3, 22, -3, -3, 25, -3, 25, -3, 25],
                    [-3, 24, 24, 24, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 4, 12, 14, 15], dtype=torch.int32),
                tensor([0, 7, 14, 15], dtype=torch.int32),
                tensor([0, 10, 14, 15], dtype=torch.int32),
                tensor([0, 8, 15], dtype=torch.int32),
                tensor([0, 10, 15], dtype=torch.int32),
            ],
            "max_seqlen": [8, 7, 10, 8, 10],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ]
            ),
        },
    ]

    compare_structures(out_expected, out_batches, message_prefix="test_packer_masking_randomly:")
    pass


def test_packer_masking_randomly_should_fail():
    input_batch_size = 4
    xds = [{"input_ids": seq} for seq in xs]
    input_batches = list(batched(xds, input_batch_size))
    d = {
        "src_iterable": input_batches,
        "src_batch_size": input_batch_size,
        "src_max_seq_len": max_seq_len,
        "out_batch_size": 5,
        "out_pseq_len": 15,
        "buffer_size": 5,
        "pad_token_id": -1,
        "mask_token_id": -2,
        "ignore_token_id": -3,
        "mask_prob": 0.5,
        "seed": 43,  ## <== seed set to 43
        "suppress_masking": False,
    }

    out_batches = list(GreedyBestFitSequencePacker(**d))

    # expected output for seed=42
    out_expected = [
        {
            "input_ids": tensor(
                [
                    [0, 4, 0, 0, -2, 1, 1, 1, -2, 4, -2, 4, 4, 4, 6],
                    [-2, 2, -2, 2, 2, 2, -2, 2, 2, 2, -2, 3, -2, -2, 8],
                    [5, 5, -2, -2, 5, -2, -2, 6, -2, 6, 10, 6, 6, -2, -1],
                    [7, -2, -2, 10, -2, -2, -2, 10, 10, 10, 11, 11, 11, -2, -2],
                    [12, 12, 12, 12, 13, 13, 13, -2, -2, 13, -2, 6, 14, -2, -2],
                ]
            ),
            "labels": tensor(
                [
                    [-3, 0, -3, -3, 1, -3, -3, -3, 1, 4, 4, -3, -3, -3, 4],
                    [2, -3, 2, -3, -3, -3, 2, -3, -3, -3, 3, 3, 3, 8, -3],
                    [-3, -3, 5, 5, 5, 5, 6, 6, 6, -3, 6, -3, -3, -3, -3],
                    [-3, 7, 7, -3, 10, 10, 10, -3, -3, -3, -3, 11, -3, 11, 18],
                    [-3, 12, -3, -3, -3, -3, -3, 13, 13, 13, 13, 13, -3, 14, 14],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 4, 9, 15], dtype=torch.int32),
                tensor([0, 10, 13, 15], dtype=torch.int32),
                tensor([0, 6, 13, 15], dtype=torch.int32),
                tensor([0, 3, 10, 14, 15], dtype=torch.int32),
                tensor([0, 4, 12, 15], dtype=torch.int32),
            ],
            "max_seqlen": [6, 10, 7, 7, 8],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        {
            "input_ids": tensor(
                [
                    [19, -2, 19, 19, 17, 0, 17, 17, 17, -2, 3, -2, 23, -2, -2],
                    [-2, 16, -2, 16, 16, -2, 16, -2, 15, 15, 15, -2, 15, 15, -1],
                    [20, 20, -2, -2, 20, 4, -2, 20, 20, -2, 21, -2, 21, -2, -1],
                    [22, 22, -2, 22, -2, -2, 22, -2, 25, 25, -2, 25, -2, 25, -2],
                    [24, -2, -2, 2, 24, 24, 24, 24, 24, 24, -2, -1, -1, -2, -2],
                ]
            ),
            "labels": tensor(
                [
                    [-3, 19, -3, -3, -3, 17, -3, -3, -3, 17, 17, 17, 23, 23, -3],
                    [16, -3, 16, -3, -3, 16, -3, 15, -3, 15, -3, 15, 15, 15, -3],
                    [-3, 20, 20, 20, -3, 20, 20, -3, -3, 20, -3, 21, -3, 21, -3],
                    [-3, -3, 22, -3, 22, 22, -3, 22, -3, -3, 25, -3, 25, -3, 25],
                    [-3, 24, 24, 24, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 4, 12, 14, 15], dtype=torch.int32),
                tensor([0, 7, 14, 15], dtype=torch.int32),
                tensor([0, 10, 14, 15], dtype=torch.int32),
                tensor([0, 8, 15], dtype=torch.int32),
                tensor([0, 10, 15], dtype=torch.int32),
            ],
            "max_seqlen": [8, 7, 10, 8, 10],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ]
            ),
        },
    ]
    # equality rule:
    #    attention mask == 0 where input_ids == pad_token_id

    failed_as_expected = False
    try:
        compare_structures(out_expected, out_batches, message_prefix="test_packer_masking_randomly_should_fail:")
    except AssertionError:
        failed_as_expected = True  # Comparison failed as expected

    assert failed_as_expected, (
        "test_packer_masking_randomly_should_fail: Comparison did NOT fail as expected (seed mismatch likely fixed or data changed)"
    )
    pass


def test_split_packed_batch():
    input_batches = [
        {
            "input_ids": tensor(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 8, 8],
                    [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, -1, -1],
                    [7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 18],
                ]
            ),
            "labels": tensor(
                [
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 10, 13, 15], dtype=torch.int32),
                tensor([0, 6, 13, 15], dtype=torch.int32),
                tensor([0, 3, 10, 14, 15], dtype=torch.int32),
            ],
            "max_seqlen": [10, 7, 7],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        {
            "input_ids": tensor(
                [
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, -1],
                    [22, 22, 22, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25, 25, 25],
                    [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, -1, -1, -1, -1, -1],
                ]
            ),
            "labels": tensor(
                [
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 10, 14, 15], dtype=torch.int32),
                tensor([0, 8, 15], dtype=torch.int32),
                tensor([0, 10, 15], dtype=torch.int32),
            ],
            "max_seqlen": [10, 8, 10],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ]
            ),
        },
    ]

    out_microbatches = [split_packed_batch(x, 3, padding_tolerance=1.0) for x in input_batches]

    # expected given a seed of 42!
    out_expected = [
        [
            {
                "input_ids": tensor(
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 8, 8],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 10, 13, 15], dtype=torch.int32),
                "max_seqlen": 10,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ),
            },
            {
                "input_ids": tensor(
                    [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, -1, -1],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 6, 13, 15], dtype=torch.int32),
                "max_seqlen": 7,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                ),
            },
            {
                "input_ids": tensor(
                    [7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 18],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 3, 10, 14, 15], dtype=torch.int32),
                "max_seqlen": 7,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ),
            },
        ],
        [
            {
                "input_ids": tensor(
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, -1],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 10, 14, 15], dtype=torch.int32),
                "max_seqlen": 10,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ),
            },
            {
                "input_ids": tensor(
                    [22, 22, 22, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25, 25, 25],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 8, 15], dtype=torch.int32),
                "max_seqlen": 8,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ),
            },
            {
                "input_ids": tensor(
                    [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, -1, -1, -1, -1, -1],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 10, 15], dtype=torch.int32),
                "max_seqlen": 10,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ),
            },
        ],
    ]

    compare_structures(out_expected, out_microbatches, message_prefix="test_split_packed_batch:")
    pass


def test_split_packed_batch_strip_padding():
    input_batches = [
        {
            "input_ids": tensor(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 8, 8],
                    [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, -1, -1],
                    [7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 18],
                ]
            ),
            "labels": tensor(
                [
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 10, 13, 15], dtype=torch.int32),
                tensor([0, 6, 13, 15], dtype=torch.int32),
                tensor([0, 3, 10, 14, 15], dtype=torch.int32),
            ],
            "max_seqlen": [10, 7, 7],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        {
            "input_ids": tensor(
                [
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, -1],
                    [22, 22, 22, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25, 25, 25],
                    [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, -1, -1, -1, -1, -1],
                ]
            ),
            "labels": tensor(
                [
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ]
            ),
            "cu_seqlens": [
                tensor([0, 10, 14, 15], dtype=torch.int32),
                tensor([0, 8, 15], dtype=torch.int32),
                tensor([0, 10, 15], dtype=torch.int32),
            ],
            "max_seqlen": [10, 8, 10],
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ]
            ),
        },
    ]

    out_microbatches = [split_packed_batch(x, 3, padding_tolerance=0.2) for x in input_batches]

    # expected given a seed of 42!
    out_expected = [
        [
            {
                "input_ids": tensor(
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 8, 8],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 10, 13, 15], dtype=torch.int32),
                "max_seqlen": 10,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ),
            },
            {
                "input_ids": tensor(
                    [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, -1, -1],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 6, 13, 15], dtype=torch.int32),
                "max_seqlen": 7,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                ),
            },
            {
                "input_ids": tensor(
                    [7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 18],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 3, 10, 14, 15], dtype=torch.int32),
                "max_seqlen": 7,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ),
            },
        ],
        [
            {
                "input_ids": tensor(
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, -1],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 10, 14, 15], dtype=torch.int32),
                "max_seqlen": 10,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ),
            },
            {
                "input_ids": tensor(
                    [22, 22, 22, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25, 25, 25],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 8, 15], dtype=torch.int32),
                "max_seqlen": 8,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ),
            },
            {
                "input_ids": tensor(
                    [24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
                ),
                "labels": tensor(
                    [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                ),
                "cu_seqlens": tensor([0, 10], dtype=torch.int32),
                "max_seqlen": 10,
                "attention_mask": tensor(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ),
            },
        ],
    ]

    compare_structures(out_expected, out_microbatches, message_prefix="test_split_packed_batch_strip_padding:")
    pass
