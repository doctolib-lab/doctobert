# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright (c) 2023, Tri Dao.
# License: Apache-2.0

# This rotary implementation is modified from the original flash attention implementation to support torch.compile using the new PyTorch 2.4 custom_op.
# It also is a simplified version of the rotary implementation in flash attention, as it only supports GPT-NeoX style rotary embeddings for variable
# length sequences and it no longer supports seqlen_offset for kvcache as bidirectional transformers do not have a kvcache


from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    # Matrix dimensions
    seqlen,
    rotary_dim,
    seqlen_ro,
    # strides
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
    OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K // 2)

    # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
    X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
    COS = COS + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    cos = tl.load(COS, mask=(rm[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=(rm[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    x1 = tl.load(
        X + rotary_dim_half * stride_x_headdim,
        mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        other=0.0,
    ).to(tl.float32)
    if CONJUGATE:
        sin = -sin
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    # write back result
    OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
    tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    tl.store(
        OUT + rotary_dim_half * stride_out_headdim,
        o1,
        mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
    )


def apply_rotary(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
    conjugate: bool = False,
) -> None:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        raise ValueError("This kernel only supports variable length sequences")
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert cos.dtype == sin.dtype, f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert x.dtype == cos.dtype, f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    output = x

    BLOCK_K = 32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
    BLOCK_M = 8 if rotary_dim <= 64 else 4

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(rotary_kernel)[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen,  # shapes
            rotary_dim,
            seqlen_ro,
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            conjugate,
            BLOCK_M,
        )


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # (total_nnz, 3, nheads, headdim)
        total_nnz, three, nheads, headdim = qkv.shape
        assert three == 3
        if qkv.stride(-1) == 1:
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
            qk = qkv[:, :2].view(total_nnz, -1, headdim)
            apply_rotary(
                qk,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            q, k = qkv[:, 0, :, :], qkv[:, 1, :, :]
            apply_rotary(
                q,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            apply_rotary(
                k,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        if do.stride(-1) == 1:
            total_nnz, three, nheads, headdim = do.shape
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            dqk = do[:, :2].view(total_nnz, -1, headdim)
            apply_rotary(
                dqk,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                conjugate=True,
            )
        else:
            dq, dk = do[:, 0, :, :], do[:, 1, :, :]
            apply_rotary(
                dq,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                conjugate=True,
            )
            apply_rotary(
                dk,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                conjugate=True,
            )

        return do, None, None, None, None, None, None
