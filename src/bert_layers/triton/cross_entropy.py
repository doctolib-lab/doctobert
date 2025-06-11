# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright (c) 2023, Tri Dao.
# License: Apache-2.0

# This simplified cross entropy implementation is modified from the original flash attention implementation
# to support torch.compile using the new PyTorch 2.4 custom_op. It doesn't support tensor parallel.

from typing import Tuple, Optional

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 2 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base


@triton.heuristics({"HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0})
@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,  # data ptrs
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignore_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    PRECOMPUTED_LSE: tl.constexpr,  # If LSE is already computed (also no smoothing and logit_scale == 1.0)
):
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    sum_logits = 0.0  # For smoothing
    if not PRECOMPUTED_LSE:
        # Statistics for online softmax
        m_i = -float("inf")
        l_i = 0.0
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            logits = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
            if HAS_SMOOTHING:
                sum_logits += tl.sum(tl.where(cols < n_cols, logits, 0.0))
            m_i_new = tl.maximum(m_i, tl.max(logits))
            l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(tl.exp(logits - m_i_new))
            m_i = m_i_new
        lse = tl.log(l_i) + m_i
        tl.store(lse_ptr + row_idx, lse)
    else:
        lse = tl.load(lse_ptr + row_idx)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= 0 and label_idx < n_cols:
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = lse - smoothing * sum_logits / total_classes - (1 - smoothing) * logits_label
            else:
                loss = lse - logits_label
        else:
            # If label is out of bounds, we set the CE loss to 0.0. But we still want the smoothing loss
            if HAS_SMOOTHING:
                loss = smoothing * (lse - sum_logits / total_classes)
            else:
                loss = 0.0
        z_loss = lse_square_scale * lse * lse
        loss += z_loss
    tl.store(loss_ptr + row_idx, loss)
    tl.store(z_loss_ptr + row_idx, z_loss)


@triton.heuristics({"HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0})
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignore_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = (
        tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
    )
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - smooth_positive, probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


@torch.library.custom_op("modernbert::cross_entropy_loss_fwd", mutates_args={}, device_types="cuda")
def cross_entropy_loss_fwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # For some reason Triton generates wrong code when labels has dtype long and its address
    # is not aligned to 16 bytes. The ld.global.b64 seems to load the wrong label index.
    if labels.dtype == torch.long and labels.data_ptr() % 16 != 0:
        labels = F.pad(labels, (0, 1))[..., :-1]
        assert labels.data_ptr() % 16 == 0
    assert logit_scale > 0.0
    n_rows, n_cols = logits.shape
    assert labels.shape == (n_rows,)

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    MAX_BLOCK_SIZE = 16 * 1024
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else (16 if BLOCK_SIZE < 128 * 1024 else 32))
    losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    lse = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(logits.device.index):
        cross_entropy_fwd_kernel[(n_rows,)](
            losses,  # data ptrs
            lse,
            z_losses,
            logits,
            labels,
            smoothing,
            logit_scale,
            lse_square_scale,
            ignore_index,
            n_cols,  # total_classes is always n_cols since this version doesn't support tensor parallel
            0,  # class_start_idx is always zero since this version doesn't support tensor parallel
            n_cols,  # shapes
            logits.stride(0),  # strides
            BLOCK_SIZE=BLOCK_SIZE,  # constants
            PRECOMPUTED_LSE=False,
            num_warps=num_warps,
        )

    return losses, z_losses, lse


@torch.library.register_fake("modernbert::cross_entropy_loss_fwd")
def _cross_entropy_loss_fwd_fake(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_rows, _ = logits.shape
    losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    lse = torch.empty(n_rows, dtype=torch.float, device=logits.device)

    return losses, z_losses, lse


@torch.library.custom_op("modernbert::cross_entropy_loss_bwd", mutates_args={}, device_types="cuda")
def cross_entropy_loss_bwd(
    grad_losses: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
    logit_scale: float,
    lse_square_scale: float,
    ignore_index: int,
) -> torch.Tensor:
    dlogits = torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
    grid = lambda META: (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))  # noqa
    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(logits.device.index):
        cross_entropy_bwd_kernel[grid](
            dlogits,  # data ptrs
            grad_losses,
            logits,
            lse,
            labels,
            smoothing,
            logit_scale,
            lse_square_scale,
            ignore_index,
            n_cols,  # total_classes is always n_cols since this version doesn't support tensor parallel
            0,  # class_start_idx is always zero since this version doesn't support tensor parallel
            n_cols,  # shapes
            logits.stride(0),  # strides
            dlogits.stride(0),
            grad_losses.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,  # constants
            num_warps=num_warps,
        )
    return dlogits


@torch.library.register_fake("modernbert::cross_entropy_loss_bwd")
def _cross_entropy_loss_bwd_fake(
    grad_losses: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
    logit_scale: float,
    lse_square_scale: float,
    ignore_index: int,
) -> torch.Tensor:
    return torch.empty_like(logits)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        smoothing: float = 0.0,
        logit_scale: float = 1.0,
        lse_square_scale: float = 0.0,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        losses, z_losses, lse = torch.ops.modernbert.cross_entropy_loss_fwd(
            logits,
            labels,
            smoothing,
            logit_scale,
            lse_square_scale,
            ignore_index,
        )
        ctx.save_for_backward(logits, lse, labels)
        ctx.mark_non_differentiable(z_losses)
        ctx.smoothing = smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignore_index = ignore_index
        return losses, z_losses

    @staticmethod
    def backward(
        ctx, grad_losses: torch.Tensor, grad_z_losses: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None, None, None, None, None]:
        del grad_z_losses  # z_losses are only for logging.

        logits, lse, labels = ctx.saved_tensors
        dlogits = torch.ops.modernbert.cross_entropy_loss_bwd(
            grad_losses,
            logits,
            lse,
            labels,
            ctx.smoothing,
            ctx.logit_scale,
            ctx.lse_square_scale,
            ctx.ignore_index,
        )

        return dlogits, None, None, None, None, None, None, None


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        logits: (batch, vocab_size)
        labels: (batch,)
        label_smoothing: float
        logit_scale: float. Multiply logits by this scale before calculating the loss.
        lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
            This is also referred to as "z-loss".
        ignore_index: int. If labels == ignore_index, the loss is set to 0.0.
    Returns:
        losses: (batch,), float
        z_losses: (batch,), float
    """
    return CrossEntropyLoss.apply(
        logits,
        labels,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        ignore_index,
    )
