# Copyright (c) 2024, Tri Dao.

import os
import sys
import pytest
import torch
import torch.nn.functional as F
from flash_attn.losses.cross_entropy import CrossEntropyLoss as CrossEntropyLossFA

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bert_layers.loss import CrossEntropyLossCompile

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.xdist_group(name="serial_group")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32] + ([torch.bfloat16] if is_sm8x else []))
@pytest.mark.parametrize("precompute_lse", [False, True])
@pytest.mark.parametrize("inplace_backward", [False, True])
@pytest.mark.parametrize("lse_square_scale", [0.0, 1e-2])
@pytest.mark.parametrize("return_z_loss", [False, True])
@pytest.mark.parametrize("logit_scale", [1.0, 0.7])
@pytest.mark.parametrize("smoothing", [0.0, 0.9])
@pytest.mark.parametrize("vocab_size", [50257])  # test vocab larger than 64k for split
def test_cross_entropy_loss(
    vocab_size,
    smoothing,
    logit_scale,
    lse_square_scale,
    return_z_loss,
    inplace_backward,
    precompute_lse,
    dtype,
):
    if precompute_lse and (logit_scale != 1.0 or smoothing != 0.0):
        pytest.skip("precompute_lse only works with logit_scale=1.0 and smoothing=0.0")
    device = "cuda"
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1 if dtype == torch.float32 else 4  # Otherwise OOM
    seqlen = 4096 if lse_square_scale == 0.0 and logit_scale == 1.0 else 1024  # Otherwise OOM
    x_pt = torch.randn(batch_size * seqlen, vocab_size, device=device, dtype=dtype, requires_grad=True)

    x_fa = x_pt.detach().clone().requires_grad_()
    y = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)

    if batch_size * seqlen > 10:
        y[torch.randperm(batch_size * seqlen)[:10]] = -100

    ce_pt = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    ce_fa = CrossEntropyLossFA(
        label_smoothing=smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        inplace_backward=inplace_backward,
    )
    if precompute_lse:
        with torch.no_grad():
            lse = torch.logsumexp(x_pt.float(), dim=-1)
    else:
        lse = None
    if return_z_loss:
        out_fa, out_fa_z_loss = ce_fa(x_fa, y, precomputed_lse=lse)
    else:
        out_fa = ce_fa(x_fa, y, precomputed_lse=lse)

    x_pt_scaled = (x_pt.float() * logit_scale) if logit_scale != 1.0 else x_pt.float()
    out_pt = ce_pt(x_pt_scaled, y)

    if lse_square_scale > 0.0:
        lse_pt = torch.logsumexp(x_pt_scaled, dim=-1)
        z_loss_pt = lse_square_scale * (lse_pt[y != -100] ** 2).mean()
        if return_z_loss:
            assert torch.allclose(out_fa_z_loss, z_loss_pt, rtol=rtol, atol=atol)
        out_pt += z_loss_pt

    assert torch.allclose(out_fa, out_pt, rtol=1e-5, atol=1e-6)

    g = torch.randn_like(out_fa)
    out_pt.backward(g)
    out_fa.backward(g)
    assert torch.allclose(x_fa.grad, x_pt.grad, rtol=rtol, atol=atol)


@pytest.mark.xdist_group(name="serial_group")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32] + ([torch.bfloat16] if is_sm8x else []))
@pytest.mark.parametrize("lse_square_scale", [0.0, 1e-2])
@pytest.mark.parametrize("return_z_loss", [False, True])
@pytest.mark.parametrize("logit_scale", [1.0, 0.7])
@pytest.mark.parametrize("smoothing", [0.0, 0.9])
@pytest.mark.parametrize("vocab_size", [50257, 128256])
def test_cross_entropy_loss_triton(
    vocab_size,
    smoothing,
    logit_scale,
    lse_square_scale,
    return_z_loss,
    dtype,
):
    device = "cuda"
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)

    torch.random.manual_seed(0)
    batch_size = 1 if dtype == torch.float32 else 4
    seqlen = 4096 if lse_square_scale == 0.0 and logit_scale == 1.0 else 1024
    seqlen = seqlen // 2 if vocab_size == 128256 else seqlen
    x_pt = torch.randn(batch_size * seqlen, vocab_size, device=device, dtype=dtype, requires_grad=True)

    # We'll compare Triton side by side with FA
    x_fa = x_pt.detach().clone().requires_grad_()
    x_triton = x_pt.detach().clone().requires_grad_()

    y = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    if batch_size * seqlen > 10:
        y[torch.randperm(batch_size * seqlen)[:10]] = -100

    ce_fa = CrossEntropyLossFA(
        label_smoothing=smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
    )
    ce_compiled = torch.compile(
        CrossEntropyLossCompile(
            label_smoothing=smoothing,
            logit_scale=logit_scale,
            lse_square_scale=lse_square_scale,
            return_z_loss=return_z_loss,
        )
    )

    if return_z_loss:
        out_fa, out_fa_z_loss = ce_fa(x_fa, y)
        out_compiled, out_compiled_z_loss = ce_compiled(x_triton, y)
        assert torch.allclose(out_fa_z_loss, out_compiled_z_loss, rtol=rtol, atol=atol)
    else:
        out_fa = ce_fa(x_fa, y)
        out_compiled = ce_compiled(x_triton, y)

    # Compare FA and Triton
    assert torch.allclose(out_fa, out_compiled, rtol=1e-5, atol=1e-6)

    # Test backward
    g = torch.randn_like(out_fa)
    out_fa.backward(g)
    out_compiled.backward(g)
    assert torch.allclose(x_fa.grad, x_triton.grad, rtol=rtol, atol=atol)
