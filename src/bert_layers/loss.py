# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import inspect
import os
import sys
import torch
import torch.nn as nn
from .configuration_bert import FlexBertConfig

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as CrossEntropyLossFA
except ImportError:
    CrossEntropyLossFA = None

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bert_layers.triton.cross_entropy import cross_entropy_loss  # noqa: F401 (registers the custom op to pytorch)


# Copied from https://github.com/Dao-AILab/flash-attention
# Copyright (c) 2023, Tri Dao. Apache-2.0 license
class CrossEntropyLossCompile(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        process_group=None,
        return_z_loss=False,
    ):
        """
        Arguments:
            ignore_index: int. If labels == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
                This is also referred to as "z-loss".
            process_group: if not None, we're doing Tensor Parallel: each process is responsible for
                one part of the vocab. The loss will be aggregated across processes.
            return_z_loss: bool. If True, we return the component of the loss contributed by
                the lse_square_scale value. This value is only for logging and does not support
                backprop.
        """
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none' or 'sum'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.process_group = process_group
        self.return_z_loss = return_z_loss

    def forward(self, logits, labels):
        """
        Arguments:
            logits: (batch, vocab_size)
            labels: (batch,)
        Returns:
            losses: (batch,) if reduction is 'none', else (1,), dtype float
            z_loss: (batch,) if reduction is 'none', else (1,), dtype float (if self.return_z_loss)
        """
        assert logits.is_cuda and labels.is_cuda, "Only support CUDA tensors"
        loss, z_loss = cross_entropy_loss(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            lse_square_scale=self.lse_square_scale,
            ignore_index=self.ignore_index,
        )
        if self.reduction == "mean":
            loss = loss.sum() / (labels != self.ignore_index).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss

        if not self.return_z_loss:
            return loss

        if self.reduction == "mean":
            z_loss = z_loss.sum() / (labels != self.ignore_index).sum()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss

        return loss, z_loss


LOSS2CLS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "binary_cross_entropy": nn.BCEWithLogitsLoss,
    "mean_squared_error": nn.MSELoss,
    "compile_cross_entropy": CrossEntropyLossCompile,
}

if CrossEntropyLossFA is not None:
    LOSS2CLS["fa_cross_entropy"] = CrossEntropyLossFA


def get_loss_fn(config: FlexBertConfig) -> nn.Module:
    try:
        loss_class = LOSS2CLS[config.loss_function]
        if config.full_model_compile and config.loss_function == "fa_cross_entropy":
            loss_class = torch._dynamo.disable(CrossEntropyLossFA, recursive=True)
        signature = inspect.signature(loss_class)
        loss_kwargs = {k: v for k, v in config.loss_kwargs.items() if k in signature.parameters}
        return loss_class(**loss_kwargs)
    except KeyError:
        raise ValueError(f"Invalid loss function type: {config.loss_function}, must be one of {LOSS2CLS.keys()}.")
