import torch
from torchmetrics import Metric

class MaskedTopKAccuracy(Metric):
    """Computes Top-K accuracy for masked language modeling with support for masked indices.
    
    Args:
        k (int): The K value for top-k accuracy. Default: 1.
        ignore_index (int): The class index to ignore. Default: -100.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, k: int = 1, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.k = k

        # Add state for this specific k value
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Updates the internal state with results from a new batch.

        Args:
            preds (torch.Tensor): Model predictions/logits of shape [batch_size, seq_len, vocab_size]
                                 or [batch_size * seq_len, vocab_size]
            target (torch.Tensor): Target labels of shape [batch_size, seq_len] or [batch_size * seq_len]
        """
        # Flatten if needed
        if preds.dim() == 3:
            preds = preds.view(-1, preds.size(-1))  # [batch*seq, vocab_size]
        if target.dim() == 2:
            target = target.view(-1)  # [batch*seq]

        assert preds.shape[0] == target.shape[0], f"Batch dimensions don't match: {preds.shape[0]} vs {target.shape[0]}"

        # Mask out the ignored indices (padding tokens, non-masked tokens)
        mask = (target != self.ignore_index)
        masked_target = target[mask]  # [num_valid_tokens]
        masked_preds = preds[mask]    # [num_valid_tokens, vocab_size]

        if masked_target.numel() == 0:
            return  # No valid tokens to process

        # Compute top-k accuracy for this specific k
        if self.k == 1:
            # Top-1 is just argmax
            top_k_preds = torch.argmax(masked_preds, dim=-1)
            correct = (top_k_preds == masked_target).sum()
        else:
            # Top-k predictions
            _, top_k_indices = torch.topk(masked_preds, k=min(self.k, masked_preds.size(-1)), dim=-1)
            # Check if target is in top-k predictions
            correct = torch.any(top_k_indices == masked_target.unsqueeze(1), dim=1).sum()
        
        # Update state
        assert isinstance(self.correct, torch.Tensor)
        self.correct += correct
        
        # Update total count
        assert isinstance(self.total, torch.Tensor)
        self.total += masked_target.numel()

    def compute(self):
        """Compute the final top-k accuracy.
        
        Returns:
            torch.Tensor: Single scalar tensor with the accuracy value.
        """
        assert isinstance(self.correct, torch.Tensor)
        assert isinstance(self.total, torch.Tensor)
        
        if self.total == 0:
            return torch.tensor(0.0)
        
        return self.correct.float() / self.total


class MaskedTop1Accuracy(MaskedTopKAccuracy):
    """Top-1 accuracy for masked language modeling."""
    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(k=1, ignore_index=ignore_index, dist_sync_on_step=dist_sync_on_step)


class MaskedTop5Accuracy(MaskedTopKAccuracy):
    """Top-5 accuracy for masked language modeling."""
    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(k=5, ignore_index=ignore_index, dist_sync_on_step=dist_sync_on_step)


class MaskedTop10Accuracy(MaskedTopKAccuracy):
    """Top-10 accuracy for masked language modeling."""
    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(k=10, ignore_index=ignore_index, dist_sync_on_step=dist_sync_on_step)


class MaskedTop25Accuracy(MaskedTopKAccuracy):
    """Top-25 accuracy for masked language modeling."""
    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(k=25, ignore_index=ignore_index, dist_sync_on_step=dist_sync_on_step)