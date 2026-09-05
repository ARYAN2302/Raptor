import torch, torch.nn as nn
class MaskedReconLoss(nn.Module):
    """Channel-independent masked MSE per Radio-FM Eq10: only masked patches."""
    def forward(self, pred, target, mask):
        # pred/target [B, L, D], mask [B, L] bool
        if pred.dim()==3:
            loss = ((pred - target)**2).mean(dim=-1)  # [B,L]
            if mask.any():
                return loss[mask].mean()
            return loss.mean()
        return ((pred - target)**2).mean()
