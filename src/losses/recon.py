import torch, torch.nn as nn
class MaskedReconLoss(nn.Module):
    def forward(self, pred, target, mask):
        # pred/target: [B,N,D] or [B,D]
        if pred.dim()==2:
            return (pred**2).mean()
        loss=( (pred-target)**2 ).mean(dim=-1)  # [B,N]
        masked=loss[mask]
        return masked.mean() if masked.numel()>0 else loss.mean()
