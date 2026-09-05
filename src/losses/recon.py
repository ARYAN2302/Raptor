import torch, torch.nn as nn
class MaskedReconLoss(nn.Module):
    def forward(self, pred, target, mask):
        if pred.dim()==3:
            loss=((pred-target)**2).mean(dim=-1)
            m=loss[mask]
            return m.mean() if m.numel()>0 else loss.mean()
        return ((pred-target)**2).mean()
