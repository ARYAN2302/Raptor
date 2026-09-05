import torch, torch.nn as nn
class SetPredictionLoss(nn.Module):
    def forward(self, pred, gt):
        B,_=pred["existence"].shape
        loss=0
        for b in range(B):
            n=len(gt[b]) if isinstance(gt,list) else 0
            tgt=torch.zeros_like(pred["existence"][b])
            tgt[:min(n, tgt.shape[0])]=1
            loss+= nn.functional.binary_cross_entropy(pred["existence"][b], tgt)
        return loss / max(B,1)
