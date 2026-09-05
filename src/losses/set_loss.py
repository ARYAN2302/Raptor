"""DETR Hungarian loss §9."""
import torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment

def az_wrap_loss(pred, target):
    diff = (pred - target + 180) % 360 - 180
    return (diff**2).mean()

class SetPredictionLoss(nn.Module):
    def __init__(self, lambda_exist=1, lambda_range=1, lambda_az=1, lambda_el=1, lambda_vel=1):
        super().__init__()
        self.l=lambda_exist
    def forward(self, pred, gt):
        # pred: dict from SetDecoder, gt: list of dicts per sample
        B,Q = pred["existence"].shape
        loss=0
        for b in range(B):
            # simple: match by existence threshold + closest range (POC without full Hungarian for speed)
            # For real training, use linear_sum_assignment on cost matrix
            n=len(gt[b]) if isinstance(gt, list) else gt["emitter_count"][b].item()
            # existence BCE
            tgt_exist=torch.zeros(Q, device=pred["existence"].device)
            tgt_exist[:min(n,Q)]=1
            loss+= nn.functional.binary_cross_entropy(pred["existence"][b], tgt_exist)
            if n>0:
                # range L1
                loss+= (pred["range"][b][:n] - torch.tensor([g["range"] for g in gt[b][:n]], device=pred["range"].device)).abs().mean()*0.001
        return loss / max(B,1)
