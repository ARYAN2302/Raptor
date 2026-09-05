import torch, torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

def az_wrap_diff(a,b):
    # periodic minimal difference in degrees, expects [-180,180] or [0,360]
    return (a - b + 180) % 360 - 180

class SetPredictionLoss(nn.Module):
    """Hungarian + existence + L1 range/az/el/velocity + NLL + identity per §12."""
    def __init__(self, w_exist=1.0, w_range=1.0, w_az=1.0, w_el=1.0, w_vel=1.0, w_nll=0.1, w_cls=0.5):
        super().__init__()
        self.w = dict(exist=w_exist, range=w_range, az=w_az, el=w_el, vel=w_vel, nll=w_nll, cls=w_cls)

    def forward(self, pred, gt):
        # pred: dict [B,K,...], gt: list per sample of emitter dicts with range/azimuth/elevation/velocity_xyz
        B,K = pred["existence"].shape
        device = pred["existence"].device
        total = 0.0
        for b in range(B):
            g = gt[b] if isinstance(gt, list) else []
            n = len(g)
            # Hungarian cost matrix [K, max(n,1)] but handle 0 case
            if n == 0:
                # only existence loss (all should be 0)
                tgt = torch.zeros(K, device=device)
                total += F.binary_cross_entropy(pred["existence"][b], tgt) * self.w["exist"]
                continue
            # Build cost [K, n]
            cost = torch.zeros(K, n, device=device)
            for k in range(K):
                for i, gi in enumerate(g):
                    c = 0
                    # existence: prefer 1 for matched
                    c += (1 - pred["existence"][b,k]) * self.w["exist"]
                    # range L1
                    c += abs(pred["range"][b,k].item() - gi.get("range",0)) * 0.001 * self.w["range"]
                    # az periodic
                    azd = abs(az_wrap_diff(pred["azimuth"][b,k].item(), gi.get("azimuth",0)))
                    c += azd * 0.01 * self.w["az"]
                    # el
                    eld = abs(pred["elevation"][b,k].item() - gi.get("elevation",0))
                    c += eld * 0.01 * self.w["el"]
                    # vel
                    if "velocity_xyz" in gi:
                        gv = torch.tensor(gi["velocity_xyz"], device=device)
                        c += (pred["velocity"][b,k] - gv).abs().mean().item() * self.w["vel"]
                    cost[k,i] = c
            # Hungarian (scipy expects cpu)
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            # Losses on matched pairs
            loss_b = 0
            # existence BCE (matched =1, unmatched=0)
            tgt = torch.zeros(K, device=device)
            tgt[row] = 1.0  # matched queries should be 1 (only first n matches, but some rows may be unmatched beyond n)
            # Actually Hungarian gives min(K,n) matches; for n<K, only n rows matched
            # Set only matched rows to 1
            tgt2 = torch.zeros(K, device=device)
            for r,c in zip(row,col):
                if c < n:
                    tgt2[r] = 1.0
            loss_b += F.binary_cross_entropy(pred["existence"][b], tgt2) * self.w["exist"]
            # regression on matched
            for r,c in zip(row,col):
                if c >= n: continue
                gi = g[c]
                # range L1
                loss_b += F.l1_loss(pred["range"][b,r], torch.tensor(gi.get("range",0), device=device, dtype=pred["range"].dtype)) * self.w["range"] * 0.001
                # az with wrap
                az_pred = pred["azimuth"][b,r]
                az_gt = torch.tensor(gi.get("azimuth",0), device=device, dtype=az_pred.dtype)
                # periodic L1 via wrap
                azd = torch.abs((az_pred - az_gt + 180) % 360 - 180)
                loss_b += azd * self.w["az"] * 0.01
                # el
                loss_b += F.l1_loss(pred["elevation"][b,r], torch.tensor(gi.get("elevation",0), device=device, dtype=pred["elevation"].dtype)) * self.w["el"] * 0.01
                # vel
                if "velocity_xyz" in gi:
                    gv = torch.tensor(gi["velocity_xyz"], device=device, dtype=pred["velocity"].dtype)
                    loss_b += F.l1_loss(pred["velocity"][b,r], gv) * self.w["vel"]
                # NLL if logvar present
                if "logvar" in pred:
                    # simple heteroscedastic on range
                    logv = pred["logvar"][b,r,0]
                    loss_b += 0.5 * (logv + (pred["range"][b,r] - gi.get("range",0))**2 / (logv.exp()+1e-6)) * self.w["nll"]
                # class/identity if present
                if "logits" in pred and "class" in gi:
                    loss_b += F.cross_entropy(pred["logits"][b,r].unsqueeze(0), torch.tensor([gi["class"]], device=device)) * self.w["cls"]
            total += loss_b
        return total / max(B,1)
