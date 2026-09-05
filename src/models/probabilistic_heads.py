"""Probabilistic heads §4.7 — μ,σ with heteroscedastic NLL, periodic az."""
import torch, torch.nn as nn
import math
def nll_gaussian(mu, logvar, target):
    return 0.5*(logvar + (target-mu)**2 / logvar.exp()).mean()
def az_wrap_loss(pred, target):
    d=(pred-target+180)%360-180
    return (d**2).mean()
