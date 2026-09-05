"""Temporal state §4.5 — real streaming recurrence state_t = F(state_{t-1}, latent_t)."""
import torch, torch.nn as nn
class TemporalMamba(nn.Module):
    def __init__(self, d_model=64, hidden=64):
        super().__init__()
        self.d_model = d_model
        # GRU-like gating per latent position (shared across M)
        self.proj_z = nn.Linear(d_model, d_model)
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_gate = nn.Linear(d_model*2, d_model)
        self.proj_cand = nn.Linear(d_model*2, d_model)
        self.norm = nn.LayerNorm(d_model)
        # optional selective SSM via mamba if available (used as sequence mixer when training over T)
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(d_model)
            self.has_mamba = True
        except Exception:
            self.has_mamba = False
            self.mamba = None

    def forward(self, latents, state=None):
        # latents [B,M,D], state [B,M,D] or None
        if state is None:
            # first call: state = latents (init)
            state = torch.zeros_like(latents)
            # one step without history still updates
        # Gated recurrence per latent slot (batch M independent)
        # This is real state carry: output depends on state
        gate = torch.sigmoid(self.proj_gate(torch.cat([latents, state], dim=-1)))
        cand = torch.tanh(self.proj_cand(torch.cat([self.proj_z(latents), self.proj_s(state*gate)], dim=-1)))
        new_state = (1 - gate) * state + gate * cand
        out = self.norm(new_state)
        return out, new_state

    def forward_sequence(self, latents_seq, state=None):
        # latents_seq [B,T,M,D] -> outputs [B,T,M,D], final state [B,M,D]
        B,T,M,D = latents_seq.shape
        if state is None:
            state = torch.zeros(B,M,D, device=latents_seq.device, dtype=latents_seq.dtype)
        outs = []
        for t in range(T):
            out, state = self.forward(latents_seq[:,t], state)
            outs.append(out)
        return torch.stack(outs, dim=1), state

    def reset(self, batch_size, m, device, dtype=torch.float32):
        return torch.zeros(batch_size, m, self.d_model, device=device, dtype=dtype)
