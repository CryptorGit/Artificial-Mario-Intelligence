#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for the Mario policy.

This module implements the Gaussian‑Gate agent using IndRNN cells together
with reverse Forward‑Forward updates. The convolutional encoder also employs
Gaussian‑gated self loops so that temporal differences directly modulate the
convolutional features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


D_RP = 4096  # output dimension of random projection
D_HIDD = [2048, 1024, 512]  # hidden sizes for stacked IndRNN layers


def _energy_project_update(param: torch.Tensor, grad: torch.Tensor, lr: float, eps: float = 1e-12) -> None:
    """Update parameter by grad while keeping its norm fixed."""
    dtheta = lr * grad
    num = -(param * dtheta).sum()
    den = dtheta.pow(2).sum() + eps
    alpha = num / den
    param.data += alpha * dtheta


# --- NEW: simple random-projection encoder ------------------------
class RandomProjectionEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, fix: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / out_dim ** 0.5)
        if fix:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.proj(x.reshape(B, -1))



# ── Gaussian-Gate IndRNN components ───────────────────────────────

class GaussianGateIndRNNCell(nn.Module):
    """IndRNN cell with a difference-based Gaussian gate."""

    def __init__(self, d_embed: int, d_hidden: int, sigma: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self.U = nn.Linear(d_embed, d_hidden)
        self.w_base = nn.Parameter(torch.randn(d_hidden) * 0.1)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.mu = nn.Parameter(torch.zeros(d_hidden))
        self.eps = eps
        self.register_buffer("h_prev", torch.zeros(1, d_hidden), persistent=False)
        self.register_buffer("x_prev", torch.zeros(1, d_embed), persistent=False)
        self._last_gate: Optional[torch.Tensor] = None
        self._last_h_prev: Optional[torch.Tensor] = None
        self._last_h: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None
        self._last_x: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, reset_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        if self.h_prev.size(0) != B:
            self.h_prev = self.h_prev.new_zeros(B, self.h_prev.size(1))
            self.x_prev = self.x_prev.new_zeros(B, self.x_prev.size(1))

        if reset_mask is not None:
            mask = reset_mask.view(B, 1)
            self.h_prev = self.h_prev * mask
            self.x_prev = self.x_prev * mask

        delta = x - self.x_prev
        delta_norm = delta / (self.eps + delta.abs())
        proj = F.linear(delta_norm, self.U.weight, bias=None)
        sigma = torch.exp(self.log_sigma)
        gate = torch.exp(-((proj - self.mu) ** 2) / (2 * sigma * sigma))
        w_eff = gate * self.w_base
        h = F.relu(self.U(x) + w_eff * self.h_prev)
        self._last_gate = gate.detach()
        self._last_h_prev = self.h_prev.detach()
        self._last_h = h.detach()
        self._last_delta_norm = proj.detach()
        self._last_x = x.detach()
        self.h_prev = h.detach()
        self.x_prev = x.detach()
        return h, gate

    def apply_negative_ffa_w_base(self, lr: float) -> None:
        if self._last_gate is None:
            return
        gate = self._last_gate
        grad = gate * self._last_h_prev * self._last_h
        grad = grad.mean(dim=0)
        _energy_project_update(self.w_base, grad, lr)
        if self.log_sigma.requires_grad:
            sigma = torch.exp(self.log_sigma)
            grad_sigma = -(
                self._last_h.pow(2)
                * (self._last_delta_norm.pow(2) / sigma)
            ).mean()
            _energy_project_update(self.log_sigma, grad_sigma, lr)
        if self.mu.requires_grad:
            sigma = torch.exp(self.log_sigma)
            grad_mu = (
                self._last_h.pow(2)
                * ((self._last_delta_norm - self.mu) / (sigma * sigma))
            ).mean(dim=0)
            _energy_project_update(self.mu, grad_mu, lr)

        zeros_h = torch.zeros_like(self._last_h)
        zeros_x = torch.zeros_like(self._last_x)
        reverse_ff_update(self.U, zeros_h, zeros_x, self._last_h, self._last_x, lr)





class GaussianGateAgent(nn.Module):
    """Agent with random projection encoder and stacked Gaussian-gated IndRNNs."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        IMG_FLAT = 3 * 256 * 256

        # random projection (frozen)
        self.encoder = RandomProjectionEncoder(IMG_FLAT, D_RP, fix=True)

        # stack of IndRNN layers with self loops
        self.rnn_layers = nn.ModuleList()
        in_dim = D_RP
        for out_dim in D_HIDD:
            self.rnn_layers.append(GaussianGateIndRNNCell(in_dim, out_dim))
            in_dim = out_dim

        # final linear actor (trainable)
        self.actor = nn.Linear(in_dim, num_actions, bias=True)

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # buffers for actor updates
        self._actor_in: Optional[torch.Tensor] = None
        self._actor_out: Optional[torch.Tensor] = None

    def apply_negative_ffa(self, lr: float) -> None:
        """Apply negative Forward-Forward update to all IndRNN layers."""
        for layer in self.rnn_layers:
            layer.apply_negative_ffa_w_base(lr)

        if self._actor_in is not None and self._actor_out is not None:
            zeros_h = torch.zeros_like(self._actor_out)
            zeros_x = torch.zeros_like(self._actor_in)
            reverse_ff_update(
                self.actor, zeros_h, zeros_x, self._actor_out, self._actor_in, lr
            )


    def forward(
        self, obs: torch.Tensor, reset_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.encoder(obs.float())
        for layer in self.rnn_layers:
            x, _ = layer(x, reset_mask)
        logits = self.actor(x)
        self._actor_in = x.detach()
        self._actor_out = logits.detach()
        return logits


def reverse_ff_update(
    layer: nn.Linear,
    h_pos: torch.Tensor,
    x_pos: torch.Tensor,
    h_neg: torch.Tensor,
    x_neg: torch.Tensor,
    lr: float,
    eps: float = 1e-12,
) -> None:
    """Apply reverse Forward-Forward update with energy preservation."""

    grad_w = h_pos.t() @ x_pos - h_neg.t() @ x_neg
    grad_b = h_pos.sum(0) - h_neg.sum(0)

    dW = lr * grad_w
    db = lr * grad_b

    num_w = -(layer.weight * dW).sum()
    den_w = dW.pow(2).sum() + eps
    alpha_w = num_w / den_w

    num_b = -(layer.bias * db).sum()
    den_b = db.pow(2).sum() + eps
    alpha_b = num_b / den_b

    layer.weight.data += alpha_w * dW
    layer.bias.data += alpha_b * db



