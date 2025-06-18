#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for the Mario policy.

This module implements the Gaussian‑Gate agent using IndRNN cells together
with reverse Forward‑Forward updates. The convolutional encoder also employs
Gaussian‑gated self loops so that temporal differences directly modulate the
convolutional features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Signed log scaling used for robust value targets."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(y: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`."""
    return torch.sign(y) * torch.expm1(y.abs())


# ── Gaussian-Gate IndRNN components ───────────────────────────────

class GaussianGateIndRNNCell(nn.Module):
    """IndRNN cell with a difference-based Gaussian gate."""

    def __init__(self, d_embed: int, d_hidden: int, sigma: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self.U = nn.Linear(d_embed, d_hidden)
        self.w_base = nn.Parameter(torch.randn(d_hidden) * 0.1)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.eps = eps
        self.register_buffer("h_prev", torch.zeros(1, d_hidden), persistent=False)
        self.register_buffer("x_prev", torch.zeros(1, d_embed), persistent=False)

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
        dist = torch.norm(delta_norm, dim=1)
        sigma = torch.exp(self.log_sigma)
        gate = torch.exp(-(dist ** 2) / (2 * sigma * sigma))
        w_eff = gate.unsqueeze(1) * self.w_base
        h = F.relu(self.U(x) + w_eff * self.h_prev)
        self.h_prev = h
        self.x_prev = x.detach()
        return h, gate


class GaussianGateConvBlock(nn.Module):
    """Convolutional block with a Gaussian self-loop gate."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        sigma: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.w_base = nn.Parameter(torch.randn(out_ch, 1, 1, 1) * 0.1)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.eps = eps
        self.register_buffer("h_prev", torch.zeros(1, out_ch, 1, 1), persistent=False)
        self.register_buffer("x_prev", torch.zeros(1, in_ch, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor, reset_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.size()
        if self.h_prev.size(0) != B or self.h_prev.size(2) != H or self.h_prev.size(3) != W:
            self.h_prev = self.h_prev.new_zeros(B, self.h_prev.size(1), H, W)
            self.x_prev = self.x_prev.new_zeros(B, self.x_prev.size(1), H, W)

        if reset_mask is not None:
            mask = reset_mask.view(B, 1, 1, 1)
            self.h_prev = self.h_prev * mask
            self.x_prev = self.x_prev * mask

        delta = x - self.x_prev
        delta_norm = delta / (self.eps + delta.abs())
        dist = delta_norm.view(B, -1).norm(dim=1)
        sigma = torch.exp(self.log_sigma)
        gate = torch.exp(-(dist ** 2) / (2 * sigma * sigma))
        w_eff = gate.view(B, 1, 1, 1) * self.w_base
        h = F.relu(self.conv(x) + w_eff * self.h_prev)
        self.h_prev = h
        self.x_prev = x.detach()
        return h, gate


class Decoder(nn.Module):
    """Simple decoder to reconstruct a 64×64 RGB image from state."""

    def __init__(self, d_state: int, out_ch: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(d_state, 256 * 6 * 6)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 3, 1, 1),
            nn.Upsample(size=(64, 64), mode="bilinear"),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.fc(s).view(-1, 256, 6, 6)
        return self.net(x)



class GaussianGateAgent(nn.Module):
    """Agent using Gaussian gated IndRNN and rev-FF updates."""

    def __init__(self, num_actions: int, d_embed: int = 256, d_hidden: int = 512, d_state: int = 256):
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            GaussianGateConvBlock(3, 32, 4, stride=2, padding=1),
            GaussianGateConvBlock(32, 64, 4, stride=2, padding=1),
            GaussianGateConvBlock(64, 128, 4, stride=2, padding=1),
            GaussianGateConvBlock(128, d_embed, 4, stride=2, padding=1),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.indrnn = GaussianGateIndRNNCell(d_embed, d_hidden)
        self.state_mlp = nn.Sequential(
            nn.Linear(d_embed + d_hidden, d_state),
            nn.ReLU(),
            nn.Linear(d_state, d_state),
            nn.LayerNorm(d_state),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, num_actions),
        )
        self.decoder = Decoder(d_state)
        self.register_buffer("gate_ma", torch.tensor(0.5), persistent=False)
        self.last_loss_img = torch.tensor(0.0)

    def regularization_terms(self, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.exp(self.indrnn.log_sigma)
        loss_sigma = 1e-4 * (sigma ** 2)
        g_mean = gate.mean()
        loss_gate = 1e-2 * ((g_mean - 0.5) ** 2)
        return loss_sigma, loss_gate

    def world_model_loss(
        self, obs: torch.Tensor, recon: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        obs_ds = F.interpolate(obs, size=(64, 64), mode="bilinear", align_corners=False)
        loss_img = F.mse_loss(symlog(recon), symlog(obs_ds))
        self.last_loss_img = loss_img.detach()
        loss_s, loss_gate = self.regularization_terms(gate)
        return loss_img + loss_s + loss_gate

    def forward(
        self, obs: torch.Tensor, step: int, reset_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        emb = x
        for block in self.enc_blocks:
            emb, _ = block(emb, reset_mask)
        emb = self.flatten(self.pool(emb))
        if step == 0:
            B = emb.size(0)
            self.indrnn.h_prev = self.indrnn.h_prev.new_zeros(B, self.indrnn.h_prev.size(1))
            self.indrnn.x_prev = emb.detach()
        h, gate = self.indrnn(emb, reset_mask)
        self.gate_ma = 0.99 * self.gate_ma + 0.01 * gate.mean()
        s = self.state_mlp(torch.cat([emb, h], dim=1))
        logits = self.actor(s)
        recon = torch.clamp(self.decoder(s), 0.0, 1.0)
        return logits, gate, recon


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

