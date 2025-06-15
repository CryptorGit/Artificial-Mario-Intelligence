#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for the Mario policy.

This module provides ``SinGateAgent``, a lightweight implementation of the
Sin-Gate IndRNN world-model. Frames are encoded by a small convolutional
network and fed into an ``IndRNN`` cell whose recurrent weight is modulated by
the change in consecutive embeddings. The resulting state is processed by
actor and critic MLPs.
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


# ── Sin-Gate IndRNN world-model variant ───────────────────────────
class SinGateIndRNNCell(nn.Module):
    """IndRNN cell with a sin gate on the recurrent weight."""

    def __init__(self, d_embed: int, d_hidden: int, init_k: float = 1.0):
        super().__init__()
        self.U = nn.Linear(d_embed, d_hidden)
        self.w_base = nn.Parameter(torch.randn(d_hidden) * 0.1)
        self.log_k = nn.Parameter(torch.log(torch.tensor(init_k)))
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

        delta_raw = torch.norm(x - self.x_prev, dim=1)
        k = torch.exp(self.log_k)
        hat_delta = torch.clamp(delta_raw / k, 0.0, 1.0)
        gate = torch.sin(math.pi * hat_delta)
        w_eff = gate.unsqueeze(1) * self.w_base
        h = F.relu(self.U(x) + w_eff * self.h_prev)
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


class SinGateAgent(nn.Module):
    """Lightweight agent using a Sin-Gate IndRNN block."""

    def __init__(self, num_actions: int, d_embed: int = 256, d_hidden: int = 512, d_state: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, d_embed, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.indrnn = SinGateIndRNNCell(d_embed, d_hidden)
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
        self.critic = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, 1),
        )
        self.decoder = Decoder(d_state)
        self.register_buffer("gate_ma", torch.tensor(0.5), persistent=False)

    def forward(
        self, obs: torch.Tensor, step: int, reset_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        emb = self.enc(x)
        if step == 0:
            B = emb.size(0)
            self.indrnn.h_prev = self.indrnn.h_prev.new_zeros(B, self.indrnn.h_prev.size(1))
            self.indrnn.x_prev = emb.detach()
        h, gate = self.indrnn(emb, reset_mask)
        self.gate_ma = 0.99 * self.gate_ma + 0.01 * gate.mean()
        s = self.state_mlp(torch.cat([emb, h], dim=1))
        logits = self.actor(s)
        value = self.critic(s)
        recon = torch.clamp(self.decoder(s), 0.0, 1.0)
        return logits, gate, value, recon

    def regularization_terms(self, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.exp(self.indrnn.log_k)
        loss_k = 1e-4 * (k ** 2)
        g_mean = gate.mean()
        loss_gate = 1e-2 * ((g_mean - 0.5) ** 2)
        return loss_k, loss_gate

    def world_model_loss(
        self, obs: torch.Tensor, recon: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        loss_img = F.mse_loss(recon, obs.float() / 255.0)
        loss_k, loss_gate = self.regularization_terms(gate)
        return loss_img + loss_k + loss_gate

