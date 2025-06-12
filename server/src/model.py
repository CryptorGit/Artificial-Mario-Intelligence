#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for the Mario policy.

The previous version mixed recurrent convolutions with Bernoulli outputs. The
updated architecture uses a standard convolutional front-end followed by two
`IndLinear` blocks that carry the temporal state. The output layer has no
recurrent connection and simply produces logits for the categorical action
space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ── 自己ループ付き全結合 ────────────────────────────────
class IndLinear(nn.Module):
    """Fully connected layer with an independent recurrent connection."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=True)
        self.u = nn.Parameter(torch.zeros(out_f))

    def forward(self, x, h_prev: Optional[torch.Tensor]):
        y = self.fc(x)
        if h_prev is None:
            h_prev = torch.zeros_like(y)
        y = y + self.u * h_prev
        h = F.relu(y)
        return h

# ── 方策ネット本体 ──────────────────────────────────────
class Img2ActInd(nn.Module):
    """Convolutional encoder followed by two recurrent ``IndLinear`` layers."""

    def __init__(self, num_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc1 = IndLinear(256, 512)
        self.fc2 = IndLinear(512, 512)
        self.out = nn.Linear(512, num_actions)

    def forward(self, x, state: Optional[List[torch.Tensor]] = None):
        if state is None:
            state = [None, None]

        h = self.conv(x).flatten(1)
        h1 = self.fc1(h, state[0])
        h2 = self.fc2(h1, state[1])

        logits = self.out(h2)
        next_state = [h1, h2]
        return logits, next_state

