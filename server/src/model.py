#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py  —  IndRNN-style CNN+MLP policy
  * 自己ループ y_t = Conv(x_t)+u⊙h_{t-1} (各ノード独立)
  * 活性化は ReLU
  * 全層に対して前ステップの出力を state として受け渡す
  * forward() は (x, prev_state) → (logits, next_state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ── 自己ループ付き畳み込み ────────────────────────────────
class IndConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=True)
        # 自己ループ重み (out_ch 個, 初期値 0)
        self.u = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, h_prev: Optional[torch.Tensor]):
        y = self.conv(x)
        if h_prev is None:                          # 1st 時刻
            h_prev = torch.zeros_like(y)
        y = y + self.u.view(1, -1, 1, 1) * h_prev   # IndRNN 項
        h = F.relu(y)
        return h

# ── 自己ループ付き全結合 ────────────────────────────────
class IndLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=True)
        self.u  = nn.Parameter(torch.zeros(out_f))

    def forward(self, x, h_prev: Optional[torch.Tensor]):
        y = self.fc(x)
        if h_prev is None:
            h_prev = torch.zeros_like(y)
        y = y + self.u * h_prev
        h = F.relu(y)
        return h

# ── 方策ネット本体 ──────────────────────────────────────
class Img2ActInd(nn.Module):
    """
    入力 :  [N, 3, 256, 256]  （単一フレーム RGB）
    状態 :  list[Tensor|None]  長さ 4  (conv1,conv2,conv3,fc1)
    出力 :  (logits[N,8], next_state[list])
    """
    def __init__(self):
        super().__init__()
        self.c1 = IndConv2d(3,  32, 8, 4, 2)
        self.c2 = IndConv2d(32, 64, 4, 2, 1)
        self.c3 = IndConv2d(64, 64, 3, 1, 1)

        self.f1 = IndLinear(64 * 32 * 32, 256)
        self.out= nn.Linear(256, 8)   # 出力層は自己ループなし

    def forward(self, x, state: Optional[List[torch.Tensor]] = None):
        if state is None:
            state = [None, None, None, None]

        h1 = self.c1(x, state[0])
        h2 = self.c2(h1, state[1])
        h3 = self.c3(h2, state[2])

        h3f = h3.flatten(1)
        h4  = self.f1(h3f, state[3])

        logits = self.out(h4)
        next_state = [h1.detach(), h2.detach(), h3.detach(), h4.detach()]
        return logits, next_state
