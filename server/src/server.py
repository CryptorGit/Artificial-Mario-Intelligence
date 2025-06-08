#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — gRPC 推論 + REINFORCE 学習 (IndRNN 版, RGB 1 frame)
"""

import os, signal
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import cv2, numpy as np, torch, grpc

import inference_pb2, inference_pb2_grpc
from model import Img2ActInd      # ← 新モデル

# ── ハイパーパラメータ ────────────────────────────────
LR     = 1e-4
GAMMA  = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADDR   = os.environ.get("MARIO_SERVER", "0.0.0.0:50051")

# ── 1. 受信バイト列 → 256×256 RGB Tensor[3,H,W]0-1 ──────────
_CAND_WIDTHS = [256, 240, 224, 192, 160]

def decode_frame(buf: bytes) -> np.ndarray:
    a = np.frombuffer(buf, np.uint8)
    pixels = a.size // 3
    cand = [(w, pixels // w) for w in _CAND_WIDTHS if pixels % w == 0]
    cand.sort(key=lambda wh: (wh[1] % 8, abs((wh[0]/wh[1]) - 4/3)))
    w, h = cand[0] if cand else (256, pixels // 256)
    return a.reshape(h, w, 3)           # RGB

def to_tensor(buf: bytes) -> torch.Tensor:
    img = decode_frame(buf)             # H×W×3 (RGB)
    h, w = img.shape[:2]

    # pad to 256×256
    top, bottom = (256 - h)//2, 256 - h - (256 - h)//2
    left, right = (256 - w)//2, 256 - w - (256 - w)//2
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=0)

    # これは確認用
    # cv2.imshow("preview (RGB)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)

    t = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # 0-1
    return t                                                 # [3,256,256]

# ── 2. ポリシーネット & オプティマイザ ───────────────────────
policy    = Img2ActInd().to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

# gRPC peer ごとに hidden-state を保持
peer_state: Dict[str, list] = {}

# ── 3. REINFORCE バッファ ───────────────────────────────
eps_logps, eps_rewards = [], []

def finish_episode(peer: str):
    """エピソード終了時：勾配更新 & hidden state リセット"""
    if not eps_logps:
        return
    R, loss = 0.0, 0.0
    for logp, reward in reversed(list(zip(eps_logps, eps_rewards))):
        R = reward + GAMMA * R
        loss += -logp * R
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    eps_logps.clear()
    eps_rewards.clear()
    peer_state[peer] = None            # state リセット

# ── 4. gRPC Service ──────────────────────────────────────────
class Infer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, req, ctx):
        peer = ctx.peer()
        x = to_tensor(req.frame).to(DEVICE).unsqueeze(0)  # [1,3,256,256]

        logits, new_state = policy(x, peer_state.get(peer))
        peer_state[peer]  = new_state                     # 更新

        dist   = torch.distributions.Bernoulli(logits=logits)
        action = dist.sample()
        logp   = dist.log_prob(action).sum()

        eps_logps.append(logp)
        eps_rewards.append(req.reward)
        if req.is_dead:
            finish_episode(peer)

        a, up, lf, b, st, ri, dn, se = action.int()[0].tolist()
        return inference_pb2.InferenceResponse(
            action=[b, 0, se, st, up, dn, lf, ri, a]
        )

# ── 5. Bootstrap ────────────────────────────────────────────
def main():
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(Infer(), server)
    server.add_insecure_port(ADDR)
    server.start()
    print(f"IndRNN server running on {ADDR} (Ctrl-C to quit)")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
