#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gRPC server using negative Forward-Forward updates.

Frames are streamed from the emulator and the policy weights are updated on
each step using only the negative phase of the reverse Forward-Forward
algorithm. No reward signal is used for learning.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List
import gc


import cv2
import numpy as np
import torch
import grpc

import inference_pb2
import inference_pb2_grpc
from model import GaussianGateAgent

# ── ハイパーパラメータ ────────────────────────────────
LR = 3e-4
FREEZE_STEPS = 2000  # steps to freeze log_sigma
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADDR = os.environ.get("MARIO_SERVER", "0.0.0.0:50051")

# --- discrete action list (B,0,SELECT,START,UP,DOWN,LEFT,RIGHT,A) ---
_ACTIONS_8 = [
    [0, 0, 0, 0, 0, 0, 0, 0],  # noop
    [0, 0, 0, 0, 0, 1, 0, 0],  # →
    [0, 0, 0, 1, 0, 1, 0, 0],  # → + B
    [1, 0, 0, 0, 0, 1, 0, 0],  # → + A
    [1, 0, 0, 1, 0, 1, 0, 0],  # → + A + B
    [0, 0, 1, 0, 0, 0, 0, 0],  # ←
    [0, 0, 1, 1, 0, 0, 0, 0],  # ← + B
    [1, 0, 1, 0, 0, 0, 0, 0],  # ← + A
    [1, 0, 1, 1, 0, 0, 0, 0],  # ← + A + B
    [1, 0, 0, 0, 0, 0, 0, 0],  # A only
    [0, 0, 0, 1, 0, 0, 0, 0],  # B only
    [0, 0, 0, 0, 0, 0, 1, 0],  # ↓
]


def _convert(a8: List[int]) -> List[int]:
    a, up, lf, b, st, ri, dn, se = a8
    return [b, 0, se, st, up, dn, lf, ri, a]


ACTIONS: List[List[int]] = [_convert(a) for a in _ACTIONS_8]
NUM_ACTIONS = len(ACTIONS)

# ── 1. 受信バイト列 → 256×256 RGB Tensor[3,H,W]0-1 ──────────
_CAND_WIDTHS = [256, 240, 224, 192, 160]


def decode_frame(buf: bytes) -> np.ndarray:
    a = np.frombuffer(buf, np.uint8)
    pixels = a.size // 3
    cand = [(w, pixels // w) for w in _CAND_WIDTHS if pixels % w == 0]
    cand.sort(key=lambda wh: (wh[1] % 8, abs((wh[0] / wh[1]) - 4 / 3)))
    w, h = cand[0] if cand else (256, pixels // 256)
    return a.reshape(h, w, 3)  # RGB


def to_tensor(buf: bytes) -> torch.Tensor:
    img = decode_frame(buf)  # H×W×3 (RGB)
    h, w = img.shape[:2]

    # pad to 256×256
    top, bottom = (256 - h) // 2, 256 - h - (256 - h) // 2
    left, right = (256 - w) // 2, 256 - w - (256 - w) // 2
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

    # Debug: Uncomment the following lines to inspect frames received
    # from the client. Useful for verifying gRPC transmission.
    # cv2.imshow("received", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)

    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # 0-1
    return t  # [3,256,256]


# ── 2. ポリシーネット & オプティマイザ ───────────────────────
policy = GaussianGateAgent(NUM_ACTIONS).to(DEVICE)

step_t = 0


# ── 4. gRPC Service ──────────────────────────────────────────
class Infer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, req, ctx):
        global step_t

        x = to_tensor(req.frame).to(DEVICE).unsqueeze(0)

        reset = 0.0 if req.is_dead else 1.0
        with torch.no_grad():
            logits, gate = policy(x, step_t, reset_mask=torch.tensor([reset], device=DEVICE))
        step_t += 1

        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()[0]
        if torch.rand(1).item() < 0.02:
            action_idx = torch.randint(NUM_ACTIONS, (1,), device=DEVICE)

        # update weights sequentially using negative FFA
        if step_t < FREEZE_STEPS:
            policy.indrnn.log_sigma.requires_grad_(False)
        else:
            policy.indrnn.log_sigma.requires_grad_(True)
        with torch.no_grad():
            policy.apply_negative_ffa(LR)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if req.is_dead:
            step_t = 0

        action = ACTIONS[action_idx.item()]
        return inference_pb2.InferenceResponse(action=action)


# ── 5. Bootstrap ────────────────────────────────────────────
def main():
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(Infer(), server)
    server.add_insecure_port(ADDR)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
