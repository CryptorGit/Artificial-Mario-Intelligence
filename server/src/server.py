#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gRPC server implementing online REINFORCE training.

Frames are streamed from the emulator and the policy is updated online every
``TRUNCATE_STEPS`` steps. This keeps the computation graph bounded in size so
that GPU memory usage stays constant. The network uses a convolutional encoder
followed by two ``IndLinear`` layers and samples actions from a categorical
distribution over predefined valid combinations.
"""

import os
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import numpy as np
import torch
import grpc

import inference_pb2
import inference_pb2_grpc
from model import Img2ActInd

# ── ハイパーパラメータ ────────────────────────────────
LR = 3e-4
GAMMA = 0.99
TRUNCATE_STEPS = 64
ENTROPY_BETA = 0.01
BASELINE_DECAY = 0.99  # moving average coefficient
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

    # これは確認用
    # cv2.imshow("preview (RGB)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)

    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # 0-1
    return t  # [3,256,256]


# ── 2. ポリシーネット & オプティマイザ ───────────────────────
policy = Img2ActInd(NUM_ACTIONS).to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

# single-environment hidden state
state: List[torch.Tensor] | None = None
step_t = 0
baseline = 0.0

# ── 3. REINFORCE バッファ ───────────────────────────────
eps_logps: List[torch.Tensor] = []
eps_rewards: List[float] = []
eps_ents:    List[torch.Tensor] = []


def _update_buffer() -> tuple[float, float] | None:
    """Apply REINFORCE update using the current buffer."""
    if not eps_logps:
        return None

    returns: List[float] = []
    R = 0.0
    for r in reversed(eps_rewards):
        R = r + GAMMA * R
        returns.insert(0, R)

    global baseline
    returns_t = torch.tensor(returns, device=DEVICE)
    mean_ret = returns_t.mean().item()
    # ── moving average baseline ──────────────────────────────
    returns_t = returns_t - baseline
    baseline = baseline * BASELINE_DECAY + mean_ret * (1 - BASELINE_DECAY)
    # ── normalize returns for stability ─────────────────────
    ret_std = returns_t.std(unbiased=False)
    if ret_std < 1e-6:
        ret_std = 1.0
    returns_t = returns_t / ret_std

    policy_loss  = torch.stack([-logp * R for logp, R in zip(eps_logps, returns_t)]).sum()
    entropy_loss = torch.stack(eps_ents).sum()
    loss = policy_loss - ENTROPY_BETA * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()

    eps_logps.clear()
    eps_rewards.clear()
    eps_ents.clear()
    return mean_ret, float(grad)


def partial_update() -> None:
    """Update policy on the latest ``TRUNCATE_STEPS`` steps."""
    result = _update_buffer()
    if result is not None:
        mean_ret, grad = result
        #print(f"step_return_mean={mean_ret:.3f}, grad_norm={grad:.3f}")


def finish_episode() -> None:
    """Update remaining steps then reset the hidden state."""
    global state, step_t
    result = _update_buffer()
    if result is not None:
        mean_ret, grad = result
        print(f"step_return_mean={mean_ret:.3f}, grad_norm={grad:.3f}")
    state = None
    step_t = 0


# ── 4. gRPC Service ──────────────────────────────────────────
class Infer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, req, ctx):
        global state, step_t

        x = to_tensor(req.frame).to(DEVICE).unsqueeze(0)

        logits, new_state = policy(x, state)
        step_t += 1
        if step_t % TRUNCATE_STEPS == 0:
            new_state = [s.detach() for s in new_state]

        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()[0]
        # ─ ε-greedy で 2% ランダム行動 ──────────────────
        if torch.rand(1).item() < 0.02:
            action_idx = torch.randint(NUM_ACTIONS, (1,), device=DEVICE)
        logp = dist.log_prob(action_idx)
        ent = dist.entropy()          # バッチ1なので shape=[1]

        eps_logps.append(logp)
        eps_rewards.append(req.reward)
        eps_ents.append(ent)

        if len(eps_logps) >= TRUNCATE_STEPS:
            partial_update()
            new_state = [s.detach() for s in new_state]

        state = new_state
        if req.is_dead:
            finish_episode()

        action = ACTIONS[action_idx.item()]
        return inference_pb2.InferenceResponse(action=action)


# ── 5. Bootstrap ────────────────────────────────────────────
def main():
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(Infer(), server)
    server.add_insecure_port(ADDR)
    server.start()
    print(f"Mario server running on {ADDR} (Ctrl-C to quit)")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
