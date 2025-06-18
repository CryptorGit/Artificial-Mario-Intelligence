#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gRPC server implementing online REINFORCE training.

Frames are streamed from the emulator and the policy is updated online every
``TRUNCATE_STEPS`` steps. This keeps the computation graph bounded in size so
that GPU memory usage stays constant. The network uses convolutional blocks
with Gaussian self-loop gates followed by a Gaussian-Gated ``IndRNN`` block and
samples actions from a categorical distribution over predefined button
combinations.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np
import torch
import grpc

import inference_pb2
import inference_pb2_grpc
from model import GaussianGateAgent

# ── ハイパーパラメータ ────────────────────────────────
LR = 3e-4
GAMMA = 0.99
TRUNCATE_STEPS = 64
ENTROPY_BETA = 0.01
BASELINE_DECAY = 0.99  # moving average coefficient
FREEZE_STEPS = 2000  # steps to freeze log_sigma
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADDR = os.environ.get("MARIO_SERVER", "0.0.0.0:50051")
writer = SummaryWriter("runs/mario")

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
policy = GaussianGateAgent(NUM_ACTIONS).to(DEVICE)
wm_params = (
    list(policy.enc_blocks.parameters())
    + list(policy.indrnn.parameters())
    + list(policy.state_mlp.parameters())
    + list(policy.decoder.parameters())
)
actor_params = list(policy.actor.parameters())
opt_wm = torch.optim.Adam(
    [
        {"params": [p for p in wm_params if p is not policy.indrnn.log_sigma]},
        {"params": [policy.indrnn.log_sigma], "lr": 1e-4},
    ],
    lr=LR,
)
opt_policy = torch.optim.Adam(actor_params, lr=LR)

step_t = 0
baseline = 0.0

# ── 3. REINFORCE バッファ ───────────────────────────────
eps_logps: List[torch.Tensor] = []
eps_rewards: List[float] = []
eps_ents:    List[torch.Tensor] = []
eps_gates:   List[torch.Tensor] = []
eps_obs:     List[torch.Tensor] = []
eps_recon:   List[torch.Tensor] = []


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

    policy_loss = torch.stack([-logp * R for logp, R in zip(eps_logps, returns_t)]).sum()
    entropy_loss = torch.stack(eps_ents).sum()
    gate_batch = torch.cat(eps_gates) if eps_gates else torch.full((1, 1), 0.5, device=DEVICE)

    obs_batch = torch.cat(eps_obs)
    recon_batch = torch.cat(eps_recon)
    wm_loss = policy.world_model_loss(obs_batch, recon_batch, gate_batch)

    loss = policy_loss - ENTROPY_BETA * entropy_loss
    total = loss + wm_loss

    opt_policy.zero_grad()
    opt_wm.zero_grad()
    if step_t < FREEZE_STEPS:
        policy.indrnn.log_sigma.requires_grad_(False)
    else:
        policy.indrnn.log_sigma.requires_grad_(True)
    total.backward()
    grad_p = torch.nn.utils.clip_grad_norm_(actor_params, 0.3)
    grad_w = torch.nn.utils.clip_grad_norm_(wm_params, 0.3)
    opt_policy.step()
    opt_wm.step()
    policy.indrnn.h_prev = policy.indrnn.h_prev.detach()
    policy.indrnn.x_prev = policy.indrnn.x_prev.detach()

    writer.add_scalar("loss/policy", loss.item(), step_t)
    writer.add_scalar("loss/world_model", wm_loss.item(), step_t)
    writer.add_scalar("loss/img", policy.last_loss_img.item(), step_t)
    writer.add_scalar("entropy", entropy_loss.item(), step_t)
    writer.add_scalar("gate/mean", gate_batch.mean().item(), step_t)
    writer.add_scalar("sigma/value", torch.exp(policy.indrnn.log_sigma).item(), step_t)
    grad = float(max(grad_p, grad_w))

    eps_logps.clear()
    eps_rewards.clear()
    eps_ents.clear()
    eps_gates.clear()
    eps_obs.clear()
    eps_recon.clear()
    return mean_ret, float(grad)


def partial_update() -> None:
    """Update policy on the latest ``TRUNCATE_STEPS`` steps."""
    result = _update_buffer()
    if result is not None:
        mean_ret, grad = result
        #print(f"step_return_mean={mean_ret:.3f}, grad_norm={grad:.3f}")


def finish_episode() -> None:
    """Update remaining steps then reset counters."""
    global step_t
    result = _update_buffer()
    if result is not None:
        mean_ret, grad = result
        writer.add_scalar("return/mean", mean_ret, step_t)
        print(f"step_return_mean={mean_ret:.3f}, grad_norm={grad:.3f}")
    step_t = 0


# ── 4. gRPC Service ──────────────────────────────────────────
class Infer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, req, ctx):
        global step_t

        x = to_tensor(req.frame).to(DEVICE).unsqueeze(0)

        logits, gate, recon = policy(x, step_t)
        step_t += 1

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
        eps_gates.append(gate)
        eps_obs.append(x)
        eps_recon.append(recon)

        if len(eps_logps) >= TRUNCATE_STEPS:
            partial_update()
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
        writer.close()


if __name__ == "__main__":
    main()
