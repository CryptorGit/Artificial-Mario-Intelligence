#!/usr/bin/env python3
import cv2
import numpy as np
import torch
from collections import deque
import grpc
from concurrent.futures import ThreadPoolExecutor

import inference_pb2
import inference_pb2_grpc
from model import Img2Act

# --- constants ----------------------------------------------------
FRAMES = 4
LR = 1e-4
GAMMA = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

ADDR = os.environ.get("MARIO_SERVER", "0.0.0.0:50051")

# --- frame buffer -------------------------------------------------
def to_tensor(buf: bytes) -> torch.Tensor:
    a = np.frombuffer(buf, np.uint8)
    h = a.size // (256 * 3)
    img = a.reshape(h, 256, 3)
    img = cv2.copyMakeBorder(img, 8, 8, 0, 0, cv2.BORDER_CONSTANT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    return (t - 0.5) / 0.5

class FrameBuf:
    def __init__(self, k: int):
        self.k = k
        self.d = {}

    def cat(self, peer: str, buf: bytes) -> torch.Tensor:
        x = to_tensor(buf).to(DEVICE).unsqueeze(0)
        dq = self.d.setdefault(peer, deque([x] * self.k, maxlen=self.k))
        dq.append(x)
        return torch.cat(list(dq), 1)

BUF = FrameBuf(FRAMES)

# --- policy and optimizer ----------------------------------------
policy = Img2Act(FRAMES).to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

eps_logps = []
eps_rewards = []

def finish_episode() -> None:
    R = 0.0
    loss = 0.0
    for logp, reward in reversed(list(zip(eps_logps, eps_rewards))):
        R = reward + GAMMA * R
        loss += -logp * R
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    eps_logps.clear()
    eps_rewards.clear()

# --- inference service -------------------------------------------
class Infer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, req, ctx):
        x = BUF.cat(ctx.peer(), req.frame)
        dist = torch.distributions.Bernoulli(logits=policy(x))
        action = dist.sample()
        logp = dist.log_prob(action).sum()
        eps_logps.append(logp)
        eps_rewards.append(req.reward)
        if req.is_dead:
            finish_episode()
        a, up, lf, b, st, ri, dn, se = action.int()[0].tolist()
        return inference_pb2.InferenceResponse(action=[b, 0, se, st, up, dn, lf, ri, a])

# --- main ---------------------------------------------------------
if __name__ == "__main__":
    srv = grpc.server(ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(Infer(), srv)
    srv.add_insecure_port(ADDR)
    srv.start()
    print(f"gRPC running on {ADDR}")
    srv.wait_for_termination()
