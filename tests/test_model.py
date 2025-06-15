import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from server.src.model import SinGateAgent

def main():
    agent = SinGateAgent(num_actions=12)
    obs = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.uint8)
    logits, gate, val, recon = agent(obs, step=0)
    assert logits.shape[0] == 4
    assert recon.shape[-2:] == (64, 64)

if __name__ == "__main__":
    main()
