import torch
import torch.nn as nn

class Img2Act(nn.Module):
    """Simple CNN+MLP policy for Img2Act."""
    def __init__(self, frames: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frames, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 8),  # 8 action bits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
