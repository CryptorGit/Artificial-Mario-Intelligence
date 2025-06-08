import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# ── Temporal-Shift (in-place) ───────────────────
def temporal_shift_(x, frames=4, n_div=8):
    n, c_tot, h, w = x.size()
    c = c_tot // frames
    fold = c // n_div
    x = x.view(n, frames, c, h, w)
    # forward shift
    x[:, 1:, :fold], x[:, 0, :fold] = x[:, :-1, :fold], 0
    # backward shift
    x[:, :-1, fold:2*fold], x[:, -1, fold:2*fold] = x[:, 1:, fold:2*fold], 0
    return x.view(n, c_tot, h, w)

class TSM(nn.Module):
    def __init__(self, block, frames=4):
        super().__init__()
        self.block, self.frames = block, frames
    def forward(self, x):
        return self.block(temporal_shift_(x, self.frames))

# ── MobileNetV2 + TSM (1-ch) ─────────────────────
class MarioGrayscaleMobileNetTSM(nn.Module):
    def __init__(self, frames=4):
        super().__init__()
        net = mobilenet_v2(weights=None)

        # conv0 in-channels: 1*frames
        old = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            1*frames, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None
        )

        # TSM 注入
        for i in range(1, 13):
            net.features[i] = TSM(net.features[i], frames)

        # ヘッド
        in_f = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, 8)          # 8-bit multi-label logits
        )
        self.net = net

    def forward(self, x):               # [B, 1*F, 256,256]
        return self.net(x)