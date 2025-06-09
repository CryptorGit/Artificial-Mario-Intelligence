import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import grpc
from concurrent import futures

import inference_pb2
import inference_pb2_grpc

# --- simple U-Net for segmentation ----------------------------
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.out  = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        return self.out(d1)

# --- PPO policy model ----------------------------------------
class Policy(nn.Module):
    def __init__(self, in_ch=2, num_actions=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_pi = nn.Linear(64*32*30, num_actions)
        self.fc_v  = nn.Linear(64*32*30, 1)

    def forward(self, x):
        h = self.conv(x)
        return self.fc_pi(h), self.fc_v(h)

# --- gRPC service --------------------------------------------
class MapInfer(inference_pb2_grpc.InferenceServicer):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seg_model = UNet().to(self.device)
        self.policy = Policy().to(self.device)
        self.opt_seg = torch.optim.Adam(self.seg_model.parameters(), lr=1e-4)
        self.opt_pol = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

    def Predict(self, request, context):
        # decode frame
        img = np.frombuffer(request.frame, np.uint8).reshape(224, 240, 3)

        # decode tile map and sprite map from client RAM dump
        tile_map = np.frombuffer(request.tile_map, np.uint8).reshape(30, 32)
        sprite_map = np.frombuffer(request.sprite_map, np.uint8).reshape(30, 32)
        map_gt = np.stack([tile_map, sprite_map], 0)
        label_gt = np.argmax(map_gt, axis=0)

        # --- fix RGB/BGR for correct display ---
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- enhance tile map (middle panel) ---
        tile_vis = cv2.resize(tile_map * 8, (240, 224), interpolation=cv2.INTER_NEAREST)

        # --- enhance sprite map (right panel) ---
        sprite_vis = cv2.resize(sprite_map * 8, (240, 224), interpolation=cv2.INTER_NEAREST)

        # --- show all three images side-by-side ---
        disp = np.concatenate([
            img_bgr,
            cv2.cvtColor(tile_vis, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sprite_vis, cv2.COLOR_GRAY2BGR)
        ], axis=1)
        cv2.imshow('server', disp)
        cv2.waitKey(1)

        # --- segmentation supervised update ---
        x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(label_gt).long().unsqueeze(0)
        x, y = x.to(self.device), y.to(self.device)
        logits_full = self.seg_model(x)
        logits = F.interpolate(logits_full, size=(30, 32), mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, y)
        self.opt_seg.zero_grad(); loss.backward(); self.opt_seg.step()

        # --- return fixed "go right" action ---
        # Format: [b, 0, se, st, up, dn, lf, ri, a]
        action = [0, 0, 0, 0, 0, 0, 0, 1, 0]  # right only
        return inference_pb2.InferenceResponse(action=action)

# --- bootstrap ------------------------------------------------
def serve():
    addr = os.environ.get('MARIO_SERVER', '0.0.0.0:50051')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(MapInfer(), server)
    server.add_insecure_port(addr)
    server.start()
    print(f'server running on {addr}')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
