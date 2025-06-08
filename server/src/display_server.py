#!/usr/bin/env python3
"""
display_server.py  ― gRPC で受け取ったフレームを表示 & 解像度を出力するだけ
"""

import cv2
import numpy as np
import grpc
from concurrent.futures import ThreadPoolExecutor
import os
import signal

import inference_pb2
import inference_pb2_grpc

# ── ユーティリティ ──────────────────────────
def buf2img(buf: bytes) -> np.ndarray:
    """幅256固定・高さ自動の BGR 画像を復元"""
    a = np.frombuffer(buf, np.uint8)
    h = a.size // (256 * 3)
    return a.reshape(h, 256, 3)           # (H,256,3) BGR

# ── gRPC サービス ────────────────────────────
class Display(inference_pb2_grpc.InferenceServicer):
    def Predict(self, req, ctx):
        img = buf2img(req.frame)           # 画像を復元
        h, w = img.shape[:2]
        print(f"[{ctx.peer()}] frame {w}x{h}")

        cv2.imshow("mario (server preview)", img)
        if cv2.waitKey(1) == 27:           # ESC でサーバ終了
            print("ESC pressed – shutting down")
            os.kill(os.getpid(), signal.SIGINT)

        # gRPC 的にはレスポンスが必要なのでダミー返却
        return inference_pb2.InferenceResponse(action=[])

# ── main ─────────────────────────────────────
def main():
    addr = os.environ.get("MARIO_SERVER", "0.0.0.0:50051")
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InferenceServicer_to_server(Display(), server)
    server.add_insecure_port(addr)
    server.start()
    print(f"display_server running on {addr}  (ESC to quit)")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
