# display_probe_server.py ― 正しい解像度を決め打ち＋一覧表示

import cv2, numpy as np, grpc, os, signal
from concurrent.futures import ThreadPoolExecutor
import inference_pb2, inference_pb2_grpc

# NES 系で “あり得る” 幅リスト
POSSIBLE_WIDTHS = [256, 240, 224, 192, 160, 128]

def recover_image(buf: bytes):
    pixels = len(buf) // 3
    a = np.frombuffer(buf, np.uint8)

    # 1) ドメイン知識でフィルタ
    cand = [(w, pixels // w) for w in POSSIBLE_WIDTHS if pixels % w == 0]

    # 2) まだ複数あるときは『高さが 8 の倍数』優先
    cand.sort(key=lambda wh: (wh[1] % 8, abs((wh[0]/wh[1]) - 4/3)))
    w, h = cand[0] if cand else (pixels, 1)          # フォールバック

    rgb  = a.reshape(h, w, 3)
    bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, (w, h), cand

class Display(inference_pb2_grpc.InferenceServicer):
    def Predict(self, req, ctx):
        img, (w, h), cand = recover_image(req.frame)
        print(f"[{ctx.peer()}]  受信 {len(req.frame):,} byte  → 候補 {cand} → 採用 {w}×{h}")

        cv2.imshow("server preview", img)
        if cv2.waitKey(1) == 27:          # ESC
            os.kill(os.getpid(), signal.SIGINT)
        return inference_pb2.InferenceResponse(action=[])

# --- gRPC bootstrap ----------------------------------------------
def main():
    addr = os.environ.get("MARIO_SERVER", "0.0.0.0:50051")
    srv  = grpc.server(ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InferenceServicer_to_server(Display(), srv)
    srv.add_insecure_port(addr)
    srv.start(); print(f"probe server on {addr} (ESC で終了)")
    try: srv.wait_for_termination()
    except KeyboardInterrupt: pass
    finally: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
