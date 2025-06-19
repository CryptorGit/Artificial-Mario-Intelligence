# client.py

import retro
import grpc
import cv2
import numpy as np
import os
import inference_pb2
import inference_pb2_grpc

def is_dead_from_ram(ram: np.ndarray) -> bool:
    """Return ``True`` if the current RAM state represents Mario's death."""
    state = int(ram[0x000E])
    return state in (0x06, 0x0B)

def main():
    env = retro.make('SuperMarioBros-Nes')
    obs = env.reset()

    action = [0]*8
    addr = os.environ.get("MARIO_SERVER", "100.64.1.26:50051")
    channel = grpc.insecure_channel(addr)
    stub = inference_pb2_grpc.InferenceStub(channel)

    cv2.namedWindow('Game', cv2.WINDOW_AUTOSIZE)
    while True:
        # ① 画面表示
        frame_rgb = obs
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('Game', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ② 前回の action を反映
        obs, _, done, _ = env.step(action)

        # ③ RAM から死亡判定
        ram = env.get_ram()
        dead = is_dead_from_ram(ram)

        # ④ サーバへ送信
        req = inference_pb2.InferenceRequest(
            frame=obs.tobytes(),
            is_dead=dead,
        )
        try:
            res = stub.Predict(req)
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")
            print(f"Unable to reach server at {addr}. Is it running?")
            break

        # ⑤ 新 action をセット
        action = list(res.action)

        # ⑥ エピソード終了時リセット
        if done or dead:
            obs = env.reset()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
