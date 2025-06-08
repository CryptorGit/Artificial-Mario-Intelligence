# client.py

import retro
import grpc
import cv2
import numpy as np
import config
import inference_pb2, inference_pb2_grpc

def extract_state_and_score(ram: np.ndarray):
    # 画面番号×256 + 画面内オフセット で絶対X位置
    screen      = int(ram[0x006D])
    on_screen_x = int(ram[0x0086])
    abs_x       = screen * 256 + on_screen_x

    # スコア（BCD 6桁 → 整数に変換）
    digits = [int(ram[0x07DD + i]) for i in range(6)]
    places = [10**(5 - i) for i in range(6)]
    score  = sum(d * p for d, p in zip(digits, places))

    # 死亡フラグ：RAMフラグ[0x000E]
    state = int(ram[0x000E])
    is_dead = (state in (0x06, 0x0B))

    return {
        'x':      abs_x,
        'score':  score,
        'dead':   is_dead,
    }

def compute_reward(prev: dict, curr: dict):
    w = config.REWARD_WEIGHTS

    # 前進距離
    dx = curr['x'] - prev['x']
    if dx > 0:
        r_dx = dx * w['delta_x_positive']
    elif dx < 0:
        r_dx = dx * w['delta_x_neutral']
    else:
        r_dx = w['delta_x_negative']

    # スコア増加
    dscore = curr['score'] - prev['score']
    r_score = dscore * w['score']

    # 死亡ペナルティ
    r_death = w['death'] if curr['dead'] else 0.0

    # 時間ペナルティ
    r_time = w['time']

    total = r_dx + r_score + r_death + r_time
    # デバッグ表示
    #print(
    #    f"[DBG] Δx={dx:4d}, "
    #    f"dead={curr['dead']}, reward={total:.3f}"
    #)
    return total

def main():
    env = retro.make('SuperMarioBros-Nes')
    obs = env.reset()

    # 初期RAM走査
    ram = env.get_ram()
    prev_state = extract_state_and_score(ram)

    action = [0] * 8
    channel = grpc.insecure_channel(config.SERVER_ADDRESS)
    stub = inference_pb2_grpc.InferenceStub(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except grpc.FutureTimeoutError:
        print(f"Could not connect to gRPC server at {config.SERVER_ADDRESS}."
              "\nMake sure the server is running and reachable.")
        return

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

        # ③ RAM → 状態＋スコア抽出 → 報酬計算
        ram        = env.get_ram()
        curr_state = extract_state_and_score(ram)
        reward     = compute_reward(prev_state, curr_state)
        prev_state = curr_state

        # ④ サーバへ送信（不要なフィールドは送らない）
        req = inference_pb2.InferenceRequest(
            frame   = obs.tobytes(),
            is_dead = curr_state['dead'],
            reward  = reward,
        )
        try:
            res = stub.Predict(req)
        except grpc.RpcError as e:
            print(f"gRPC error: {e}. Is the server still running?")
            break

        # ⑤ 新 action をセット
        action = list(res.action)

        # ⑥ エピソード終了時リセット
        if done or curr_state['dead']:
            obs = env.reset()
            ram = env.get_ram()
            prev_state = extract_state_and_score(ram)

    cv2.destroyAllWindows()
    env.close()

if __name__ == '__main__':
    main()
