# test_buttons.py
import retro
import time
import numpy as np
import cv2

def test_all_buttons(game='SuperMarioBros-Nes', hold_sec=5, fps=60):
    # 環境生成
    env = retro.make(game=game)
    obs = env.reset()
    buttons = env.buttons  # ('B','A','SELECT','START','UP','DOWN','LEFT','RIGHT')
    n = len(buttons)

    # ウィンドウ準備
    cv2.namedWindow('Test', cv2.WINDOW_AUTOSIZE)

    try:
        for idx, name in enumerate(buttons):
            print(f"=== Testing button {idx}: {name} ===")
            # そのボタンだけを長押しするアクションベクトル
            action = np.zeros(n, dtype=np.int8)
            action[idx] = 1

            start = time.time()
            while time.time() - start < hold_sec:
                # step
                obs, _, done, _ = env.step(action.tolist())

                # 映像表示（RGB→BGR）
                frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                cv2.imshow('Test', frame)
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    return

                # もしゲームオーバー／クリアになったらリセット
                if done:
                    obs = env.reset()

            print(f"--- Finished {name} ---\n")
    finally:
        cv2.destroyAllWindows()
        env.close()

if __name__ == '__main__':
    test_all_buttons()
