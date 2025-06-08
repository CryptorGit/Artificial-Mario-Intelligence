# config.py
# ─────────────────────────────────────────────
# 報酬設計パラメータ

REWARD_WEIGHTS = {
    'delta_x_positive': 0,   # 1px 前進で +0.1
    'delta_x_neutral': 0,
    'delta_x_negative': 0,  # 停止で −0.5
    'score':           0,  # スコア1点あたり +0.001
    'death':          0,    # 死亡で −5.0
    'time':           0,   # 1フレームあたり −0.01
}
