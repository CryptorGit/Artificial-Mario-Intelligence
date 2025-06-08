"""Client configuration."""

# ---------------------------------------------------------------------------
# 報酬設計パラメータ
REWARD_WEIGHTS = {
    # 1px 前進で +0.1
    "delta_x_positive": 0.1,
    "delta_x_neutral": 0.0,
    # 停止で −0.5
    "delta_x_negative": -0.5,
    # スコア1点あたり +0.001
    "score": 0.001,
    # 死亡で −5.0
    "death": -5.0,
    # 1フレームあたり −0.01
    "time": -0.01,
}