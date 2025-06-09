import cv2
import grpc
import numpy as np
import retro
import inference_pb2
import inference_pb2_grpc


def get_maps(ram: np.ndarray):
    """
    RAMデータからタイルマップとスプライトマップを抽出する。
    スーパーマリオブラザーズのRAMレイアウトに基づいて修正。
    """
    # 1. タイルマップの抽出
    # スーパーマリオブラザーズのRAMでは、0x0500から始まる416バイトに
    # 13x32タイルの形式で画面上部の地形データが格納されている。
    # これを読み込み、30x32の全体マップに配置する。
    level_map_bytes = 416
    level_map_ram = ram[0x0500:0x0500 + level_map_bytes]
    partial_tile_map = np.reshape(level_map_ram, (13, 32))

    # 30x32の空のマップを作成
    tile_map = np.zeros((30, 32), dtype=np.uint8)
    
    # NESの画面上部2タイルはステータスバーなので、2行目から地形データを配置
    tile_map[2:2+13, :] = partial_tile_map

    # 2. スプライトマップの抽出
    # スプライト情報はOAM($0200-$02FF)にあり、256バイト(64スプライト * 4バイト)。
    sprite_ram = ram[0x0200:0x0300]

    sprite_map = np.zeros((30, 32), dtype=np.uint8)
    for i in range(0, len(sprite_ram), 4):
        # スプライトデータ: [y, tile_id, attributes, x]
        y_pos = sprite_ram[i]
        x_pos = sprite_ram[i + 3]

        # y=240以上は画面外なので無視
        if y_pos < 240:
            # スプライトのY座標は画面表示位置-1。タイル座標に変換する。
            tile_y = (y_pos + 1) // 8
            tile_x = x_pos // 8
            if 0 <= tile_y < 30 and 0 <= tile_x < 32:
                # スプライトが存在する位置をマーク
                sprite_map[tile_y, tile_x] = 255
                
    return tile_map.astype(np.uint8), sprite_map.astype(np.uint8)

def main():
    env = retro.make('SuperMarioBros-Nes')
    obs = env.reset()

    channel = grpc.insecure_channel('100.64.1.26:50051')
    stub = inference_pb2_grpc.InferenceStub(channel)

    while True:
        # env.render() # retroのデフォルトレンダラを使いたい場合
        ram = env.get_ram()
        tile_map, sprite_map = get_maps(ram)

        req = inference_pb2.InferenceRequest(
            frame=obs.tobytes(),
            tile_map=tile_map.tobytes(),
            sprite_map=sprite_map.tobytes(),
            is_dead=False,
            reward=0.0,
        )
        try:
            res = stub.Predict(req)
            action = list(res.action)
        except grpc.RpcError as e:
            print(f"RPC failed: {e}")
            # エラー発生時はデフォルトのアクション（何もしない）などで継続
            action = [0] * 9 

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

        # ローカルでの画面表示
        # cv2.imshowはBGR形式を期待するため、RGBから変換する
        display_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow('client', display_img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()