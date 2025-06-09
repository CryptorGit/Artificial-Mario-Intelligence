import cv2
import grpc
import numpy as np
import retro
import inference_pb2
import inference_pb2_grpc


def get_maps(ram: np.ndarray):
    tile_map = np.reshape(ram[0x0500:0x0500 + 960], (30, 32))  # 960 bytes = 30x32
    sprite_ram = ram[0x0200:0x02F0]  # 予備：スプライト情報（仮）

    sprite_map = np.zeros((30, 32), dtype=np.uint8)
    for i in range(0, len(sprite_ram), 4):
        y = sprite_ram[i]
        x = sprite_ram[i + 3]
        if 0 <= y < 240 and 0 <= x < 256:
            sprite_map[y // 8, x // 8] = 255
    return tile_map.astype(np.uint8), sprite_map.astype(np.uint8)

def main():
    env = retro.make('SuperMarioBros-Nes')
    obs = env.reset()

    channel = grpc.insecure_channel('100.64.1.26:50051')
    stub = inference_pb2_grpc.InferenceStub(channel)

    while True:
        ram = env.get_ram()
        tile_map, sprite_map = get_maps(ram)

        req = inference_pb2.InferenceRequest(
            frame=obs.tobytes(),
            tile_map=tile_map.tobytes(),
            sprite_map=sprite_map.tobytes(),
            is_dead=False,
            reward=0.0,
        )
        res = stub.Predict(req)
        action = list(res.action)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

        # local display
        cv2.imshow('client', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == 27:
            break

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
