import cv2
import grpc
import numpy as np
import retro
import inference_pb2
import inference_pb2_grpc


def get_maps(ram: np.ndarray):
    # NOTE: For real use these should decode tile/sprite info from RAM.
    # Here we create dummy arrays for demonstration.
    tile_map = np.reshape(ram[:960], (30, 32))
    sprite_map = np.reshape(ram[960:1920], (30, 32))
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
