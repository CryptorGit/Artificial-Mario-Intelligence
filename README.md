# Artificial Mario Intelligence

This project contains a simple reinforcement learning setup for playing
**Super Mario Bros** using [Gym Retro](https://github.com/openai/retro). Frames
from the emulator are streamed via gRPC to a PyTorch server that returns the
next action. The server uses a lightweight convolutional recurrent policy and is
trained online with the REINFORCE algorithm. The policy is updated every
``32`` environment steps so memory usage stays constant regardless of episode
length.

## Model details

The policy network is defined in `server/src/model.py`. Frames are processed
by a convolutional encoder and passed through a Sin-Gate ``IndRNN`` block that
modulates the recurrent weight based on the change in input features. The
resulting state is fed to small actor and critic MLPs.
Training still uses the online REINFORCE algorithm
([Williams, 1992](https://doi.org/10.1007/BF00992696)) but now includes
regularization terms for the gating mechanism.

## Project structure

- `client/src` – client that runs the emulator and communicates with the server.
- `server/src` – gRPC server and neural network policy.
- `client/roms` – example ROM files (must be imported with `retro.import`).
- `server/src/display_server.py` – utility to inspect frames received over gRPC.
- `client/src/action_test.py` – cycles through all buttons for input testing.

## Setup

1. Install Python 3.8 or newer.
2. Install dependencies (requires **gRPC 1.71.0 or newer**):
   ```bash
   pip install -r requirements.txt
   ```
3. Import the ROMs for Gym Retro (once):
   ```bash
   python -m retro.import ./client/roms
   ```
4. (Optional) Regenerate gRPC code if `inference.proto` changes:
   ```bash
   python -m grpc_tools.protoc -I server/src --python_out=server/src \
       --grpc_python_out=server/src server/src/inference.proto
   python -m grpc_tools.protoc -I client/src --python_out=client/src \
       --grpc_python_out=client/src client/src/inference.proto
       ```
5. (Optional) Adjust reward weights in `client/src/config.py` to tune training.

## Running

1. Start the server. The listening address can be changed via the
   `MARIO_SERVER` environment variable (default `0.0.0.0:50051`):
   ```bash
   python server/src/server.py
   ```
2. Edit `client/src/client.py` to specify the server address in
   `grpc.insecure_channel` (default `100.64.1.26:50051`) and start the client:
   ```bash
   python client/src/client.py
   ```

The client window will display the game and send frames to the server, which
responds with actions predicted by the policy.

## Utilities

- `server/src/display_server.py` can be run to check the resolution of frames
  arriving over gRPC.
- `client/src/action_test.py` presses each controller button in turn for
  debugging.
- Reward calculation parameters are located in `client/src/config.py`.

## Troubleshooting

If the client exits with a gRPC connection error such as `UNAVAILABLE: failed
to connect to all addresses`, ensure that the server is running and reachable
at the address configured for the client. The server listening address is
controlled by the `MARIO_SERVER` environment variable. Start the server using:

```bash
python server/src/server.py
```

## License

This repository is provided for educational purposes only. ROM files are
included for convenience and may be subject to additional licensing terms.
