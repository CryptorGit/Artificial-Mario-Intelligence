# Artificial Mario Intelligence

This project contains a simple reinforcement learning setup for playing
**Super Mario Bros** using [Gym Retro](https://github.com/openai/retro) and a
PyTorch policy served via gRPC.

## Project structure

- `client/src` – client that runs the emulator and communicates with the server.
- `server/src` – gRPC server and neural network policy.
- `client/roms` – example ROM files (must be imported with `retro.import`).

## Setup

1. Install Python 3.8 or newer.
2. Install dependencies:
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

## Running

1. Start the server:
   ```bash
   python server/src/server.py
   ```
2. In another terminal start the client (set `MARIO_SERVER` if the server is on
   another host):
   ```bash
   python client/src/client.py
   ```

The client window will display the game and send frames to the server, which
responds with actions predicted by the policy.

## License

This repository is provided for educational purposes only. ROM files are
included for convenience and may be subject to additional licensing terms.
