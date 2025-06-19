# Artificial Mario Intelligence

This project contains a simple reinforcement learning setup for playing
**Super Mario Bros** using [Gym Retro](https://github.com/openai/retro). Frames
from the emulator are streamed via gRPC to a PyTorch server that returns the
next action. The server uses a lightweight convolutional recurrent policy and is
trained online using a negative Forward-Forward update on each step. This
sequential update scheme keeps memory usage small.

## Model details

The policy network is defined in `server/src/model.py`. Frames are flattened by
a small random projection and fed into a Gaussian‑gated ``IndRNN`` cell. The
gate value for unit *i* is
\[g_{t,i} = \exp\bigl(-((\tilde d_{t,i}-\mu_i)^2)/(2\sigma_i^2)\bigr),\]
where \(\tilde d_{t,i}\) is the normalised frame difference. The hidden state
is updated as
\[h_{t,i}=\mathrm{ReLU}\bigl(w_i g_{t,i} h_{t-1,i} + U_i^\top x_t\bigr).\]
Both gate parameters $\mu_i$ and $\sigma_i$ are trained from the start with no
initial freezing period.
Action logits come from an MLP. Online training uses the negative
Forward‑Forward rule with energy projection, updating weights by
\(\Delta W \propto h^{-}x^{-}\) while keeping their Frobenius norm constant.
The final actor layer is trained with the same local rule.

## Project structure

- `client/src` – client that runs the emulator and communicates with the server.
- `server/src` – gRPC server and neural network policy.
- `client/roms` – example ROM files (must be imported with `retro.import`).
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
   # remove any previously generated stubs
   rm client/src/inference_pb2*.py server/src/inference_pb2*.py

   # generate stubs from the server copy of inference.proto
   python -m grpc_tools.protoc \
       -I server/src \
       --python_out=server/src \
       --grpc_python_out=server/src \
       server/src/inference.proto

   # generate stubs from the client copy of inference.proto
   python -m grpc_tools.protoc ^
       -I client\src ^
       --python_out=client\src ^
       --grpc_python_out=client\src ^
       client\src\inference.proto
   ```

## Running

1. Start the server. The listening address can be changed via the
   `MARIO_SERVER` environment variable (default `0.0.0.0:50051`):
   ```bash
   python server/src/server.py
   ```
2. Set `MARIO_SERVER` to the server address (default
   `127.0.0.1:50051` when running locally) and start the client:
   ```bash
   MARIO_SERVER=127.0.0.1:50051 python client/src/client.py
   ```

The client window will display the game and send frames to the server, which
responds with actions predicted by the policy.

## Utilities

- `client/src/action_test.py` presses each controller button in turn for
  debugging.

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
