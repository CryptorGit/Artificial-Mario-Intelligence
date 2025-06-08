WindowsPC
1. Python3.8のインストール

2. Gym Retro のインストール
pip install gym-retro

3. ROMの登録
python -m retro.import ./roms
Importing SuperMarioBros-Nes
Importing SuperMarioWorld2-Snes

4. 動作確認
python -m retro.examples.interactive --game SuperMarioBros-Nes

5. ネットワーク送受信用ライブラリの準備
pip install grpcio grpcio-tools
pip install requests

UbuntuPC
6. Ubuntuのファイアウォール設定
# UFW が無ければインストール
sudo apt update
sudo apt install ufw
# SSH（デフォルト22番）を許可
sudo ufw allow ssh
# gRPC 用の TCP 50051番を許可
sudo ufw allow 50051/tcp
# （もし HTTP REST を 8000番で使うなら）
sudo ufw allow 8000/tcp
# ファイアウォールを有効化
sudo ufw enable
# ステータス確認
sudo ufw status

WindowsPC & UbuntuPC
7. grpcio-toolsのインストール
pip install grpcio grpcio-tools

8. インポート確認
python -c "import grpc_tools.protoc; print('OK')"

9. プロトコル定義ファイルInference.protonoの作成

10. Pythonスタブ作成
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

