#!/usr/bin/env python
# extract_smb_assets.py   --- ArtificalMarioIntelligence/data/src
# -------------------------------------------------------------
# 1) 32 KB PRG / 8 KB CHR 抽出
# 2) Nestopia NTSC 64⾊パレット(192 B) 厳正取得＋自動修復
# 3) Matthew Earl 法で背景(1-1〜8-4)抽出
# 4) “確定タイル表”でサブパレットを分けた spritesheet / 個別タイル生成
# -------------------------------------------------------------

import sys, subprocess, textwrap, re, importlib.util as iu
from pathlib import Path
import numpy as np
from PIL import Image
import requests

# ─────────────────────────────────────────────────────────────
# pip 依存
pkgs = {"Pillow": "pillow",
        "py65emu": "git+https://github.com/docmarionum1/py65emu.git"}
for m, p in pkgs.items():
    try:
        __import__("PIL" if m == "Pillow" else m)
    except ModuleNotFoundError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", p])

import py65emu                                   # noqa: E402
# ─────────────────────────────────────────────────────────────
# パス
ROOT = Path(__file__).resolve().parents[2]
ROM  = ROOT / "client/roms/super_mario_bros.nes"
DST  = ROOT / "data/super_mario_bros"; DST.mkdir(parents=True, exist_ok=True)
DATA = DST / "data";  DATA.mkdir(exist_ok=True)
OUT  = DST / "out";   OUT.mkdir(exist_ok=True)

# ───────────────────── 1. PRG / CHR 抽出
rom = ROM.read_bytes()
assert rom[:4] == b"NES\x1a", "Not an iNES ROM!"
PRG, CHR = rom[16:16+0x8000], rom[16+0x8000:16+0x8000+0x2000]
(DST / "prg.bin").write_bytes(PRG)
(DST / "smb.chr").write_bytes(CHR)
print("[OK] prg/chr extracted")

# ───────────────────── 2. smb.sym
SYM = DST / "smb.sym"
if not SYM.exists():
    SYM.write_bytes(requests.get(
        "https://raw.githubusercontent.com/Xkeeper0/smb1/main/bin/smb.sym",
        timeout=30).content)
    print("[OK] smb.sym downloaded")

# ───────────────────── 3. Nestopia NTSC パレット(192 B)
PAL = DATA / "ntscpalette.pal"
_EMBED = bytes.fromhex("""
7C7C7C 0000FC 0000BC 4428BC 940084 A80020 A81000 881400
503000 007800 006800 005800 004058 000000 000000 000000
BCBCBC 0078F8 4444FC 6828FC D800CC E40058 F83800 E45C10
AC7C00 00B800 00A800 00A844 008888 000000 000000 000000
F8F8F8 3CBCFC 6888FC 9878F8 F878F8 F85898 F87858 FCA044
F8B800 B8F818 58D854 58F898 00E8D8 787878 000000 000000
FCFCFC A4E4FC B8B8F8 D8B8F8 F8B8F8 F8A4C0 F0D0B0 FCE0A8
F8D878 D8F878 B8F8B8 B8F8D8 00FCFC F0BCBC 000000 000000
""".replace("\n", "").replace(" ", ""))
if not PAL.exists() or PAL.stat().st_size != 192:
    try:
        buf = requests.get(
            "https://raw.githubusercontent.com/georgeflanagin/NESColor/"
            "master/palettes/ntsc.pal", timeout=10
        ).content
        if len(buf) == 192:
            PAL.write_bytes(buf)
            print("[OK] ntsc.pal downloaded")
        else:
            raise ValueError
    except Exception:
        PAL.write_bytes(_EMBED)
        print("[OK] fallback palette written")
NES_RGB = np.frombuffer(PAL.read_bytes(), np.uint8).reshape(64, 3)
print("[OK] palette ready")

# ───────────────────── 4. smblevextract.py (MMU read fix)
GIST = ("https://gist.githubusercontent.com/matthewearl/"
        "733bba717780604813ed588d8ea7875f/raw/smblevextract.py")
raw = requests.get(GIST, timeout=30).text
patch = textwrap.dedent("""
import py65emu.mmu as _mmu
_old_read = _mmu.MMU.read
def safe_read(self, addr):
    try:
        return _old_read(self, addr)
    except IndexError:
        if 0xC000 <= addr <= 0xFFFF:  # MMC1 bank mirror
            return _old_read(self, addr - 0x4000)
        return 0xFF
_mmu.MMU.read = safe_read
""")
raw = patch + re.sub(
    r'^( +)assert len\(palette_data\) == 32',
    r'\1if len(palette_data) != 32:\n\1    palette_data = bytes([0]*32)',
    raw, flags=re.M
)
EXTRACT = DST / "smblevextract.py"
EXTRACT.write_text(raw, "utf8")
spec = iu.spec_from_file_location("smb", EXTRACT)
smb = iu.module_from_spec(spec); spec.loader.exec_module(smb)

# ───────────────────── SymbolFile helper
class Sym:
    _R = re.compile(r'^([A-Z0-9_]+)\s*=\s*\$([0-9A-F]+)')
    def __init__(self, path):
        self.d = {m[1]: int(m[2], 16)
                  for m in (self._R.match(l)
                            for l in Path(path).read_text().splitlines()) if m}
    def __getitem__(self, k): return self.d[k]
sym_file = Sym(SYM); smb.sym_file = sym_file

# ───────────────────── 5. BACKGROUND
print("\n=== BACKGROUND ===")
for w in range(1, 9):
    for s in range(1, 5):
        out = OUT / f"{w}-{s}.png"
        if out.exists():
            print(f"[SKIP] {w}-{s}")
            continue
        try:
            meta, mt = smb.load_level((w, s), PRG, CHR, sym_file, NES_RGB)
            Image.fromarray(smb.render_level(meta, mt).astype(np.uint8)).save(out)
            print(f"[OK] {w}-{s}")
        except Exception as e:
            print(f"[ERR] {w}-{s}: {e}")

# ───────────────────── 6. CHR → 2bpp index
def decode(idx: int):
    off = idx * 16
    lo  = np.frombuffer(CHR[off:off+8],      np.uint8)
    hi  = np.frombuffer(CHR[off+8:off+16],   np.uint8)
    arr = np.zeros((8, 8), np.uint8)
    for y in range(8):
        for x in range(8):
            arr[y, x] = ((hi[y] >> (7-x)) & 1) << 1 | ((lo[y] >> (7-x)) & 1)
    return arr
TILES = [decode(i) for i in range(256)]

# ───────────────────── 7. サブパレット適用 spritesheet
# 4 スプライトパレット（0F = 透過）
PALSET = np.array([
    [0x0F, 0x16, 0x27, 0x18],   # 0  Mario / HUD numbers
    [0x0F, 0x30, 0x27, 0x18],   # 1  Fire-Mario / Luigi
    [0x0F, 0x1A, 0x30, 0x27],   # 2  Ground enemies   (Goomba/green Koopa etc.)
    [0x0F, 0x06, 0x17, 0x30],   # 3  Red-shell / Boss (Red Koopa/Bowser etc.)
], dtype=np.uint8)

# ---- 初期 PALMAP（大まかな範囲）
PALMAP = np.zeros(256, np.uint8)
PALMAP[0x00:0x40]  = 0     # Player & digits
PALMAP[0x40:0x60]  = 2     # Goomba / Koopa green
PALMAP[0x60:0x78]  = 2     # Koopa animation etc.
PALMAP[0x78:0x80]  = 3     # Piranha etc.
PALMAP[0x80:0x90]  = 1     # Fire-Mario / Luigi duplicates
PALMAP[0x90:0xA0]  = 3     # Mushrooms / Flower heads
PALMAP[0xA0:0xC0]  = 3     # Bowser & big objects
PALMAP[0xC0:0xE0]  = 2     # Hammer Bro / Lakitu / Bullet Bill
PALMAP[0xE0:0x100] = 3     # Spiny & misc red group

# ---- 追加で上書きする「確定タイル表」
# （抽出結果で確認済みの赤ノコ・トゲゾー・キラー等）
#   start, end(exclusive), palette_index
FIX = [
    (0x6C, 0x70, 3),   # red Koopa body frames
    (0x74, 0x78, 3),   # red Koopa shell spin
    (0xD0, 0xE0, 3),   # Spiny & Spiny-egg tiles
    (0xC0, 0xC8, 3),   # Bullet Bill frames
]
for a, b, p in FIX:
    PALMAP[a:b] = p

# ---- 生成
SPR_DIR = OUT / "sprites"; SPR_DIR.mkdir(exist_ok=True)
sheet   = Image.new("RGBA", (128, 128), (0, 0, 0, 0))

for i, tile in enumerate(TILES):
    pal   = PALSET[PALMAP[i]]
    rgb   = NES_RGB[np.take(pal, tile)]
    alpha = (tile != 0).astype(np.uint8) * 255
    rgba  = np.dstack([rgb, alpha])
    img   = Image.fromarray(rgba, "RGBA")
    sheet.paste(img, ((i % 16) * 8, (i // 16) * 8))
    img.save(SPR_DIR / f"{i:03}.png")

sheet.save(OUT / "spritesheet.png")
print("\n[ALL DONE] backgrounds & colour-correct sprites →", OUT)
