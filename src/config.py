"""パイプライン全体の設定。

調整可能なパラメータはすべてここに集約し、notebook 側は宣言的に保つ。
既定値は「そこそこのGPUで動かせる」ことを優先している(数ヶ月ぶんのデータ)。
本格的に検証する場合は `start` / `end` を伸ばすこと。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


@dataclass
class Config:
    # --- 市場 / 期間 ---
    symbol: str = "BTCUSDT"
    start: str = "2026-01-01"          # 開始日(UTC、この日を含む)
    end: str = "2026-04-01"            # 終了日(UTC、この日を含まない)
    bar_sec: int = 10                  # バーの秒数(生の約定から構築)

    # --- ターゲット ---
    horizon: int = 1                   # 何バー先のリターンを予測するか
    vol_window: int = 48               # ボラティリティ正規化のEWMA span(過去のみ)

    # --- 窓(画像の一辺 == 窓長) ---
    window: int = 180                  # 1サンプルのバー数(ここでは 180 * 10秒 = 30分)
    step: int = 30                     # 窓のスライド幅(重複させてサンプル数を稼ぐ)

    # --- 分割(時系列順、シャッフルなし) ---
    train_frac: float = 0.70
    val_frac: float = 0.15             # 残り → test

    # --- 学習 ---
    model: str = "small_cnn"           # "small_cnn" | "resnet18"
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 10                 # val損失でのearly stopping
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    seed: int = 42
    num_workers: int = 0               # 0 が最も移植性が高い(Docker等の共有メモリ問題を避ける)

    def __post_init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def image_size(self) -> int:
        """画像の一辺の長さ = 窓長(1バー → 1行/1列)。"""
        return self.window

    @property
    def tag(self) -> str:
        return f"{self.symbol}_{self.bar_sec}s_{self.start}_{self.end}"
