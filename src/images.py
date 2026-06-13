"""時系列の窓 -> マルチチャンネル画像を **オンザフライ** で生成。

このリポジトリの研究テーマ: *市場のミクロ構造を2D画像として描画し CNN で学習すると、
同じ窓を平坦な特徴量ベクトルとして扱うより良くなるか?* 各チャンネルは窓の1列(または
数列)を決定論的に変換したものなので:

  * 画像化で **情報は増えない** — 表現が変わるだけ;
  * **リークは起きえない** — 各チャンネルは *その窓だけ* の関数(シーケンス版と同じ保証);
  * 画像は **ディスクに保存しない** — バッチ毎に ``__getitem__`` 内で生成する。これで
    リポジトリは小さく保たれ、素朴な実装が書き出す数十GBの事前生成画像も不要になる。

提供する変換:
  * ``auto_field``  : Gramian風の加算フィールド  G[i,j] = (x_i + x_j)/2
  * ``cross_field`` : 非対称な相互作用フィールド   G[i,j] = a_i * b_j
  * ``cwt``         : Morletウェーブレットのスカログラム(時間×周波数)
  * ``hist_map``    : 1秒毎の約定回数の生2Dマップ
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


def _upsample_int(a: np.ndarray, target: int, axis: int) -> np.ndarray:
    """整数倍 repeat で `target` まで拡大(補間なし → 情報を壊さない)。"""
    n = a.shape[axis]
    if n == target:
        return a
    a = np.repeat(a, max(1, target // n), axis=axis)
    pad = target - a.shape[axis]
    if pad > 0:
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(-1, None)
        a = np.concatenate([a, np.repeat(a[tuple(sl)], pad, axis=axis)], axis=axis)
    return a


def auto_field(x: np.ndarray) -> np.ndarray:
    return (x[:, None] + x[None, :]) * 0.5


def cross_field(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.outer(a, b)


def hist_map(block: np.ndarray, size: int) -> np.ndarray:
    return _upsample_int(block, size, axis=1).T


def cwt_image(x: np.ndarray, size: int) -> np.ndarray:
    import pywt
    scales = np.arange(2, 38)                       # 約180点の窓に対し cone of influence 内
    coef, _ = pywt.cwt(np.asarray(x, dtype=np.float64), scales, "morl")
    return _upsample_int(np.abs(coef), size, axis=0)


# 既定の6チャンネル: 4種類の変換(auto/cross/hist/cwt)を最小構成で示す。
# チャンネルを増やす(= 情報の入れ方を変える)実験は読者に委ねる。
def default_spec() -> list[dict]:
    p = "binance_"
    return [
        {"kind": "auto", "col": f"{p}log_return", "name": "gaf_log_return"},
        {"kind": "auto", "col": f"{p}volume_change", "name": "gaf_volume_change"},
        {"kind": "cross", "a": f"{p}imbalance", "b": f"{p}log_return", "name": "flow_x_ret"},
        {"kind": "hist", "cols": [f"{p}buy_count_change_1s_{k}" for k in range(1, 11)], "name": "buy_1s"},
        {"kind": "hist", "cols": [f"{p}sell_count_change_1s_{k}" for k in range(1, 11)], "name": "sell_1s"},
        {"kind": "cwt", "col": f"{p}log_return", "name": "cwt_log_return"},
    ]


def _resolve(spec: list[dict], columns: list[str]) -> list[dict]:
    pos = {c: i for i, c in enumerate(columns)}
    out = []
    for ch in spec:
        try:
            if ch["kind"] == "cross":
                out.append({**ch, "a": pos[ch["a"]], "b": pos[ch["b"]]})
            elif ch["kind"] == "hist":
                out.append({**ch, "cols": [pos[c] for c in ch["cols"]]})
            else:
                out.append({**ch, "col": pos[ch["col"]]})
        except KeyError as exc:
            warnings.warn(f"チャンネル '{ch['name']}' をスキップ(列が無い: {exc})")
    if not out:
        raise ValueError("解決できたチャンネルが0個 — 列名がデータと一致していない")
    return out


def build_image(seq: np.ndarray, resolved: list[dict], size: int) -> np.ndarray:
    """seq: (window, n_features) -> (channels, size, size) float32。"""
    seq = np.asarray(seq, dtype=np.float32)
    chans = []
    for ch in resolved:
        k = ch["kind"]
        if k == "auto":
            img = auto_field(seq[:, ch["col"]])
        elif k == "cross":
            img = cross_field(seq[:, ch["a"]], seq[:, ch["b"]])
        elif k == "hist":
            img = hist_map(seq[:, ch["cols"]], size)
        elif k == "cwt":
            img = cwt_image(seq[:, ch["col"]], size)
        chans.append(np.asarray(img, dtype=np.float32))
    return np.stack(chans, axis=0)


class ImageDataset(Dataset):
    """(軽量な)シーケンスだけを保持し、画像は要素取得時に遅延生成する。"""

    def __init__(self, sequences, columns, size, spec=None):
        self.sequences = sequences                # list[(np.ndarray(window, feat), float)]
        self.size = size
        self.spec_raw = spec if spec is not None else default_spec()
        self.spec = _resolve(self.spec_raw, list(columns))

    @property
    def channel_names(self):
        return [c["name"] for c in self.spec]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        seq, y = self.sequences[i]
        img = build_image(seq, self.spec, self.size)
        return torch.from_numpy(img), torch.tensor([float(y)], dtype=torch.float32)
