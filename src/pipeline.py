"""データ準備の一気通貫: バー → 特徴量 → ターゲット → スケール済みシーケンス。

重く決定論的な工程は ``data/`` にキャッシュし、notebook の再実行時は再計算せず
既存の成果物を再利用する。

ここで担保しているリーク対策:
  * 時系列順の分割を **スケーリングより前** に行う;
  * scaler は **train のみ** で fit する;
  * シーケンスは **各分割の内部** で構築し、train/val/test の境界をまたぐ窓を作らない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import Config, DATA_DIR
from .data import load_bars
from .features import compute_features
from .target import make_target


def build_features(cfg: Config) -> pd.DataFrame:
    """バー毎の全特徴量テーブル(+ ret, target)。parquet にキャッシュ。"""
    cache = DATA_DIR / f"features_{cfg.tag}.parquet"
    if cache.exists():
        print(f"  キャッシュ使用: {cache.name}")
        return pd.read_parquet(cache)

    bars = load_bars(cfg.symbol, cfg.start, cfg.end, cfg.bar_sec)
    feat = compute_features(bars, "binance_")
    ret, target = make_target(bars["Close"], cfg.horizon, cfg.vol_window)
    feat["ret"], feat["target"] = ret, target
    feat = feat.replace([np.inf, -np.inf], 0.0).dropna()
    feat.to_parquet(cache)
    print(f"  特徴量: {feat.shape[1] - 2} 列 x {len(feat):,} 行 -> キャッシュ {cache.name}")
    return feat


def _make_sequences(scaled: pd.DataFrame, target: pd.Series, window: int, step: int):
    """バー e で終わる窓 -> ラベル target[e](窓の直後のリターン)。"""
    vals = scaled.to_numpy(dtype=np.float32)
    tgt = target.to_numpy(dtype=np.float32)
    seqs = []
    for e in range(window - 1, len(scaled), step):
        y = tgt[e]
        if np.isfinite(y):
            seqs.append((vals[e - window + 1: e + 1], float(y)))
    return seqs


def prepare(cfg: Config) -> dict:
    """train/val/test のシーケンス、特徴量の列順、(描画/バックテスト用の)生の
    test フレームを返す。"""
    feat = build_features(cfg)
    feat_cols = [c for c in feat.columns if c not in ("ret", "target")]

    n = len(feat)
    n_tr = int(n * cfg.train_frac)
    n_va = int(n * cfg.val_frac)
    parts = {"train": feat.iloc[:n_tr],
             "val": feat.iloc[n_tr:n_tr + n_va],
             "test": feat.iloc[n_tr + n_va:]}

    scaler = RobustScaler().fit(parts["train"][feat_cols])   # train のみで fit
    seqs = {}
    for name, part in parts.items():
        scaled = pd.DataFrame(scaler.transform(part[feat_cols]),
                              columns=feat_cols, index=part.index)
        seqs[name] = _make_sequences(scaled, part["target"], cfg.window, cfg.step)
        print(f"  {name}: {len(seqs[name]):,} シーケンス")

    return {"sequences": seqs, "columns": feat_cols,
            "test_frame": parts["test"], "scaler": scaler}
