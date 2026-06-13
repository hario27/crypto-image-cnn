# -*- coding: utf-8 -*-
"""CPU専用のスモークテスト: 合成データでコード全体を一気通貫で動かす。
ネットワーク・GPU不要。実行: python smoke_test.py"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.config import Config
from src.data import BAR_COLS
from src.features import compute_features
from src.target import make_target
from src.pipeline import _make_sequences
from src.train import make_datasets, train_model, evaluate

print("1) import OK")

# --- 合成バーを10秒グリッドで作る ---
N = 4000
idx = pd.date_range("2024-05-01", periods=N, freq="10s", tz="UTC", name="timestamp")
r = np.random.default_rng(1)
close = 60000 * np.exp(np.cumsum(r.normal(0, 3e-4, N)))
bars = pd.DataFrame(index=idx)
bars["Close"] = close
bars["Open"] = np.r_[close[0], close[:-1]]
bars["High"] = np.maximum(bars.Open, bars.Close) * (1 + r.uniform(0, 2e-4, N))
bars["Low"] = np.minimum(bars.Open, bars.Close) * (1 - r.uniform(0, 2e-4, N))
bars["Buy_volume"] = r.gamma(2, 2, N)
bars["Sell_volume"] = r.gamma(2, 2, N)
bars["buy_count"] = r.poisson(20, N).astype(float)
bars["sell_count"] = r.poisson(20, N).astype(float)
for k in range(1, 11):
    bars[f"buy_count_1s_{k}"] = r.poisson(2, N).astype(float)
    bars[f"sell_count_1s_{k}"] = r.poisson(2, N).astype(float)
bars = bars[BAR_COLS]
print(f"2) 合成バー OK  cols={list(bars.columns)[:6]}...")

# --- 特徴量 + target ---
feat = compute_features(bars, "binance_")
ret, target = make_target(bars["Close"], horizon=1, vol_window=48)
feat["ret"], feat["target"] = ret, target
feat = feat.replace([np.inf, -np.inf], 0).dropna()
print(f"3) 特徴量 OK  total={feat.shape[1]-2}")

# --- 因果性テスト: 未来を改変しても過去の特徴量が変わってはいけない ---
CUT = 3000
bars2 = bars.copy()
bars2.iloc[CUT:, bars2.columns.get_loc("Close")] *= 1.5
base = compute_features(bars, "binance_")
pert = compute_features(bars2, "binance_")
bad = [c for c in base.columns if not np.allclose(base[c].iloc[:CUT - 1], pert[c].iloc[:CUT - 1], atol=1e-9)]
print(f"4) 因果性テスト: 未来リークした特徴量 = {len(bad)} {'OK' if not bad else bad[:3]}")

# --- 分割 + スケール + シーケンス(pipeline.prepare を再現) ---
cfg = Config(window=180, step=60, batch_size=16, max_epochs=2, patience=2, num_workers=0)
feat_cols = [c for c in feat.columns if c not in ("ret", "target")]
n = len(feat); n_tr, n_va = int(n * 0.7), int(n * 0.15)
parts = {"train": feat.iloc[:n_tr], "val": feat.iloc[n_tr:n_tr + n_va], "test": feat.iloc[n_tr + n_va:]}
scaler = RobustScaler().fit(parts["train"][feat_cols])
seqs = {k: _make_sequences(pd.DataFrame(scaler.transform(v[feat_cols]), columns=feat_cols, index=v.index),
                           v["target"], cfg.window, cfg.step) for k, v in parts.items()}
print("5) シーケンス OK  " + " ".join(f"{k}={len(s)}" for k, s in seqs.items()))

# --- 画像 + モデル + 学習(CPU, 2エポック) ---
prepared = {"sequences": seqs, "columns": feat_cols}
datasets = make_datasets(prepared, cfg)
print(f"6) 画像データセット OK  channels={len(datasets['train'].channel_names)} "
      f"shape={tuple(datasets['train'][0][0].shape)}")
model, hist = train_model(datasets, cfg, device="cpu", verbose=False)
m = evaluate(model, datasets["test"], cfg, device="cpu")
print(f"7) 学習+評価 OK  test R2={m['r2']:+.4f} dir_acc={m['dir_acc']:.3f}")
print("\n全スモークテスト合格")
