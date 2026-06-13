"""最小限の、**厳密に因果的**(過去のみ参照)な特徴量。

これは研究用スキャフォールドである。ここでは精度を狙った特徴量(テクニカル指標、
レジーム変数など)を **あえて入れていない** — 生に近い、ほぼ無加工の量だけを置く。
**エッジの探索(特徴量設計)は読者に委ねる。**

ここにある特徴量はすべて、バー t と t-1 だけから作る点演算 / 1階差分なので、
構造的にリークしえない。窓を使う特徴量を自分で足す場合は、中心化フィルタ
(``savgol_filter`` や ``np.convolve(mode="same")``)が未来を覗くことに注意し、
因果的なEWMA / 過去側のみの重み付き和に置き換えること。notebook の未来改変テストで
「未来を変えても過去の特徴量が変わらない」ことをいつでも確認できる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def compute_features(ohlcv: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """1取引所ぶんの最小特徴量ブロックを計算する。

    `ohlcv` には Open/High/Low/Close/Buy_volume/Sell_volume と、1秒毎のカウント列
    buy_count_1s_*/sell_count_1s_* が必要。
    """
    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    high = df["High"].ffill().astype(float)
    low = df["Low"].ffill().astype(float)
    close = df["Close"].ffill().astype(float)
    open_ = df["Open"].ffill().astype(float)
    buy_v = df["Buy_volume"].fillna(0).astype(float)
    sell_v = df["Sell_volume"].fillna(0).astype(float)
    vol = buy_v + sell_v
    logc = np.log(close)

    f = pd.DataFrame(index=df.index)
    # 価格 / バー形状(点演算・1階差分のみ → 自明に因果的)
    f[f"{prefix}log_return"] = logc.diff().fillna(0)
    f[f"{prefix}high_low_ratio"] = _safe_div((np.log(high) - np.log(low)).to_numpy(), np.log(low).to_numpy())
    f[f"{prefix}open_close_ratio"] = _safe_div((logc - np.log(open_)).to_numpy(), np.log(open_).to_numpy())
    # フロー(taker の買い/売り)
    f[f"{prefix}imbalance"] = pd.Series(_safe_div((buy_v - sell_v).to_numpy(), vol.to_numpy()), index=df.index).fillna(0)
    f[f"{prefix}volume_change"] = vol.pct_change().fillna(0).clip(-10, 10)
    # バー内1秒毎の約定回数の変化(生のミクロ構造)
    for i in range(1, 11):
        bc, sc = f"buy_count_1s_{i}", f"sell_count_1s_{i}"
        if bc in df.columns:
            f[f"{prefix}buy_count_change_1s_{i}"] = df[bc].fillna(0).diff().fillna(0)
            f[f"{prefix}sell_count_change_1s_{i}"] = df[sc].fillna(0).diff().fillna(0)

    return f.replace([np.inf, -np.inf], 0).clip(-10, 10).fillna(0)
