"""予測ターゲット: ボラティリティ正規化した forward return。

    生リターン    r_t      = log(close_{t+h} / close_t)        # ラベル(未来を使う)
    ボラティリティ sigma_t  = EWMA_std( t までの過去リターン )   # 過去のみ(リークなし)
    ターゲット    y_t      = r_t / sigma_t

過去側のボラティリティで正規化することで、平穏な局面と荒れた局面でターゲットを
比較可能にする。ボラティリティは過去リターンのみを使うので t 時点で既知であり、
未来を覗くのは分子だけ(それが予測対象そのもの)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_target(close: pd.Series, horizon: int = 1, vol_window: int = 48):
    close = close.astype(float)
    raw = np.log(close.shift(-horizon) / close)             # forward return(ラベル)
    past_ret = np.log(close).diff()                         # 過去リターン、t 時点で既知
    sigma = past_ret.ewm(span=vol_window).std()
    target = raw / sigma
    return raw.rename("ret"), target.rename("target")
