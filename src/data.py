"""市場データ取得(Binance、公開・認証不要)。

データは「生の約定」から一定秒数のバーに集計する。こうすることで、ミクロ構造
(バー内の1秒毎の約定回数ヒストグラム、taker の買い/売り内訳)を保持できる —
これらは kline / OHLCV エンドポイントでは失われてしまう情報である。

データ源(公開・認証不要):
  - Binance USDM無期限 aggTrades : https://data.binance.vision/

すべて ``data/*.parquet`` にキャッシュする。キャッシュが存在すればネットワーク
アクセスは行わない(notebook の再実行が安く、オフラインでも動く)。
"""
from __future__ import annotations

import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR

# OHLCV は features.compute_features に合わせて大文字始まり、カウント系は小文字。
BAR_COLS = (
    ["Open", "High", "Low", "Close", "Buy_volume", "Sell_volume", "buy_count", "sell_count"]
    + [f"buy_count_1s_{k}" for k in range(1, 11)]
    + [f"sell_count_1s_{k}" for k in range(1, 11)]
)


# --------------------------------------------------------------------------- #
# 約定 -> バー
# --------------------------------------------------------------------------- #
def build_bars(trades: pd.DataFrame, bar_sec: int) -> pd.DataFrame:
    """約定フレームをバーに集計する。

    `trades` の列: timestamp(tz-aware UTC), side('Buy'/'Sell'), price, amount。
    OHLC、taker の買い/売り出来高・回数、バー内の1秒毎の約定回数ヒストグラム
    (`*_count_1s_1..N`)を生成する。
    """
    t = trades.sort_values("timestamp")
    bar = t["timestamp"].dt.floor(f"{bar_sec}s")
    out = t.groupby(bar)["price"].agg(Open="first", High="max", Low="min", Close="last")

    is_buy = t["side"].eq("Buy")
    vol = t.groupby([bar, is_buy])["amount"].sum().unstack(fill_value=0.0)
    cnt = t.groupby([bar, is_buy]).size().unstack(fill_value=0)
    zero = pd.Series(0.0, index=out.index)
    out["Buy_volume"] = vol[True] if True in vol.columns else zero
    out["Sell_volume"] = vol[False] if False in vol.columns else zero
    out["buy_count"] = cnt[True] if True in cnt.columns else zero
    out["sell_count"] = cnt[False] if False in cnt.columns else zero

    sec = (t["timestamp"] - bar).dt.total_seconds().astype(int).clip(0, bar_sec - 1)
    hist = t.groupby([bar, is_buy, sec]).size().unstack(level=[1, 2], fill_value=0)
    hist = hist.reindex(columns=pd.MultiIndex.from_product([[True, False], range(bar_sec)]),
                        fill_value=0)
    for k in range(bar_sec):
        out[f"buy_count_1s_{k + 1}"] = hist[(True, k)]
        out[f"sell_count_1s_{k + 1}"] = hist[(False, k)]

    out.index.name = "timestamp"
    return out[BAR_COLS].astype("float64")


def _to_full_grid(bars: pd.DataFrame, start: str, end: str, bar_sec: int) -> pd.DataFrame:
    """欠損のないグリッドに整列。約定のないバーは close を ffill(O=H=L=C)、出来高=0。"""
    idx = pd.date_range(start, end, freq=f"{bar_sec}s", tz="UTC", inclusive="left", name="timestamp")
    df = bars.reindex(idx)
    df["Close"] = df["Close"].ffill()
    df = df[df["Close"].notna()]
    for c in ["Open", "High", "Low"]:
        df[c] = df[c].fillna(df["Close"])
    return df.fillna(0.0).astype("float64")


# --------------------------------------------------------------------------- #
# 日次の約定取得(Binance aggTrades)
# --------------------------------------------------------------------------- #
def _get(url: str, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  リトライ {attempt + 1} ({type(exc).__name__}) {url.split('/')[-1]}")
            time.sleep(8)


def fetch_trades_day(symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    url = (f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/"
           f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip")
    zf = zipfile.ZipFile(io.BytesIO(_get(url).content))
    name = zf.namelist()[0]
    first = zf.open(name).readline()
    header = 0 if not first.split(b",")[0].strip().lstrip(b"-").isdigit() else None
    cols = ["agg_id", "price", "qty", "first_id", "last_id", "transact_time", "is_buyer_maker"]
    raw = pd.read_csv(zf.open(name), header=header, names=cols,
                      usecols=["price", "qty", "transact_time", "is_buyer_maker"])
    ts = pd.to_datetime(raw["transact_time"], unit="ms", utc=True)
    # is_buyer_maker == True なら taker(成行)は売り手 → 'Sell'
    side = np.where(raw["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"]), "Sell", "Buy")
    return pd.DataFrame({"timestamp": ts, "side": side,
                         "price": raw["price"].astype(float), "amount": raw["qty"].astype(float)})


# --------------------------------------------------------------------------- #
# キャッシュ付きの高レベルローダー
# --------------------------------------------------------------------------- #
def load_bars(symbol: str, start: str, end: str, bar_sec: int) -> pd.DataFrame:
    """[start, end) の欠損なしバーを返す。キャッシュ済みなら再取得しない。"""
    cache = DATA_DIR / f"bars_binance_{symbol}_{bar_sec}s_{start}_{end}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    days = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(seconds=1), freq="D", tz="UTC")
    daily = []
    for i, d in enumerate(days):
        tr = fetch_trades_day(symbol, d)
        daily.append(build_bars(tr, bar_sec))
        if i % 10 == 0 or i == len(days) - 1:
            print(f"  [binance] {i + 1}/{len(days)} {d:%Y-%m-%d} trades={len(tr):,}")
    bars = pd.concat(daily).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    grid = _to_full_grid(bars, start, end, bar_sec)
    grid.to_parquet(cache)
    return grid
