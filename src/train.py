"""誠実でリークのない手順による学習と評価。

手順(金融MLで最も重要な部分):
  * **validation** で early stopping とモデル選択を行う;
  * **test は最後に1回だけ** 触る(選択済みモデルに対して)。

test で選択してはいけない。サンプルが短いと、チューニング対象にしたデータでは
偶然良く見えてしまう(選択バイアス / winner's curse)。最終測定まで test を
触らないことが、報告する数値を信頼できるものにする。
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .config import Config
from .images import ImageDataset
from .models import build_model, n_params


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_datasets(prepared: dict, cfg: Config):
    spec = None
    return {name: ImageDataset(seqs, prepared["columns"], cfg.image_size, spec)
            for name, seqs in prepared["sequences"].items()}


def _loader(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())


def train_model(datasets: dict, cfg: Config, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    in_ch = len(datasets["train"].channel_names)
    model = build_model(cfg.model, in_ch, cfg.dropout).to(device)
    if verbose:
        print(f"model={cfg.model}  channels={in_ch}  params={n_params(model):,}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.MSELoss()
    tr, va = _loader(datasets["train"], cfg, True), _loader(datasets["val"], cfg, False)

    best_state, best_val, bad = None, np.inf, 0
    history = {"train": [], "val": []}
    for epoch in range(cfg.max_epochs):
        model.train()
        tl = []
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                vl.append(crit(model(x), y).item())
        tr_l, va_l = float(np.mean(tl)), float(np.mean(vl))
        history["train"].append(tr_l)
        history["val"].append(va_l)

        if va_l < best_val - 1e-6:
            best_val, bad = va_l, 0
            best_state = copy.deepcopy(model.state_dict())   # deepcopy: 参照ではなく実体を保存
        else:
            bad += 1
        if verbose and (epoch % 5 == 0 or bad == 0):
            print(f"  epoch {epoch:3d}  train {tr_l:.4f}  val {va_l:.4f}  (best {best_val:.4f})")
        if bad >= cfg.patience:
            if verbose:
                print(f"  early stop @ epoch {epoch} (best val {best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model, dataset, cfg: Config, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, ys = [], []
    for x, y in _loader(dataset, cfg, False):
        preds.append(model(x.to(device)).cpu().numpy().ravel())
        ys.append(y.numpy().ravel())
    p, t = np.concatenate(preds), np.concatenate(ys)
    return {
        "mse": float(mean_squared_error(t, p)),
        "mae": float(mean_absolute_error(t, p)),
        "r2": float(r2_score(t, p)),
        "dir_acc": float(accuracy_score(t >= 0, p >= 0)),   # 方向一致率
        "pred": p, "true": t,
    }


def run(cfg: Config, prepared: dict, device=None):
    """学習(valで選択)した後、test を1回だけ評価する。"""
    datasets = make_datasets(prepared, cfg)
    model, history = train_model(datasets, cfg, device)
    val_metrics = evaluate(model, datasets["val"], cfg, device)
    test_metrics = evaluate(model, datasets["test"], cfg, device)
    print(f"\nVAL : R2 {val_metrics['r2']:+.4f}  方向一致率 {val_metrics['dir_acc']:.3f}")
    print(f"TEST: R2 {test_metrics['r2']:+.4f}  方向一致率 {test_metrics['dir_acc']:.3f}   (触れるのは1回だけ)")
    return {"model": model, "history": history,
            "val": val_metrics, "test": test_metrics, "datasets": datasets}
