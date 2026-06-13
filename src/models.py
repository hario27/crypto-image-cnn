"""マルチチャンネル市場画像のためのCNN回帰モデル。

容量を比較できるよう2種類用意している:

  * ``SmallCNNRegressor``(約0.1Mパラメータ) — 重複なし窓で典型的な、少なめの
    サンプル数に合わせたサイズ;
  * ``ResNet18Regressor`` — 標準的なバックボーン。最初のconvを任意チャンネル数を
    受け取れるよう作り替えてある。「あえて大きすぎる」ベースラインとして有用。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SmallCNNRegressor(nn.Module):
    def __init__(self, in_channels: int, base_ch: int = 32, dropout: float = 0.3, hidden: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 5, stride=2, padding=2),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(base_ch * 4, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


class ResNet18Regressor(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, hidden: int = 256):
        super().__init__()
        net = models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, hidden),
                               nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.net = net

    def forward(self, x):
        return self.net(x)


def build_model(name: str, in_channels: int, dropout: float = 0.3) -> nn.Module:
    if name == "small_cnn":
        return SmallCNNRegressor(in_channels, dropout=dropout)
    if name == "resnet18":
        return ResNet18Regressor(in_channels, dropout=dropout)
    raise ValueError(f"未知のモデル '{name}'")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
