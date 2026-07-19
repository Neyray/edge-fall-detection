# -*- coding: utf-8 -*-
"""轻量时序卷积网络 (TCN) —— ST-GCN 的备选时序判别头。

当端侧 NPU 对图卷积 (einsum/gather) 支持不佳时，可退化为纯 1D 因果卷积方案：
把 17 个关键点的 (x, y, score) 展平成 51 维特征，沿时间做膨胀因果卷积。
算子仅含 Conv1d / BN / ReLU，对海思、瑞芯微 NPU 极为友好。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """裁掉因果卷积右侧多余的 padding，保证严格因果（只看过去帧）。"""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.down(x))


class LiteTCN(nn.Module):
    """输入 (N, C, T, V) -> 展平为 (N, C*V, T) -> 因果 TCN -> 分类。"""

    def __init__(self, num_class: int = 2, num_nodes: int = 17,
                 in_channels: int = 3, channels=(48, 64, 64)):
        super().__init__()
        in_feat = in_channels * num_nodes
        layers = []
        prev = in_feat
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(prev, ch, kernel=3, dilation=2 ** i))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)  # (N, C*V, T)
        x = self.tcn(x)
        x = self.pool(x).view(n, -1)
        return self.fc(x)


if __name__ == "__main__":
    net = LiteTCN()
    out = net(torch.randn(2, 3, 30, 17))
    print("输出形状:", out.shape)
    print("参数量: %.3f M" % (sum(p.numel() for p in net.parameters()) / 1e6))
