# -*- coding: utf-8 -*-
"""轻量级时空图卷积网络 (Lite ST-GCN)

用于对 YOLO-Pose 提取的人体骨骼关键点序列做时序动作判别（跌倒 / 非跌倒）。
设计目标：极小参数量（< 0.5M），可在端侧 NPU 上以微秒~毫秒级完成推理，
作为两阶段方案的第二阶段（时序判别头）。

输入张量形状:  (N, C, T, V)
    N: batch
    C: 每个关键点的通道数 (x, y, score) = 3
    T: 时间帧数 (滑动窗口长度, 默认 30)
    V: 关键点数 (COCO-17)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# COCO-17 骨架连接（父->子），用于构建图邻接矩阵
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 头部
    (5, 6),                                   # 双肩
    (5, 7), (7, 9), (6, 8), (8, 10),          # 双臂
    (5, 11), (6, 12), (11, 12),               # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),   # 双腿
]


def build_adjacency(num_nodes: int = 17, edges=COCO17_EDGES) -> torch.Tensor:
    """构建带自环并做对称归一化的邻接矩阵 A_hat = D^-1/2 (A+I) D^-1/2。"""
    A = torch.eye(num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)
    return D @ A @ D


class GraphConv(nn.Module):
    """空间图卷积：对同一帧内的关键点按骨架邻接聚合。"""

    def __init__(self, in_ch: int, out_ch: int, adj: torch.Tensor):
        super().__init__()
        # 邻接矩阵注册为 buffer（不训练，随模型保存/导出）
        self.register_buffer("adj", adj)
        self.fc = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        x = self.fc(x)
        # 在节点维做图聚合: (N,C,T,V) x (V,V) -> (N,C,T,V)
        x = torch.einsum("nctv,vw->nctw", x, self.adj)
        return x


class STGCNBlock(nn.Module):
    """ST-GCN 基本单元 = 空间图卷积 + 时间卷积 + 残差。"""

    def __init__(self, in_ch: int, out_ch: int, adj: torch.Tensor,
                 t_kernel: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gcn = GraphConv(in_ch, out_ch, adj)
        self.gcn_bn = nn.BatchNorm2d(out_ch)

        pad = (t_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(t_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout, inplace=True),
        )

        if in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.gcn_bn(self.gcn(x)))
        x = self.tcn(x)
        return self.relu(x + res)


class LiteSTGCN(nn.Module):
    """轻量 ST-GCN 主干 + 分类头。

    通道配置刻意做小 (3->32->64->96)，在保证判别力的同时把参数控制在 ~0.2M，
    远低于赛题 20M 上限，为第一阶段 YOLO-Pose 预留预算。
    """

    def __init__(self, num_class: int = 2, num_nodes: int = 17,
                 in_channels: int = 3, edges=COCO17_EDGES):
        super().__init__()
        adj = build_adjacency(num_nodes, edges)
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 32, adj, stride=1),
            STGCNBlock(32, 32, adj, stride=1),
            STGCNBlock(32, 64, adj, stride=2),   # 时间下采样
            STGCNBlock(64, 96, adj, stride=2),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        # 数据标准化（把节点特征展平做 BN，稳定不同人体尺度）
        x = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        x = self.data_bn(x)
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()

        for blk in self.blocks:
            x = blk(x)

        x = self.pool(x).view(n, -1)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = LiteSTGCN(num_class=2)
    dummy = torch.randn(4, 3, 30, 17)
    out = net(dummy)
    print("输出形状:", out.shape)
    print("参数量: %.3f M" % (count_parameters(net) / 1e6))
