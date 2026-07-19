# -*- coding: utf-8 -*-
"""红外 / 长尾场景数据增强。

赛题要求"支持红外识别"且面对微光、夜间、遮挡等长尾场景。
由于公开跌倒数据以可见光为主，我们通过增强在训练期"合成"红外与恶劣光照样本，
提升跨域泛化，无需额外采集成本。这些增强既可作用于图像（第一阶段），
也可通过随机遮挡/丢弃关键点作用于骨骼序列（第二阶段）。
"""
from __future__ import annotations

import numpy as np


# ---------------- 图像域增强（可见光 -> 类红外 / 微光） ----------------

def to_infrared_like(img: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """把可见光 RGB 近似转换为红外热成像风格：转灰度 + 反相/伽马 + 单通道堆叠。"""
    if img.ndim == 3:
        gray = (0.299 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0])
    else:
        gray = img.astype(np.float32)
    gray = gray / 255.0
    gray = np.power(gray, gamma)            # 伽马压暗/提亮
    ir = (gray * 255).astype(np.uint8)
    return np.stack([ir, ir, ir], axis=-1)


def low_light(img: np.ndarray, factor: float = 0.35, noise_std: float = 8.0) -> np.ndarray:
    """模拟傍晚微光/夜间：整体压暗 + 传感器噪声。"""
    out = img.astype(np.float32) * factor
    out += np.random.randn(*out.shape) * noise_std
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------- 骨骼域增强（第二阶段时序模型） ----------------

def random_joint_occlusion(kp_seq: np.ndarray, max_joints: int = 4,
                           prob: float = 0.5) -> np.ndarray:
    """随机把若干关键点置信度清零，模拟遮挡/漏检，增强对缺失点的鲁棒性。
    kp_seq: (T, V, 3)。"""
    if np.random.rand() > prob:
        return kp_seq
    out = kp_seq.copy()
    v = out.shape[1]
    n = np.random.randint(1, max_joints + 1)
    joints = np.random.choice(v, n, replace=False)
    out[:, joints, 2] = 0.0
    out[:, joints, :2] = 0.0
    return out


def jitter(kp_seq: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """对归一化坐标加高斯抖动，模拟关键点定位噪声。"""
    out = kp_seq.copy()
    out[..., :2] += np.random.randn(*out[..., :2].shape).astype(np.float32) * sigma
    return out


def random_viewpoint_flip(kp_seq: np.ndarray, prob: float = 0.5,
                          swap_pairs=((5, 6), (7, 8), (9, 10),
                                      (11, 12), (13, 14), (15, 16),
                                      (1, 2), (3, 4))) -> np.ndarray:
    """水平镜像 + 左右关键点交换，模拟不同机位视角。"""
    if np.random.rand() > prob:
        return kp_seq
    out = kp_seq.copy()
    out[..., 0] = -out[..., 0]
    for a, b in swap_pairs:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def temporal_scale(kp_seq: np.ndarray, scale_range=(0.8, 1.2)) -> np.ndarray:
    """时间轴随机重采样，模拟跌倒快慢差异。返回同长序列（线性插值）。"""
    t = kp_seq.shape[0]
    scale = np.random.uniform(*scale_range)
    src = np.linspace(0, t - 1, num=max(2, int(t * scale)))
    src = np.clip(src, 0, t - 1)
    idx = np.linspace(0, len(src) - 1, num=t)
    resampled = np.stack([np.interp(idx, np.arange(len(src)),
                                    np.interp(src, np.arange(t), kp_seq[:, j, c]))
                          for j in range(kp_seq.shape[1]) for c in range(kp_seq.shape[2])], axis=1)
    return resampled.reshape(t, kp_seq.shape[1], kp_seq.shape[2]).astype(np.float32)


def augment_skeleton(kp_seq: np.ndarray) -> np.ndarray:
    """训练期骨骼序列增强组合。kp_seq: (T, V, 3)。"""
    kp_seq = random_viewpoint_flip(kp_seq)
    kp_seq = random_joint_occlusion(kp_seq)
    kp_seq = jitter(kp_seq)
    return kp_seq
