# -*- coding: utf-8 -*-
"""时序跌倒分类器使用的骨骼序列数据集。

数据集读取 ``scripts/extract_skeletons.py`` 生成的压缩 ``.npz`` 窗口文件。
每个文件建议包含：

    kp:    形状为 (T, 17, 3) 的骨骼序列，也兼容常见转置格式
    label: 标量标签，0 表示日常动作，1 表示跌倒

同时提供合成模式，方便在没有下载公开视频前验证训练和评测链路。合成样本只能
用于工程自测，不能替代真实视频数据上的验证指标。
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.augment.ir_augment import augment_skeleton, temporal_scale

NUM_JOINTS = 17
CHANNELS = 3


class SkeletonDataset(Dataset):
    def __init__(self, root: str | None = None, window: int = 30,
                 train: bool = True, synthetic: int = 0):
        self.root = root
        self.window = window
        self.train = train
        self.synthetic = int(synthetic or 0)
        self.files = _collect_npz(root) if root else []

        if self.synthetic <= 0 and not self.files:
            raise ValueError(
                "未找到骨骼 .npz 文件。可使用 --synthetic N 做链路自测，"
                "或用 --data 指向已抽取的骨骼窗口目录。"
            )

    def __len__(self) -> int:
        return self.synthetic if self.synthetic > 0 else len(self.files)

    def __getitem__(self, idx: int):
        if self.synthetic > 0:
            seq, label = _make_synthetic(idx, self.window, train=self.train)
        else:
            seq, label = _load_npz(self.files[idx])
            seq = _fix_window(seq, self.window)
            if self.train:
                seq = temporal_scale(seq)
                seq = augment_skeleton(seq)

        x = torch.from_numpy(seq.astype(np.float32)).permute(2, 0, 1).contiguous()
        y = torch.tensor(int(label), dtype=torch.long)
        return x, y


def _collect_npz(root: str | None) -> list[str]:
    if not root:
        return []
    p = Path(root)
    if p.is_file() and p.suffix == ".npz":
        return [str(p)]
    files = sorted(glob.glob(os.path.join(str(p), "**", "*.npz"), recursive=True))
    return files


def _load_npz(path: str):
    data = np.load(path, allow_pickle=False)
    key = "kp" if "kp" in data else "keypoints" if "keypoints" in data else "skeleton"
    seq = data[key].astype(np.float32)
    if seq.ndim != 3:
        raise ValueError(f"{path}: 期望 3 维骨骼数组，实际形状为 {seq.shape}")
    if seq.shape[0] == CHANNELS and seq.shape[2] == NUM_JOINTS:
        seq = np.transpose(seq, (1, 2, 0))
    if seq.shape[1:] != (NUM_JOINTS, CHANNELS):
        raise ValueError(f"{path}: 期望形状 (T,17,3)，实际为 {seq.shape}")

    if "label" in data:
        label = int(np.asarray(data["label"]).reshape(-1)[0])
    elif "y" in data:
        label = int(np.asarray(data["y"]).reshape(-1)[0])
    else:
        name = Path(path).as_posix().lower()
        label = 1 if "/fall" in name or "fall_" in name else 0
    return seq, label


def _fix_window(seq: np.ndarray, window: int) -> np.ndarray:
    if len(seq) == window:
        return seq.astype(np.float32)
    if len(seq) > window:
        start = np.random.randint(0, len(seq) - window + 1)
        return seq[start:start + window].astype(np.float32)
    pad_len = window - len(seq)
    pad = np.repeat(seq[:1], pad_len, axis=0)
    return np.concatenate([pad, seq], axis=0).astype(np.float32)


def _base_pose(kind: str = "stand") -> np.ndarray:
    kp = np.zeros((NUM_JOINTS, CHANNELS), dtype=np.float32)
    kp[:, 2] = 0.9

    if kind == "lie":
        coords = {
            0: (-0.75, -0.05), 5: (-0.45, -0.10), 6: (-0.45, 0.10),
            7: (-0.20, -0.22), 8: (-0.20, 0.22), 9: (0.05, -0.28), 10: (0.05, 0.28),
            11: (0.10, -0.10), 12: (0.10, 0.10), 13: (0.42, -0.12), 14: (0.42, 0.12),
            15: (0.78, -0.14), 16: (0.78, 0.14), 1: (-0.82, -0.08), 2: (-0.82, 0.08),
            3: (-0.88, -0.10), 4: (-0.88, 0.10),
        }
    elif kind == "sit":
        coords = {
            0: (0.00, -0.78), 5: (-0.22, -0.55), 6: (0.22, -0.55),
            7: (-0.35, -0.25), 8: (0.35, -0.25), 9: (-0.28, 0.02), 10: (0.28, 0.02),
            11: (-0.18, -0.02), 12: (0.18, -0.02), 13: (-0.35, 0.25), 14: (0.35, 0.25),
            15: (-0.48, 0.65), 16: (0.48, 0.65), 1: (-0.08, -0.83), 2: (0.08, -0.83),
            3: (-0.13, -0.82), 4: (0.13, -0.82),
        }
    elif kind == "bend":
        coords = {
            0: (0.45, -0.45), 5: (0.20, -0.45), 6: (0.42, -0.38),
            7: (0.45, -0.18), 8: (0.62, -0.08), 9: (0.55, 0.08), 10: (0.72, 0.18),
            11: (-0.12, 0.02), 12: (0.12, 0.02), 13: (-0.18, 0.45), 14: (0.18, 0.45),
            15: (-0.24, 0.92), 16: (0.24, 0.92), 1: (0.40, -0.50), 2: (0.50, -0.43),
            3: (0.35, -0.48), 4: (0.56, -0.40),
        }
    else:
        coords = {
            0: (0.00, -0.88), 5: (-0.22, -0.58), 6: (0.22, -0.58),
            7: (-0.38, -0.18), 8: (0.38, -0.18), 9: (-0.30, 0.20), 10: (0.30, 0.20),
            11: (-0.16, 0.02), 12: (0.16, 0.02), 13: (-0.18, 0.45), 14: (0.18, 0.45),
            15: (-0.20, 0.92), 16: (0.20, 0.92), 1: (-0.08, -0.92), 2: (0.08, -0.92),
            3: (-0.13, -0.90), 4: (0.13, -0.90),
        }

    for j, xy in coords.items():
        kp[j, :2] = xy
    return kp


def _make_synthetic(idx: int, window: int, train: bool = True):
    rng = np.random.default_rng(None if train else idx + 10007)
    label = idx % 2
    stand = _base_pose("stand")
    lie = _base_pose("lie")

    seq = []
    if label == 1:
        fall_start = int(window * rng.uniform(0.20, 0.45))
        fall_len = max(4, int(window * rng.uniform(0.18, 0.35)))
        for t in range(window):
            a = np.clip((t - fall_start) / fall_len, 0.0, 1.0)
            a = a * a * (3 - 2 * a)
            kp = (1 - a) * stand + a * lie
            kp[:, 1] += 0.35 * a
            seq.append(kp)
    else:
        mode = ["stand", "walk", "sit", "bend", "lie_static"][idx % 5]
        target = _base_pose("stand")
        if mode == "sit":
            target = _base_pose("sit")
        elif mode == "bend":
            target = _base_pose("bend")
        elif mode == "lie_static":
            target = _base_pose("lie")
        for t in range(window):
            if mode in {"sit", "bend"}:
                a = np.clip(t / max(window * 0.55, 1), 0.0, 1.0)
                kp = (1 - a) * stand + a * target
            elif mode == "walk":
                kp = stand.copy()
                kp[:, 0] += 0.10 * np.sin(2 * np.pi * t / max(window, 1))
                kp[[13, 15], 0] += 0.08 * np.sin(2 * np.pi * t / 8)
                kp[[14, 16], 0] -= 0.08 * np.sin(2 * np.pi * t / 8)
            else:
                kp = target.copy()
            seq.append(kp)

    arr = np.stack(seq, axis=0).astype(np.float32)
    arr[..., :2] += rng.normal(0, 0.025 if train else 0.015, size=arr[..., :2].shape).astype(np.float32)
    low_conf = rng.random(arr.shape[:2]) < (0.04 if train else 0.01)
    arr[..., 2] = np.where(low_conf, 0.0, arr[..., 2])
    return arr, label