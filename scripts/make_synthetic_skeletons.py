# -*- coding: utf-8 -*-
"""生成一小份合成骨骼窗口数据，用于验证工程链路。

这不是可汇报的真实训练证据，只用于确认 `--data` 模式能读取与
`extract_skeletons.py` 一致的 `.npz` 格式。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.datasets.skeleton_dataset import _make_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/skeletons/sample", help="输出目录")
    ap.add_argument("--samples", type=int, default=40, help="生成样本数")
    ap.add_argument("--window", type=int, default=30, help="每个样本的帧窗口长度")
    ap.add_argument("--keep-existing", action="store_true",
                    help="保留已有 .npz 文件并追加生成；默认会先清空旧样本")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.keep_existing:
        for old in os.listdir(args.out):
            if old.endswith(".npz"):
                os.remove(os.path.join(args.out, old))

    for i in range(args.samples):
        seq, label = _make_synthetic(i, args.window, train=False)
        np.savez_compressed(
            os.path.join(args.out, f"sample_{i:04d}.npz"),
            kp=seq.astype(np.float32),
            label=int(label),
        )
    print(f"[已保存] {args.samples} 个合成骨骼窗口 -> {args.out}")


if __name__ == "__main__":
    main()