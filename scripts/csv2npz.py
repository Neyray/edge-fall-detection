# -*- coding: utf-8 -*-
"""把 Kaggle payutch/fall-video-dataset 的骨骼关键点 CSV 转成本项目训练用的 (T,17,3)+label 的 .npz 窗口。

CSV 为长表格式，每帧 17 行：
    Frame,Keypoint,X,Y,Confidence
    1,Nose,711.29,176.38,0.9998
    1,Left Eye,721.22,158.00,0.9995
    ...
标签由路径决定：路径含 no_fall / nofall / adl -> 0（日常）；否则含 fall -> 1（跌倒）。

归一化与 src/pose/extractor.py::normalize_keypoints 保持一致——逐帧按人体框把坐标缩放到 [-1,1]，
消除位置/尺度差异；这里没有 YOLO 检测框，改用当帧可见关键点的外接框近似（自洽即可）。

用法:
    python scripts/csv2npz.py --src data/kaggle_csv --out data/skeletons/all \
        --window 30 --stride 15 --limit-per-class 500
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

NUM_JOINTS = 17
CHANNELS = 3

# COCO-17 关键点顺序（与 ultralytics YOLO-Pose 一致）
COCO17 = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle",
]
NAME2IDX = {n: i for i, n in enumerate(COCO17)}


def _norm_name(s: str) -> str:
    return str(s).strip().lower().replace("_", " ")


def csv_to_seq(path: str) -> np.ndarray:
    """读一个 CSV -> (T,17,3) 像素坐标序列（未归一化）。读失败返回 None。"""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["frame", "keypoint", "x", "y", "confidence"]
    if not all(k in cols for k in need):
        return None
    fcol, kcol, xcol, ycol, ccol = (cols[k] for k in need)

    frames = sorted(df[fcol].unique(), key=lambda v: float(v))
    seq = np.zeros((len(frames), NUM_JOINTS, CHANNELS), dtype=np.float32)
    fidx = {f: i for i, f in enumerate(frames)}
    for _, row in df.iterrows():
        j = NAME2IDX.get(_norm_name(row[kcol]))
        if j is None:
            continue
        i = fidx[row[fcol]]
        seq[i, j, 0] = row[xcol]
        seq[i, j, 1] = row[ycol]
        seq[i, j, 2] = row[ccol]
    return seq


def normalize_seq(seq: np.ndarray, bbox_conf: float = 0.1) -> np.ndarray:
    """逐帧按可见关键点外接框归一化到 [-1,1]，复刻 normalize_keypoints 的语义。"""
    out = seq.copy().astype(np.float32)
    for t in range(len(seq)):
        pts = seq[t]
        vis = pts[:, 2] > bbox_conf
        use = pts[vis] if vis.sum() >= 2 else pts
        x1, x2 = use[:, 0].min(), use[:, 0].max()
        y1, y2 = use[:, 1].min(), use[:, 1].max()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w = max(x2 - x1, 1e-3)
        h = max(y2 - y1, 1e-3)
        out[t, :, 0] = (pts[:, 0] - cx) / (w / 2)
        out[t, :, 1] = (pts[:, 1] - cy) / (h / 2)
        out[t, :, 2] = pts[:, 2]
    return out


def windows_from_seq(seq: np.ndarray, window: int, stride: int):
    """切滑动窗口；不足一窗则前向 pad 成一窗。"""
    T = len(seq)
    if T >= window:
        for s in range(0, T - window + 1, stride):
            yield seq[s:s + window]
    elif T > 0:
        pad = np.repeat(seq[:1], window - T, axis=0)
        yield np.concatenate([pad, seq], axis=0)


def label_of(path: str):
    p = path.replace("\\", "/").lower()
    if "no_fall" in p or "nofall" in p or "no fall" in p or "/adl" in p:
        return 0
    if "fall" in p:
        return 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="解压后的 CSV 根目录（递归查找 *.csv）")
    ap.add_argument("--out", required=True, help="输出 npz 目录")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help=">0 时每类最多取这么多个 CSV（均衡+提速），0 表示全部")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_csv = sorted(glob.glob(os.path.join(args.src, "**", "*.csv"), recursive=True))
    buckets = {0: [], 1: []}
    for c in all_csv:
        lb = label_of(c)
        if lb is not None:
            buckets[lb].append(c)
    if args.limit_per_class > 0:
        for lb in buckets:
            buckets[lb] = buckets[lb][:args.limit_per_class]

    print(f"CSV 命中: fall={len(buckets[1])}  no_fall={len(buckets[0])}")
    if not buckets[0] or not buckets[1]:
        sys.exit("错误：某一类为空，检查 --src 路径与目录名（需含 Fall / No_Fall）。")

    total_win = {0: 0, 1: 0}
    saved = 0
    for lb in (1, 0):
        tag = "fall" if lb == 1 else "adl"
        for ci, path in enumerate(buckets[lb]):
            seq = csv_to_seq(path)
            if seq is None or len(seq) == 0:
                continue
            seq = normalize_seq(seq)
            for wi, win in enumerate(windows_from_seq(seq, args.window, args.stride)):
                np.savez_compressed(
                    os.path.join(args.out, f"{tag}_{ci:05d}_{wi:03d}.npz"),
                    kp=win.astype(np.float32), label=np.int64(lb),
                )
                total_win[lb] += 1
                saved += 1
            if (ci + 1) % 200 == 0:
                print(f"  [{tag}] {ci + 1}/{len(buckets[lb])} 个 CSV 已处理")

    print(f"完成：共写出 {saved} 个窗口  (fall={total_win[1]}, adl={total_win[0]}) -> {args.out}")


if __name__ == "__main__":
    main()
