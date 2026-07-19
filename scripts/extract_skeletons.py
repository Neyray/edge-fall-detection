# -*- coding: utf-8 -*-
"""离线骨骼抽取：把跌倒视频数据集转成 (T,17,3) 骨骼序列 .npz。

按滑动窗口切片，标签由目录名（fall / adl）或标注文件决定。
适配参考数据集:
    - HuggingFace: simplexsigil2/omnifall
    - Kaggle: payutch/fall-video-dataset
目录组织约定:
    <root>/fall/*.mp4   -> label 1
    <root>/adl/*.mp4    -> label 0   (activities of daily living)

用法:
    python scripts/extract_skeletons.py --videos data/raw --out data/skeletons/train
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.pose.extractor import PoseExtractor, normalize_keypoints


def process_video(path, extractor, window, stride, label, out_dir, prefix):
    try:
        import cv2
    except ImportError:
        raise SystemExit("需要 opencv-python 读取视频: pip install opencv-python")
    cap = cv2.VideoCapture(path)
    buf = []
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        persons = extractor(frame)
        if persons:
            p = max(persons, key=lambda x: x.det_conf)   # 取主体
            buf.append(normalize_keypoints(p.keypoints, p.bbox))
        else:
            buf.append(np.zeros((17, 3), dtype=np.float32))
        if len(buf) >= window:
            seq = np.stack(buf[-window:], axis=0)
            np.savez_compressed(os.path.join(out_dir, f"{prefix}_{saved:05d}.npz"),
                                kp=seq.astype(np.float32), label=label)
            saved += 1
            del buf[:stride]
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="视频根目录(含 fall/ 与 adl/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    extractor = PoseExtractor(args.weights, conf=0.3)

    total = 0
    for sub, label in [("fall", 1), ("adl", 0)]:
        vids = glob.glob(os.path.join(args.videos, sub, "*.*"))
        for v in vids:
            n = process_video(v, extractor, args.window, args.stride, label,
                              args.out, f"{sub}_{os.path.splitext(os.path.basename(v))[0]}")
            total += n
            print(f"{v} -> {n} 片段")
    print(f"[done] 共生成 {total} 个骨骼片段 -> {args.out}")


if __name__ == "__main__":
    main()
