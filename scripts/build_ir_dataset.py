# -*- coding: utf-8 -*-
"""为微调 YOLO-Pose 构建"退化域"训练集（伪标签自蒸馏）。

思路：预训练 YOLO 在**干净（可见光）帧**上检测姿态 → 作为伪标签（教师）；
把同一帧做**类红外 / 微光**退化 → 作为训练图像（学生输入）。
于是模型学到"即使画面变红外/变暗，也要输出干净时该有的人体框与 17 关键点"，
从而缓解长尾场景下的关键点漏检（= 跌倒漏报的根因）。

输出为 ultralytics pose 数据集格式：
    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt   # 每行: 0 cx cy w h  kx1 ky1 v1 ... kx17 ky17 v17 (归一化)
    <out>/data.yaml

用法（GPU 上跑，YOLO 推理快）:
    python scripts/build_ir_dataset.py --videos data/finetune_videos \
        --out data/ir_pose --weights yolo11n-pose.pt --stride 4 --conf 0.55 --device cuda
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.augment.ir_augment import to_infrared_like, low_light  # noqa: E402

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def degrade(img, rng):
    r = rng.random()
    if r < 0.45:
        return to_infrared_like(img, gamma=float(rng.uniform(0.5, 0.9)))
    if r < 0.85:
        return low_light(img, factor=float(rng.uniform(0.25, 0.45)),
                         noise_std=float(rng.uniform(5, 12)))
    return img            # 15% 保留干净，防止灾难性遗忘


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/ir_pose")
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--stride", type=int, default=4, help="每隔几帧取一帧")
    ap.add_argument("--conf", type=float, default=0.55, help="伪标签置信度下限")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    rng = random.Random(args.seed)

    vids = sorted(glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True))
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_ratio))
    split_of = {v: ("val" if i < n_val else "train") for i, v in enumerate(vids)}
    print(f"视频 {len(vids)} 段（train {len(vids)-n_val} / val {n_val}）")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    kept = {"train": 0, "val": 0}
    for vi, v in enumerate(vids):
        split = split_of[v]
        stem = os.path.splitext(os.path.basename(v))[0]
        cap = cv2.VideoCapture(v)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride != 0:
                idx += 1
                continue
            H, W = frame.shape[:2]
            res = model.predict(frame, verbose=False, device=args.device)[0]
            if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
                idx += 1
                continue
            confs = res.boxes.conf.cpu().numpy()
            bi = int(confs.argmax())
            if confs[bi] < args.conf:
                idx += 1
                continue
            x1, y1, x2, y2 = res.boxes.xyxy.cpu().numpy()[bi]
            kpts = res.keypoints.data.cpu().numpy()[bi]   # (17,3) 像素

            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
            for kx, ky, kc in kpts:
                vis = 2 if kc > 0.5 else (1 if kc > 0.2 else 0)
                parts.append(f"{kx/W:.6f} {ky/H:.6f} {vis}")
            label = " ".join(parts)

            dimg = degrade(frame, rng)
            name = f"{stem}_{idx:05d}"
            cv2.imwrite(os.path.join(args.out, "images", split, name + ".jpg"), dimg)
            with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
                f.write(label + "\n")
            kept[split] += 1
            idx += 1
        cap.release()
        if (vi + 1) % 30 == 0:
            print(f"  {vi+1}/{len(vids)} 段处理完，累计样本 {sum(kept.values())}")

    yaml = (
        f"path: {os.path.abspath(args.out)}\n"
        f"train: images/train\nval: images/val\n"
        f"kpt_shape: [17, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        f"names:\n  0: person\n"
    )
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(yaml)
    print(f"完成：train {kept['train']} + val {kept['val']} 张 -> {args.out}")
    print(f"data.yaml 已写入。微调命令示例：")
    print(f"  yolo pose train model={args.weights} data={args.out}/data.yaml "
          f"epochs=40 imgsz=640 batch=32 device=0")


if __name__ == "__main__":
    main()
