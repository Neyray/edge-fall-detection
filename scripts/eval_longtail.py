# -*- coding: utf-8 -*-
"""长尾场景鲁棒性评测：对同一批视频施加「类红外 / 微光 / 遮挡」退化，
跑完整流水线（YOLO-Pose → ST-GCN → 几何 → 状态机），给出**每种条件下的事件级指标**。

回答的问题：骨骼中间表征在跨域/退化场景下是否真的保持鲁棒——用数字，而不是文字。

评测协议与 `ablation.py` 一致：以**视频片段**为单位，整段是否报警；
跌倒片段报警=TP，日常片段报警=FP。视频均来自**未参与训练**的留出集。

用法:
    python scripts/eval_longtail.py --videos data/longtail_videos \
        --ckpt checkpoints/stgcn_fall.pt --weights yolo11n-pose.pt \
        --device cuda --out reports/longtail.json
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import FallDetector  # noqa: E402
from src.augment.ir_augment import to_infrared_like, low_light, occlusion_patch  # noqa: E402


def clahe_enhance(img):
    """CLAHE 自适应直方图均衡：在送入 YOLO 前拉回对比度。

    这是**零模型成本**的图像预处理（端侧 ISP 即可完成），用于缓解红外/微光下
    因对比度塌缩导致的关键点漏检。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def make_degrader(cond, seed, shape):
    """返回一个逐帧退化函数。遮挡框整段固定（静态遮挡物）。"""
    if cond == "clean":
        return lambda im: im
    if cond == "ir":
        return lambda im: to_infrared_like(im, gamma=0.7)
    if cond == "lowlight":
        return lambda im: low_light(im, factor=0.35, noise_std=8.0)
    if cond == "occlusion":
        h, w = shape[:2]
        rng = np.random.default_rng(seed)
        pw, ph = int(w * 0.30), int(h * 0.40)
        x0 = int(rng.integers(int(w * 0.15), max(int(w * 0.15) + 1, w - pw - int(w * 0.15))))
        y0 = int(rng.integers(int(h * 0.30), max(int(h * 0.30) + 1, h - ph)))
        return lambda im: occlusion_patch(im, (x0, y0, pw, ph))
    raise ValueError(cond)


def run_clip(det, path, cond, lie_seconds, fps_fallback, preprocess="none"):
    cap = cv2.VideoCapture(path)
    ok, first = cap.read()
    if not ok:
        cap.release()
        return None
    f = cap.get(cv2.CAP_PROP_FPS)
    if not f or f <= 0 or f != f:
        f = fps_fallback
    det.set_fps(f)                      # 用真实帧率驱动状态机
    degrade = make_degrader(cond, abs(hash(os.path.basename(path))) % (2**31), first.shape)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    alarmed = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = degrade(frame)
        if preprocess == "clahe":
            img = clahe_enhance(img)
        res = det.process_frame(img)
        alarmed |= any(r["alarm"] for r in res)
    cap.release()
    return alarmed


def metrics(preds, ys):
    tp = sum(p and y for p, y in zip(preds, ys))
    fp = sum(p and not y for p, y in zip(preds, ys))
    fn = sum((not p) and y for p, y in zip(preds, ys))
    tn = sum((not p) and (not y) for p, y in zip(preds, ys))
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(recall=round(rec, 3), precision=round(prec, 3), fpr=round(fpr, 3),
                f1=round(f1, 3), tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--ckpt", default="checkpoints/stgcn_fall.pt")
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/longtail.json")
    ap.add_argument("--lie-seconds", type=float, default=0.2)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--conditions", default="clean,ir,lowlight,occlusion")
    ap.add_argument("--preprocess", default="none", choices=["none", "clahe"],
                    help="送入 YOLO 前的图像预处理（clahe=自适应直方图均衡，零模型成本）")
    args = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True))
    ys = [("no_fall" not in v.replace("\\", "/")) for v in vids]
    print(f"视频 {len(vids)} 段（跌倒 {sum(ys)} / 日常 {len(ys)-sum(ys)}），设备 {args.device}")

    det = FallDetector(pose_weights=args.weights, temporal_ckpt=args.ckpt,
                       device=args.device, fps=args.fps,
                       fsm_kwargs={"lie_seconds": args.lie_seconds})

    table = {}
    for cond in args.conditions.split(","):
        preds, keep = [], []
        for i, v in enumerate(vids):
            a = run_clip(det, v, cond, args.lie_seconds, args.fps, args.preprocess)
            if a is None:
                continue
            preds.append(a)
            keep.append(ys[i])
            if (i + 1) % 20 == 0:
                print(f"  [{cond}] {i+1}/{len(vids)}")
        table[cond] = metrics(preds, keep)
        m = table[cond]
        print(f"[{cond:>9}] 召回 {m['recall']:.3f}  精确 {m['precision']:.3f}  "
              f"误报率 {m['fpr']:.3f}  F1 {m['f1']:.3f}")

    name = {"clean": "干净(可见光)", "ir": "类红外", "lowlight": "微光/夜间", "occlusion": "遮挡"}
    print("\n条件            召回    精确    误报率   F1")
    print("-" * 48)
    for c, m in table.items():
        print(f"{name.get(c,c):<12}  {m['recall']:.3f}  {m['precision']:.3f}  {m['fpr']:.3f}   {m['f1']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_clips": len(vids), "fall": sum(ys), "adl": len(ys) - sum(ys),
                   "device": args.device, "lie_seconds": args.lie_seconds,
                   "preprocess": args.preprocess,
                   "note": "视频全部来自未参与训练的留出片段；退化在图像域施加后再跑完整流水线",
                   "conditions": {name.get(c, c): m for c, m in table.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {args.out}")


if __name__ == "__main__":
    main()
