# -*- coding: utf-8 -*-
"""端到端 demo：对真实视频跑完整流水线（YOLO-Pose → ST-GCN → 几何 → 状态机），
在画面上叠加骨骼、状态与「跌倒报警」，输出标注后的 mp4。

为避免 CPU 过热，可用 --threads 限制线程数（默认 4）。这是纯推理，不训练，负载远低于训练。

用法:
    python scripts/demo_video.py --videos data/demo_videos --out reports/demo \
        --ckpt checkpoints/stgcn_fall.pt --weights yolo11n-pose.pt --threads 4
"""
import argparse
import glob
import os
import sys

# 线程限制必须在 import torch 之前设置环境变量
_ap = argparse.ArgumentParser()
_ap.add_argument("--videos", required=True, help="视频目录（递归找 *.mp4；建议含 fall/ 与 no_fall/）")
_ap.add_argument("--out", default="reports/demo")
_ap.add_argument("--ckpt", default="checkpoints/stgcn_fall.pt")
_ap.add_argument("--weights", default="yolo11n-pose.pt")
_ap.add_argument("--threads", type=int, default=4, help="限制 CPU 线程数，降温用")
_ap.add_argument("--fps", type=float, default=30.0)
_ap.add_argument("--lie-seconds", type=float, default=0.4, help="状态机'持续倒地'阈值(短事件片段用小值)")
args = _ap.parse_args()

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(args.threads)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(args.threads)
cv2.setNumThreads(args.threads)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import FallDetector  # noqa: E402

# COCO-17 骨架连线
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6)]


def draw(frame, res):
    alarm_any = False
    for r in res:
        kp = np.array(r["keypoints"], dtype=np.float32)
        state = r["state"]
        alarm_any |= bool(r["alarm"])
        color = (0, 180, 0) if state == "NORMAL" else (0, 165, 255) if state == "FALLING" else (0, 0, 255)
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for a, b in SKEL:
            if kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
                cv2.line(frame, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), color, 2)
        for j in range(17):
            if kp[j, 2] > 0.2:
                cv2.circle(frame, tuple(kp[j, :2].astype(int)), 3, color, -1)
        cv2.putText(frame, f"{state}  p={r['fused_prob']:.2f}", (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if alarm_any:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 255), -1)
        cv2.putText(frame, "FALL DETECTED - ALARM", (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    return frame


def run_video(det, path, out_path):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = cap.get(cv2.CAP_PROP_FPS)
    if not f or f <= 0 or f != f:      # 读不到或 NaN 时回退
        f = args.fps
    # 用本视频的真实帧率重建状态机（"持续N秒"与下坠速度都依赖帧率）
    det.set_fps(f)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), f, (w, h))
    ever, nframe = False, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = det.process_frame(frame)
        ever |= any(r["alarm"] for r in res)
        vw.write(draw(frame, res))
        nframe += 1
    cap.release()
    vw.release()
    return ever, nframe


def main():
    os.makedirs(args.out, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True))
    if not vids:
        sys.exit(f"未找到视频：{args.videos}/**/*.mp4")
    print(f"[device] cpu  [threads] {args.threads}  [videos] {len(vids)}")

    det = FallDetector(pose_weights=args.weights, temporal_ckpt=args.ckpt,
                       device="cpu", fps=args.fps,
                       fsm_kwargs={"lie_seconds": args.lie_seconds})

    print("\n视频                              真值    是否报警   帧数")
    print("-" * 60)
    rows = []
    for p in vids:
        truth = "跌倒" if ("fall" in p.lower() and "no_fall" not in p.lower()) else "日常"
        name = os.path.basename(p)
        outp = os.path.join(args.out, f"demo_{name}")
        alarmed, nf = run_video(det, p, outp)
        flag = "★报警" if alarmed else "—"
        rows.append((truth, alarmed))
        print(f"{name:<32}  {truth}    {flag:<6}   {nf}")

    tp = sum(1 for t, a in rows if t == "跌倒" and a)
    nf_total = sum(1 for t, a in rows if t == "跌倒")
    fp = sum(1 for t, a in rows if t == "日常" and a)
    adl_total = sum(1 for t, a in rows if t == "日常")
    print("-" * 60)
    print(f"跌倒视频报警: {tp}/{nf_total}   日常视频误报: {fp}/{adl_total}")
    print(f"标注视频已输出到: {args.out}")


if __name__ == "__main__":
    main()
