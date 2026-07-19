# -*- coding: utf-8 -*-
"""端到端演示：对视频/摄像头逐帧检测并可视化报警。

用法:
    python scripts/run_demo.py --source path/to/video.mp4 --save out.mp4
    python scripts/run_demo.py --source 0                 # 摄像头
    python scripts/run_demo.py --synthetic                # 无摄像头/无权重时的合成自测
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.pipeline import FallDetector


def run_synthetic(n_frames=150):
    """无外设自测：合成一段"先站立后跌倒"的帧序列，验证流水线与报警逻辑。"""
    det = FallDetector(dummy_pose=True, window=30, fps=30.0)
    triggered = False
    for i in range(n_frames):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        res = det.process_frame(frame)
        for r in res:
            if r["newly_triggered"]:
                triggered = True
                print(f"[frame {i}] *** 跌倒报警 *** track={r['track_id']} "
                      f"fused={r['fused_prob']:.2f} state={r['state']}")
    print(f"[synthetic] 处理 {n_frames} 帧完成, 触发报警: {triggered}")
    return triggered


def run_video(source, weights, ckpt, save):
    try:
        import cv2
    except ImportError:
        raise SystemExit("需要 opencv-python: pip install opencv-python")

    det = FallDetector(pose_weights=weights, temporal_ckpt=ckpt,
                       device="cuda" if _has_cuda() else "cpu", window=30)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if save:
        writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = det.process_frame(frame)
        for r in results:
            x1, y1, x2, y2 = map(int, r["bbox"])
            alarm = r["alarm"]
            color = (0, 0, 255) if alarm else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{r['state']} p={r['fused_prob']:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if alarm:
                cv2.putText(frame, "FALL ALARM", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if writer:
            writer.write(frame)
        frames += 1
    cap.release()
    if writer:
        writer.release()
    dt = time.time() - t0
    print(f"[done] {frames} 帧, 平均 {frames/max(dt,1e-6):.1f} FPS")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic or args.source is None:
        run_synthetic()
    else:
        run_video(args.source, args.weights, args.ckpt, args.save)


if __name__ == "__main__":
    main()
