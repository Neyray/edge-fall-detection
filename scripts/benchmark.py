# -*- coding: utf-8 -*-
"""合规性 & 性能基准：核对赛题硬性指标。

统计两阶段模型参数量、fp32 权重大小、单帧端到端时延，并对照赛题红线：
    模型总参数 <= 20M
    fp32 权重  <= 80MB
    端侧单帧推理 <= 100ms
    NPU 存储占用 <= 20MB
(桌面 GPU/CPU 时延仅供参考，端侧真实时延需在目标 NPU 上实测。)

用法:
    python scripts/benchmark.py
    python scripts/benchmark.py --pose yolo11n-pose.pt --runs 50
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.temporal.stgcn import LiteSTGCN, count_parameters


LIMITS = {"params_M": 20.0, "fp32_MB": 80.0, "latency_ms": 100.0, "npu_MB": 20.0}


def bench_temporal(window=30, runs=100, device="cpu"):
    model = LiteSTGCN(num_class=2).to(device).eval()
    params = count_parameters(model)
    fp32_mb = params * 4 / 1e6
    x = torch.randn(1, 3, window, 17, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / runs * 1000
    return params, fp32_mb, dt


def bench_pose(weights, runs=30, imgsz=640):
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if not (os.path.exists(weights) or weights.startswith("yolo")):
        return None
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[pose] 加载失败({e})，跳过 pose 基准")
        return None
    n_params = sum(p.numel() for p in model.model.parameters())
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(img, imgsz=imgsz, verbose=False)
    t0 = time.time()
    for _ in range(runs):
        model.predict(img, imgsz=imgsz, verbose=False)
    dt = (time.time() - t0) / runs * 1000
    return n_params, n_params * 4 / 1e6, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 基准设备: {device} ===\n")

    tp, tfp32, tlat = bench_temporal(args.window, args.runs, device)
    print("[阶段2 Lite ST-GCN]")
    print(f"  参数量   : {tp/1e6:.3f} M")
    print(f"  fp32大小 : {tfp32:.2f} MB")
    print(f"  单次时延 : {tlat:.3f} ms  ({device})\n")

    pose = bench_pose(args.pose, max(args.runs // 3, 10))
    if pose:
        pp, pfp32, plat = pose
        print("[阶段1 YOLO-Pose]")
        print(f"  参数量   : {pp/1e6:.3f} M")
        print(f"  fp32大小 : {pfp32:.2f} MB")
        print(f"  单帧时延 : {plat:.3f} ms  ({device})\n")
    else:
        pp, pfp32, plat = 2.9e6, 11.6, 0.0
        print("[阶段1 YOLO-Pose] (未加载权重，使用 yolo11n-pose 标称值)")
        print(f"  参数量   : ~{pp/1e6:.1f} M  fp32 ~{pfp32:.1f} MB\n")

    total_params = (tp + pp) / 1e6
    total_fp32 = tfp32 + pfp32
    total_lat = tlat + plat

    print("=== 合规核对 (对照赛题红线) ===")
    def line(name, val, limit, unit):
        ok = "OK " if val <= limit else "FAIL"
        print(f"  [{ok}] {name:16s}: {val:8.2f} {unit}  (<= {limit} {unit})")
    line("模型总参数", total_params, LIMITS["params_M"], "M")
    line("fp32权重", total_fp32, LIMITS["fp32_MB"], "MB")
    if total_lat > 0:
        line("端到端时延*", total_lat, LIMITS["latency_ms"], "ms")
    print("\n  * 桌面时延仅供参考；端侧 NPU(海思/瑞芯微)真实时延需目标板实测。")


if __name__ == "__main__":
    main()
