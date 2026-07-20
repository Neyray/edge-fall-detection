# -*- coding: utf-8 -*-
"""赛题硬性指标核对与时延测试。

脚本默认保持离线安全：直接测试时序头；只有在本地存在有效 YOLO-Pose 权重时，
才测试第一阶段姿态模型时延。确实需要让 Ultralytics 自动下载官方权重时，再显式使用
`--allow-download`。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.temporal.stgcn import LiteSTGCN, count_parameters


LIMITS = {"params_M": 20.0, "fp32_MB": 80.0, "latency_ms": 100.0, "npu_MB": 20.0}
YOLO11N_POSE_NOMINAL = {"params": 2.87e6, "fp32_mb": 11.5}


def bench_temporal(window=30, runs=100, device="cpu"):
    model = LiteSTGCN(num_class=2).to(device).eval()
    params = count_parameters(model)
    fp32_mb = params * 4 / 1e6
    x = torch.randn(1, 3, window, 17, device=device)
    with torch.no_grad():
        for _ in range(min(10, runs)):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / max(runs, 1) * 1000
    return params, fp32_mb, dt


def bench_pose(weights, runs=30, imgsz=640, allow_download=False):
    if not weights:
        return None
    local_exists = os.path.exists(weights)
    if not local_exists and not allow_download:
        print(f"[pose] 未找到本地权重：{weights}，跳过姿态模型时延测试")
        return None
    if local_exists and os.path.getsize(weights) < 3_000_000:
        print(f"[pose] 本地权重疑似不完整：{weights}，跳过姿态模型时延测试")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[pose] 未安装 ultralytics，跳过姿态模型时延测试")
        return None
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[pose] 加载失败（{e}），跳过姿态模型时延测试")
        return None

    n_params = sum(p.numel() for p in model.model.parameters())
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(img, imgsz=imgsz, verbose=False)
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(img, imgsz=imgsz, verbose=False)
    dt = (time.perf_counter() - t0) / max(runs, 1) * 1000
    return n_params, n_params * 4 / 1e6, dt


def print_limit(name, val, limit, unit):
    ok = "OK " if val <= limit else "FAIL"
    print(f"  [{ok}] {name:16s}: {val:8.2f} {unit}  (<= {limit} {unit})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="yolo11n-pose.pt", help="YOLO-Pose 权重路径")
    ap.add_argument("--runs", type=int, default=100, help="重复测试次数")
    ap.add_argument("--window", type=int, default=30, help="时序窗口帧数")
    ap.add_argument("--skip-pose", action="store_true", help="跳过 YOLO-Pose 时延测试")
    ap.add_argument("--allow-download", action="store_true",
                    help="允许 Ultralytics 自动下载缺失的官方权重")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 基准设备：{device} ===\n")

    tp, tfp32, tlat = bench_temporal(args.window, args.runs, device)
    print("[阶段2 Lite ST-GCN]")
    print(f"  参数量   : {tp/1e6:.3f} M")
    print(f"  fp32大小 : {tfp32:.2f} MB")
    print(f"  时延     : {tlat:.3f} ms/window ({device})\n")

    pose = None if args.skip_pose else bench_pose(
        args.pose, max(args.runs // 3, 10), allow_download=args.allow_download
    )
    if pose:
        pp, pfp32, plat = pose
        print("[阶段1 YOLO-Pose]")
        print(f"  参数量   : {pp/1e6:.3f} M")
        print(f"  fp32大小 : {pfp32:.2f} MB")
        print(f"  时延     : {plat:.3f} ms/frame ({device})\n")
    else:
        pp = YOLO11N_POSE_NOMINAL["params"]
        pfp32 = YOLO11N_POSE_NOMINAL["fp32_mb"]
        plat = None
        print("[阶段1 YOLO-Pose] 仅使用标称值")
        print(f"  参数量   : ~{pp/1e6:.2f} M")
        print(f"  fp32大小 : ~{pfp32:.1f} MB")
        print("  时延     : 未实测（提供本地权重或使用 --allow-download 后可测）\n")

    total_params = (tp + pp) / 1e6
    total_fp32 = tfp32 + pfp32

    print("=== 合规核对 ===")
    print_limit("模型总参数", total_params, LIMITS["params_M"], "M")
    print_limit("fp32权重", total_fp32, LIMITS["fp32_MB"], "MB")
    if plat is not None:
        print_limit("端到端时延*", tlat + plat, LIMITS["latency_ms"], "ms")
    else:
        print("  [INFO] 端到端时延*: 已测时序头，未测姿态模型")
    print("\n  * 桌面 CPU/GPU 时延仅供参考；最终端侧 NPU 时延需要在板端实测。")


if __name__ == "__main__":
    main()