# -*- coding: utf-8 -*-
"""评测骨骼时序跌倒分类器。

示例：
    python scripts/evaluate_temporal.py --synthetic 512 --ckpt checkpoints/stgcn_fall.pt
    python scripts/evaluate_temporal.py --data data/skeletons/val --ckpt checkpoints/stgcn_fall.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.datasets.skeleton_dataset import SkeletonDataset
from src.temporal.stgcn import LiteSTGCN, count_parameters


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return {
        "samples": total,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    latencies = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000 / max(x.size(0), 1))

        pred = logits.argmax(1)
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())

    metrics = compute_metrics(tp, fp, tn, fn)
    metrics["temporal_latency_ms_per_window"] = sum(latencies) / max(len(latencies), 1)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="验证集骨骼 .npz 目录")
    ap.add_argument("--ckpt", default="checkpoints/stgcn_fall.pt", help="时序模型权重路径")
    ap.add_argument("--synthetic", type=int, default=256,
                    help="未指定 --data 时使用的合成验证样本数")
    ap.add_argument("--window", type=int, default=30, help="滑动窗口帧数")
    ap.add_argument("--bs", type=int, default=128, help="评测 batch size")
    ap.add_argument("--out", default="reports/eval_metrics.json", help="指标 JSON 输出路径")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SkeletonDataset(args.data, window=args.window, train=False,
                              synthetic=0 if args.data else args.synthetic)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=0)

    model = LiteSTGCN(num_class=2).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state.get("model", state))
        ckpt_note = args.ckpt
    else:
        ckpt_note = "未加载权重_随机初始化"
        print(f"[警告] 未找到权重文件：{args.ckpt}，将评测随机初始化模型")

    metrics = evaluate(model, loader, device)
    metrics.update({
        "device": device,
        "checkpoint": ckpt_note,
        "data": args.data or f"synthetic:{args.synthetic}",
        "window": args.window,
        "model_params_M": count_parameters(model) / 1e6,
    })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== 时序模型评测 ===")
    print(f"数据       : {metrics['data']}")
    print(f"权重       : {metrics['checkpoint']}")
    print(f"设备       : {metrics['device']}")
    print(f"样本数     : {metrics['samples']}")
    print(f"准确率     : {metrics['accuracy']:.3f}")
    print(f"召回率     : {metrics['recall']:.3f}")
    print(f"精确率     : {metrics['precision']:.3f}")
    print(f"F1         : {metrics['f1']:.3f}")
    print(f"误报/漏报率: {metrics['false_positive_rate']:.3f} / {metrics['false_negative_rate']:.3f}")
    print(f"混淆矩阵   : {metrics['confusion']}")
    print(f"时延       : {metrics['temporal_latency_ms_per_window']:.3f} ms/window")
    print(f"[已保存] {args.out}")


if __name__ == "__main__":
    main()