# -*- coding: utf-8 -*-
"""训练第二阶段时序判别头 (Lite ST-GCN)。

用法:
    # 无真实数据也能跑通（合成数据自测）
    python scripts/train_temporal.py --synthetic 1024 --epochs 5

    # 使用离线抽取的骨骼 .npz
    python scripts/train_temporal.py --data data/skeletons/train --val data/skeletons/val
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.skeleton_dataset import SkeletonDataset
from src.temporal.stgcn import LiteSTGCN, count_parameters


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    acc = correct / max(total, 1)
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return acc, recall, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="训练集骨骼 npz 目录")
    ap.add_argument("--val", default=None, help="验证集骨骼 npz 目录")
    ap.add_argument("--synthetic", type=int, default=0, help=">0 时用合成样本数")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/stgcn_fall.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    train_set = SkeletonDataset(args.data, window=args.window, train=True,
                                synthetic=args.synthetic)
    val_root = args.val or args.data
    val_set = SkeletonDataset(val_root, window=args.window, train=False,
                              synthetic=max(args.synthetic // 4, 128) if args.synthetic else 0)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.bs, num_workers=2)
    print(f"[data] train={len(train_set)} val={len(val_set)}")

    model = LiteSTGCN(num_class=2).to(device)
    print(f"[model] params = {count_parameters(model)/1e6:.3f} M")

    # 跌倒类加权，倾向高召回（漏报代价高于误报）
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
        sched.step()
        acc, rec, prec = evaluate(model, val_loader, device)
        print(f"epoch {ep:02d} | loss {run/len(train_set):.4f} | "
              f"val acc {acc:.3f} recall {rec:.3f} prec {prec:.3f}")
        if acc >= best:
            best = acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "acc": acc, "window": args.window}, args.out)
    print(f"[done] best val acc {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
