# -*- coding: utf-8 -*-
"""真实数据准备（选片 -> 按片段划分 -> 各自转窗口）。

相比 csv2npz + split_dataset 两步，这里**先在 CSV(片段)层面划分 train/val，再分别转窗口**，
好处：验证集用的是哪些原始视频片段完全可控且零泄漏，后续片段级消融(ablation.py)正好在这批
val 片段上跑。同时记录 val 片段清单到 --val-list。

用法:
    python scripts/prepare_real.py --src data/kaggle_csv --per-class 1500 \
        --train data/skeletons/train --val data/skeletons/val \
        --val-list reports/val_clips.txt --window 30 --stride 15
"""
import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csv2npz import csv_to_seq, normalize_seq, windows_from_seq, label_of  # noqa: E402


def collect(src):
    fall, adl = [], []
    for c in sorted(glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)):
        lb = label_of(c)
        (fall if lb == 1 else adl if lb == 0 else []).append(c)
    return fall, adl


def convert(csv_list, out_dir, window, stride, rng=None, balance=True):
    os.makedirs(out_dir, exist_ok=True)
    written = {0: [], 1: []}
    for ci, path in enumerate(csv_list):
        lb = label_of(path)
        tag = "fall" if lb == 1 else "adl"
        seq = csv_to_seq(path)
        if seq is None or len(seq) == 0:
            continue
        seq = normalize_seq(seq)
        for wi, win in enumerate(windows_from_seq(seq, window, stride)):
            fp = os.path.join(out_dir, f"{tag}_{ci:05d}_{wi:03d}.npz")
            np.savez_compressed(fp, kp=win.astype(np.float32), label=np.int64(lb))
            written[lb].append(fp)
    # 窗口级类别均衡：下采样多数类（删除多余窗口文件）
    if balance and rng is not None:
        n = min(len(written[0]), len(written[1]))
        for lb in (0, 1):
            extra = written[lb][n:] if len(written[lb]) > n else []
            keep = set(rng.sample(written[lb], n)) if len(written[lb]) > n else set(written[lb])
            for fp in written[lb]:
                if fp not in keep:
                    os.remove(fp)
        return n * 2, n, n
    return sum(len(v) for v in written.values()), len(written[1]), len(written[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--train", default="data/skeletons/train")
    ap.add_argument("--val", default="data/skeletons/val")
    ap.add_argument("--val-list", default="reports/val_clips.txt")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fall, adl = collect(args.src)
    print(f"可用片段: fall={len(fall)}  adl={len(adl)}")

    def pick_split(lst):
        rng.shuffle(lst)
        lst = lst[:args.per_class]
        nval = max(1, int(len(lst) * args.val_ratio))
        return lst[nval:], lst[:nval]      # train_csvs, val_csvs

    f_tr, f_va = pick_split(fall)
    a_tr, a_va = pick_split(adl)

    # 片段级均衡：train/val 内各类片段数取齐
    def bal(a, b):
        n = min(len(a), len(b))
        return a[:n], b[:n]
    f_tr, a_tr = bal(f_tr, a_tr)
    f_va, a_va = bal(f_va, a_va)

    for d in (args.train, args.val):
        if os.path.isdir(d):
            import shutil
            shutil.rmtree(d)

    ntr, ntr_f, ntr_a = convert(f_tr + a_tr, args.train, args.window, args.stride, rng)
    nva, nva_f, nva_a = convert(f_va + a_va, args.val, args.window, args.stride, rng)

    os.makedirs(os.path.dirname(args.val_list) or ".", exist_ok=True)
    with open(args.val_list, "w") as fh:
        for p in (f_va + a_va):
            fh.write(f"{label_of(p)}\t{p}\n")

    print(f"训练片段: fall={len(f_tr)} adl={len(a_tr)} -> 窗口均衡后 {ntr} (fall={ntr_f}, adl={ntr_a}) -> {args.train}")
    print(f"验证片段: fall={len(f_va)} adl={len(a_va)} -> 窗口均衡后 {nva} (fall={nva_f}, adl={nva_a}) -> {args.val}")
    print(f"验证片段清单 -> {args.val_list}")


if __name__ == "__main__":
    main()
