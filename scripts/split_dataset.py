# -*- coding: utf-8 -*-
"""把 csv2npz.py 产出的窗口划分成训练/验证集。

要点：
1. **按片段(clip)划分**——同一段视频切出的多个窗口整体分到 train 或 val，绝不跨集，
   避免"同一视频的相邻窗口同时出现在训练和验证"造成的指标虚高（数据泄漏）。
2. **类别均衡**——跌倒视频短、窗口少，日常视频长、窗口多。默认在每个 split 内把
   多数类下采样到少数类，得到 1:1，让 accuracy 也有意义、模型不偏向多数类。

窗口文件名格式来自 csv2npz.py： <tag>_<clipidx>_<winidx>.npz  (tag ∈ {fall, adl})

用法:
    python scripts/split_dataset.py --src data/skeletons/real_all \
        --train data/skeletons/train --val data/skeletons/val --val-ratio 0.2
"""
import argparse
import glob
import os
import random
import shutil


def clip_map(src, tag):
    m = {}
    for f in glob.glob(os.path.join(src, f"{tag}_*.npz")):
        clip_id = os.path.basename(f).split("_")[1]   # <tag>_<clipidx>_<winidx>
        m.setdefault(clip_id, []).append(f)
    return m


def split_clips(m, val_ratio, rng):
    keys = sorted(m)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val])
    train = [f for k in keys if k not in val_keys for f in m[k]]
    val = [f for k in val_keys for f in m[k]]
    return train, val


def balance(a, b, rng):
    """把两个窗口列表下采样到等长（1:1）。"""
    n = min(len(a), len(b))
    return rng.sample(a, n), rng.sample(b, n)


def dump(files, out):
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(out, os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--train", default="data/skeletons/train")
    ap.add_argument("--val", default="data/skeletons/val")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--no-balance", action="store_true", help="不做类别均衡")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fall = clip_map(args.src, "fall")
    adl = clip_map(args.src, "adl")
    print(f"片段数: fall={len(fall)}  adl={len(adl)}")

    f_tr, f_va = split_clips(fall, args.val_ratio, rng)
    a_tr, a_va = split_clips(adl, args.val_ratio, rng)

    if not args.no_balance:
        f_tr, a_tr = balance(f_tr, a_tr, rng)
        f_va, a_va = balance(f_va, a_va, rng)

    train = f_tr + a_tr
    val = f_va + a_va
    rng.shuffle(train)
    rng.shuffle(val)
    dump(train, args.train)
    dump(val, args.val)

    print(f"训练集: {len(train)} 窗口  (fall={len(f_tr)}, adl={len(a_tr)}) -> {args.train}")
    print(f"验证集: {len(val)} 窗口  (fall={len(f_va)}, adl={len(a_va)}) -> {args.val}")


if __name__ == "__main__":
    main()
