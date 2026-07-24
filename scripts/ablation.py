# -*- coding: utf-8 -*-
"""片段级(事件级)消融：在验证片段上比较四种配置的召回/精确/误报。

对每段验证视频(一个 CSV = 一段视频的逐帧骨骼)做流式处理，逐帧算出：
    stgcn_prob(时序头)、geo_score(几何先验)、fused(0.6:0.4 融合)、is_lying(是否倒地)
四种配置各自判定"这段视频是否报警"(事件级)：
    1) 仅 ST-GCN     —— 任一帧 stgcn_prob >= 阈值
    2) 仅 几何先验    —— 任一帧 geo_score  >= 阈值
    3) 融合(无状态机) —— 任一帧 fused      >= 阈值
    4) 融合 + 状态机  —— FallStateMachine 触发报警(需"下坠+持续倒地")
跌倒片段报警=正确(TP)，日常片段报警=误报(FP)。据此算 recall/precision/FPR。

这套协议正是赛题所要求的"以视频片段为单位"的评测。状态机的"持续倒地"帧数按数据集
片段时长(跌倒片段中位~56帧)调小；长监控流部署时应设回 1.5s。

用法:
    python scripts/ablation.py --val-list reports/val_clips.txt \
        --ckpt checkpoints/stgcn_fall.pt --out reports/ablation.json
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv2npz import csv_to_seq  # noqa: E402
from src.pose.extractor import normalize_keypoints  # noqa: E402
from src.temporal.stgcn import LiteSTGCN  # noqa: E402
from src.features.geometric import geometric_fall_score  # noqa: E402
from src.postprocess.temporal_filter import FallStateMachine  # noqa: E402


def frame_bbox(pts, thr=0.1):
    vis = pts[pts[:, 2] > thr]
    use = vis if len(vis) >= 2 else pts
    x1, y1 = use[:, 0].min(), use[:, 1].min()
    x2, y2 = use[:, 0].max(), use[:, 1].max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


@torch.no_grad()
def stgcn_prob(model, norm_deque, window):
    arr = np.stack(list(norm_deque), axis=0)
    if len(arr) < window:
        arr = np.concatenate([np.repeat(arr[:1], window - len(arr), axis=0), arr], axis=0)
    x = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
    return torch.softmax(model(x), dim=1)[0, 1].item()


def process_clip(path, model, window, thr, fps, lie_list):
    seq = csv_to_seq(path)
    if seq is None or len(seq) == 0:
        return None
    norm_dq, raw_dq, bbox_dq = deque(maxlen=window), deque(maxlen=window), deque(maxlen=window)
    # 每个"持续倒地时长"配一台状态机，同一帧流并行喂入（ST-GCN 推理只做一次）
    fsms = {ls: FallStateMachine(fps=fps, prob_thr=thr, vote_window=9,
                                 vote_ratio=0.6, lie_seconds=ls, recover_frames=15)
            for ls in lie_list}
    a_st = a_geo = a_fus = False
    a_sm = {ls: False for ls in lie_list}
    for t in range(len(seq)):
        raw = seq[t]
        bb = frame_bbox(raw)
        norm_dq.append(normalize_keypoints(raw, bb))
        raw_dq.append(raw)
        bbox_dq.append(bb)
        ps = stgcn_prob(model, norm_dq, window)
        geo = geometric_fall_score(list(raw_dq), list(bbox_dq), fps=fps)
        fused = 0.6 * ps + 0.4 * geo["score"]
        is_lying = (geo["aspect_ratio"] > 1.0) or (
            not np.isnan(geo["torso_angle"]) and geo["torso_angle"] > 55)
        a_st |= ps >= thr
        a_geo |= geo["score"] >= thr
        a_fus |= fused >= thr
        for ls, fsm in fsms.items():
            # 传入竖直下坠速度，启用「先下坠」门控（与线上 pipeline 行为一致）
            if fsm.update(fused, is_lying, vy=geo["vertical_velocity"])["newly_triggered"]:
                a_sm[ls] = True
    out = dict(stgcn=a_st, geo=a_geo, fused=a_fus)
    for ls in lie_list:
        out[f"sm@{ls}s"] = a_sm[ls]
    return out


def metrics(preds, ys):
    tp = sum(p and y for p, y in zip(preds, ys))
    fp = sum(p and not y for p, y in zip(preds, ys))
    fn = sum((not p) and y for p, y in zip(preds, ys))
    tn = sum((not p) and (not y) for p, y in zip(preds, ys))
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(recall=rec, precision=prec, fpr=fpr, f1=f1,
                tp=tp, fp=fp, tn=tn, fn=fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-list", default="reports/val_clips.txt")
    ap.add_argument("--ckpt", default="checkpoints/stgcn_fall.pt")
    ap.add_argument("--out", default="reports/ablation.json")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--lie-sweep", default="0.2,0.3,0.4,0.6",
                    help="状态机'持续倒地'时长(秒)扫描，逗号分隔")
    args = ap.parse_args()

    lie_list = [float(x) for x in args.lie_sweep.split(",")]

    model = LiteSTGCN(num_class=2).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state.get("model", state))

    clips = []
    with open(args.val_list) as f:
        for line in f:
            lb, path = line.rstrip("\n").split("\t")
            clips.append((int(lb), path))
    print(f"验证片段: {len(clips)}")

    keys = ["stgcn", "geo", "fused"] + [f"sm@{ls}s" for ls in lie_list]
    ys, res = [], {k: [] for k in keys}
    for i, (lb, path) in enumerate(clips):
        out = process_clip(path, model, args.window, args.thr, args.fps, lie_list)
        if out is None:
            continue
        ys.append(lb == 1)
        for k in keys:
            res[k].append(out[k])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clips)}")

    names = {"stgcn": "仅 ST-GCN", "geo": "仅 几何先验", "fused": "融合(无状态机)"}
    for ls in lie_list:
        names[f"sm@{ls}s"] = f"融合+状态机(倒地≥{ls}s)"
    table = {k: metrics(res[k], ys) for k in keys}

    print("\n配置                       召回    精确    误报率   F1     (TP/FP/FN/TN)")
    print("-" * 74)
    for k in keys:
        m = table[k]
        print(f"{names[k]:<22}  {m['recall']:.3f}  {m['precision']:.3f}  "
              f"{m['fpr']:.3f}   {m['f1']:.3f}  ({m['tp']}/{m['fp']}/{m['fn']}/{m['tn']})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_clips": len(ys), "thr": args.thr, "lie_sweep": lie_list,
                   "variants": {names[k]: table[k] for k in keys}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {args.out}")


if __name__ == "__main__":
    main()
