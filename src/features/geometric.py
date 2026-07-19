# -*- coding: utf-8 -*-
"""几何跌倒先验特征。

轻量、可解释的手工特征，与 ST-GCN 时序判别结果做后融合，
用于抑制"快速坐下/弯腰/躺下"等日常动作造成的误报，并在骨骼点噪声大的
长尾场景（遮挡、夜间红外）下提供兜底判据。全部为纯 numpy 计算，零推理成本。
"""
from __future__ import annotations

import numpy as np

# COCO-17 关键点索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def _valid(kp: np.ndarray, idx, thr: float = 0.2) -> bool:
    return kp[idx, 2] >= thr


def torso_angle(kp: np.ndarray) -> float:
    """躯干与竖直方向的夹角（度）。

    站立时躯干接近竖直(≈0°)，跌倒后接近水平(≈90°)。
    kp: (17, 3) -> (x, y, score)，图像坐标 y 向下。
    """
    if not (_valid(kp, L_SHOULDER) or _valid(kp, R_SHOULDER)):
        return float("nan")
    if not (_valid(kp, L_HIP) or _valid(kp, R_HIP)):
        return float("nan")
    sh = np.nanmean([kp[L_SHOULDER, :2] if _valid(kp, L_SHOULDER) else [np.nan] * 2,
                     kp[R_SHOULDER, :2] if _valid(kp, R_SHOULDER) else [np.nan] * 2], axis=0)
    hip = np.nanmean([kp[L_HIP, :2] if _valid(kp, L_HIP) else [np.nan] * 2,
                      kp[R_HIP, :2] if _valid(kp, R_HIP) else [np.nan] * 2], axis=0)
    vec = hip - sh  # 肩->髋 的躯干向量
    angle = np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6))
    return float(angle)


def bbox_aspect_ratio(bbox) -> float:
    """人体检测框宽高比 w/h。站立 <1，倒地 >1。"""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    return float(w / h)


def centroid(kp: np.ndarray, thr: float = 0.2):
    v = kp[kp[:, 2] >= thr]
    if len(v) == 0:
        return None
    return v[:, :2].mean(axis=0)


def vertical_velocity(kp_seq, fps: float = 30.0, bbox_h: float = 1.0) -> float:
    """质心竖直下坠速度（按人体高度归一化，单位 高/秒）。

    跌倒的显著标志是短时间内质心快速下移。用最近若干帧线性拟合斜率。
    """
    ys = []
    for kp in kp_seq:
        c = centroid(kp)
        ys.append(np.nan if c is None else c[1])
    ys = np.asarray(ys, dtype=np.float32)
    mask = ~np.isnan(ys)
    if mask.sum() < 2:
        return 0.0
    idx = np.arange(len(ys))[mask]
    ys = ys[mask]
    # 一阶线性拟合斜率 (像素/帧) -> 归一化到 高/秒
    slope = np.polyfit(idx, ys, 1)[0]
    return float(slope * fps / max(bbox_h, 1e-3))


def geometric_fall_score(kp_seq, bbox_seq, fps: float = 30.0) -> dict:
    """综合几何判据，返回 0~1 的几何跌倒分及各分量（便于可解释性/调参）。"""
    kp_last = kp_seq[-1]
    bbox_last = bbox_seq[-1]
    ar = bbox_aspect_ratio(bbox_last)
    angle = torso_angle(kp_last)
    bh = max(bbox_last[3] - bbox_last[1], 1.0)
    vy = vertical_velocity(kp_seq[-int(fps // 2):] or kp_seq, fps=fps, bbox_h=bh)

    # 各分量归一化到 0~1
    s_ar = np.clip((ar - 0.8) / 0.8, 0, 1)                    # 宽高比>0.8 起算
    s_angle = 0.0 if np.isnan(angle) else np.clip((angle - 45) / 45, 0, 1)
    s_vy = np.clip(vy / 1.5, 0, 1)                             # 1.5 身高/秒 视为强下坠

    score = 0.4 * s_angle + 0.35 * s_ar + 0.25 * s_vy
    return {
        "score": float(score),
        "aspect_ratio": ar,
        "torso_angle": angle,
        "vertical_velocity": vy,
    }
