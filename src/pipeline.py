# -*- coding: utf-8 -*-
"""端到端跌倒检测流水线。

    帧 -> [阶段1] YOLO-Pose 关键点 -> 按人跟踪的滑动窗口
        -> [阶段2] Lite ST-GCN 时序判别 + 几何先验后融合
        -> [后处理] 状态机 + 投票 -> 报警

按 track_id 维护每个人的关键点历史，天然支持多人场景。
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch

from src.pose.extractor import PoseExtractor, normalize_keypoints
from src.temporal.stgcn import LiteSTGCN
from src.features.geometric import geometric_fall_score
from src.postprocess.temporal_filter import FallStateMachine


class PersonTrack:
    def __init__(self, window: int):
        self.norm_kp = deque(maxlen=window)   # 归一化关键点 (17,3)
        self.raw_kp = deque(maxlen=window)    # 原始关键点（几何特征用）
        self.bbox = deque(maxlen=window)
        self.fsm = None


class FallDetector:
    def __init__(self, pose_weights: str = "yolo11n-pose.pt",
                 temporal_ckpt: str | None = None, device: str = "cpu",
                 window: int = 30, fps: float = 30.0,
                 stgcn_weight: float = 0.6, geo_weight: float = 0.4,
                 dummy_pose: bool = False, fsm_kwargs: dict | None = None):
        self.device = device
        self.window = window
        self.fps = fps
        self.stgcn_weight = stgcn_weight
        self.geo_weight = geo_weight
        self.fsm_kwargs = fsm_kwargs or {}

        self.pose = PoseExtractor(pose_weights, device=device, dummy=dummy_pose)
        self.temporal = LiteSTGCN(num_class=2).to(device).eval()
        if temporal_ckpt:
            state = torch.load(temporal_ckpt, map_location=device)
            self.temporal.load_state_dict(state.get("model", state))

        self.tracks: dict[int, PersonTrack] = defaultdict(lambda: PersonTrack(window))

    @torch.no_grad()
    def _temporal_prob(self, norm_seq) -> float:
        """把 (T,17,3) 序列送入 ST-GCN，返回跌倒类概率。"""
        if len(norm_seq) < 2:
            return 0.0
        arr = np.stack(list(norm_seq), axis=0)            # (T,17,3)
        # 左侧 pad 到窗口长度
        if len(arr) < self.window:
            pad = np.repeat(arr[:1], self.window - len(arr), axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        x = torch.from_numpy(arr).float()                 # (T,17,3)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1,3,T,17)
        logits = self.temporal(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        return float(prob)

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """处理单帧，返回每个人的检测结果列表。"""
        persons = self.pose(frame)
        results = []
        for p in persons:
            tid = p.track_id if p.track_id >= 0 else 0
            tr = self.tracks[tid]
            if tr.fsm is None:
                tr.fsm = FallStateMachine(fps=self.fps, **self.fsm_kwargs)

            norm = normalize_keypoints(p.keypoints, p.bbox)
            tr.norm_kp.append(norm)
            tr.raw_kp.append(p.keypoints)
            tr.bbox.append(p.bbox)

            stgcn_prob = self._temporal_prob(tr.norm_kp)
            geo = geometric_fall_score(list(tr.raw_kp), list(tr.bbox), fps=self.fps)
            fused = self.stgcn_weight * stgcn_prob + self.geo_weight * geo["score"]

            is_lying = (geo["aspect_ratio"] > 1.0) or (
                not np.isnan(geo["torso_angle"]) and geo["torso_angle"] > 55)
            fsm_out = tr.fsm.update(fused, is_lying)

            results.append({
                "track_id": tid,
                "bbox": p.bbox.tolist(),
                "keypoints": p.keypoints.tolist(),
                "stgcn_prob": stgcn_prob,
                "geo_score": geo["score"],
                "fused_prob": fused,
                "torso_angle": geo["torso_angle"],
                "aspect_ratio": geo["aspect_ratio"],
                **fsm_out,
            })
        return results

    def reset(self):
        self.tracks.clear()
