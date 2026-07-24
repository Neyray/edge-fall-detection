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
        self.last_seen = 0                    # 最后一次被匹配到的帧号（用于超时清理）


def _bbox_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _bbox_diag(b):
    return max(((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5, 1e-3)


class FallDetector:
    def __init__(self, pose_weights: str = "yolo11n-pose.pt",
                 temporal_ckpt: str | None = None, device: str = "cpu",
                 window: int = 30, fps: float = 30.0,
                 stgcn_weight: float = 0.6, geo_weight: float = 0.4,
                 dummy_pose: bool = False, fsm_kwargs: dict | None = None,
                 max_track_age: float = 2.0):
        self.device = device
        self.window = window
        self.fps = fps
        self.stgcn_weight = stgcn_weight
        self.geo_weight = geo_weight
        self.fsm_kwargs = fsm_kwargs or {}
        self.max_track_age = max_track_age      # 轨迹多少秒未出现就清理

        self.pose = PoseExtractor(pose_weights, device=device, dummy=dummy_pose)
        self.temporal = LiteSTGCN(num_class=2).to(device).eval()
        if temporal_ckpt:
            state = torch.load(temporal_ckpt, map_location=device)
            self.temporal.load_state_dict(state.get("model", state))

        self.tracks: dict[int, PersonTrack] = {}
        self._frame_idx = 0        # 已处理帧计数（轨迹超时用）
        self._next_fallback_id = 10_000   # 无 track_id 时分配的合成 ID 起点

    def _assign_ids(self, persons):
        """给本帧每个人确定轨迹 ID。

        跟踪器给了 track_id 就直接用；**没给的不能统统当成 0**（否则多人骨骼历史会串味）。
        对未跟踪的检测，按质心就近匹配到本帧尚未被占用的已有轨迹；匹配不上则新开一条轨迹。
        """
        assigned = {}
        taken = set()
        untracked = []
        for i, p in enumerate(persons):
            if getattr(p, "track_id", -1) is not None and p.track_id >= 0:
                assigned[i] = int(p.track_id)
                taken.add(int(p.track_id))
            else:
                untracked.append(i)

        for i in untracked:
            b = persons[i].bbox
            cx, cy = _bbox_center(b)
            best, best_d = None, None
            for tid, tr in self.tracks.items():
                if tid in taken or not tr.bbox:
                    continue
                pcx, pcy = _bbox_center(tr.bbox[-1])
                d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 / _bbox_diag(b)
                if d < 0.6 and (best_d is None or d < best_d):   # 半个框以内视为同一人
                    best, best_d = tid, d
            if best is None:
                best = self._next_fallback_id
                self._next_fallback_id += 1
            assigned[i] = best
            taken.add(best)
        return assigned

    def _prune_tracks(self):
        """清理长时间未出现的轨迹，避免内存与状态无限累积。"""
        max_age = max(1, int(self.max_track_age * self.fps))
        for tid in [t for t, tr in self.tracks.items()
                    if self._frame_idx - tr.last_seen > max_age]:
            del self.tracks[tid]

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
        self._frame_idx += 1
        ids = self._assign_ids(persons)
        results = []
        for i, p in enumerate(persons):
            tid = ids[i]
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = PersonTrack(self.window)
            if tr.fsm is None:
                tr.fsm = FallStateMachine(fps=self.fps, **self.fsm_kwargs)
            tr.last_seen = self._frame_idx

            norm = normalize_keypoints(p.keypoints, p.bbox)
            tr.norm_kp.append(norm)
            tr.raw_kp.append(p.keypoints)
            tr.bbox.append(p.bbox)

            stgcn_prob = self._temporal_prob(tr.norm_kp)
            geo = geometric_fall_score(list(tr.raw_kp), list(tr.bbox), fps=self.fps)
            fused = self.stgcn_weight * stgcn_prob + self.geo_weight * geo["score"]

            is_lying = (geo["aspect_ratio"] > 1.0) or (
                not np.isnan(geo["torso_angle"]) and geo["torso_angle"] > 55)
            # 把竖直下坠速度交给状态机，强制「先下坠」判据
            fsm_out = tr.fsm.update(fused, is_lying, vy=geo["vertical_velocity"])

            results.append({
                "track_id": tid,
                "bbox": p.bbox.tolist(),
                "keypoints": p.keypoints.tolist(),
                "stgcn_prob": stgcn_prob,
                "geo_score": geo["score"],
                "fused_prob": fused,
                "torso_angle": geo["torso_angle"],
                "aspect_ratio": geo["aspect_ratio"],
                "vertical_velocity": geo["vertical_velocity"],
                **fsm_out,
            })
        self._prune_tracks()
        return results

    def set_fps(self, fps: float):
        """切换视频时用真实帧率重建状态（状态机的"持续N秒""下坠速度"都依赖它）。"""
        self.fps = float(fps) if fps and fps > 0 else self.fps
        self.reset()

    def reset(self):
        self.tracks.clear()
        self._frame_idx = 0
        self._next_fallback_id = 10_000
