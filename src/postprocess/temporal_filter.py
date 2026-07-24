# -*- coding: utf-8 -*-
"""时序后处理：有限状态机 + 滑动投票。

即使单帧/单窗口判别正确，直接报警仍会因抖动产生误报。
本模块引入"跌倒事件"状态机：
    NORMAL --(连续 K 帧高分)--> FALLING --(持续倒地 >= T_lie 秒)--> FALLEN(报警)
只有"发生下坠 + 之后长时间保持倒地"才确认为真实跌倒，
可有效区分"快速坐下/弯腰"（短暂、随后恢复站立）等日常动作。
"""
from __future__ import annotations

from collections import deque
from enum import Enum


class State(Enum):
    NORMAL = 0
    FALLING = 1
    FALLEN = 2   # 已确认跌倒 -> 触发报警


class FallStateMachine:
    """跌倒事件状态机。

    进入 FALLING 需要**同时**满足：融合概率投票通过 **且** 近期发生过"质心快速下坠"。
    只有"高分"而没有下坠（例如本来就静止躺着、在床上休息），不会被判为跌倒事件——
    这正是把"跌倒"与"躺卧姿态"区分开的关键。

    下坠判据用 `src/features/geometric.py::vertical_velocity` 算出的归一化竖直速度
    （单位：人体高/秒）。下坠是瞬时事件，之后人躺在地上速度归零，因此用
    `drop_memory` 秒的"记忆窗"保留下坠事实。
    """

    def __init__(self, fps: float = 30.0, prob_thr: float = 0.6,
                 vote_window: int = 9, vote_ratio: float = 0.6,
                 lie_seconds: float = 1.5, recover_frames: int = 15,
                 require_drop: bool = True, drop_thr: float = 0.35,
                 drop_memory: float = 1.0):
        self.fps = fps
        self.prob_thr = prob_thr
        self.vote_ratio = vote_ratio
        self.lie_frames = int(lie_seconds * fps)
        self.recover_frames = recover_frames

        self.require_drop = require_drop
        self.drop_thr = drop_thr
        self.drop_mem_frames = max(1, int(drop_memory * fps))
        self.drop_recent = 0          # >0 表示近期发生过下坠

        self.buf = deque(maxlen=vote_window)
        self.state = State.NORMAL
        self.lie_counter = 0
        self.recover_counter = 0
        self.alarm = False

    def _voted_fall(self) -> bool:
        if not self.buf:
            return False
        hit = sum(1 for p in self.buf if p >= self.prob_thr)
        return hit / len(self.buf) >= self.vote_ratio

    def _update_drop(self, vy):
        """维护"近期是否发生下坠"的记忆窗。vy 为归一化竖直速度(人体高/秒)。"""
        if vy is not None and vy >= self.drop_thr:
            self.drop_recent = self.drop_mem_frames
        elif self.drop_recent > 0:
            self.drop_recent -= 1
        # vy 未提供时不启用下坠门控（保持向后兼容；真实流水线均会传入）
        if vy is None and self.require_drop:
            return True
        return (not self.require_drop) or self.drop_recent > 0

    def update(self, fall_prob: float, is_lying: bool, vy: float | None = None) -> dict:
        """输入当前帧融合概率、"是否倒地姿态"、以及竖直下坠速度，返回状态与报警标志。

        vy: 归一化竖直速度（人体高/秒），来自几何先验。用于强制"先下坠"判据。
        """
        self.buf.append(fall_prob)
        voted = self._voted_fall()
        dropped = self._update_drop(vy)
        newly_triggered = False

        if self.state == State.NORMAL:
            # 必须「投票通过」且「近期确实发生过下坠」才进入 FALLING
            if voted and dropped:
                self.state = State.FALLING
                self.lie_counter = 0

        elif self.state == State.FALLING:
            if is_lying:
                self.lie_counter += 1
                if self.lie_counter >= self.lie_frames:
                    self.state = State.FALLEN
                    self.alarm = True
                    newly_triggered = True
            elif not voted:
                # 快速起身/未维持倒地 -> 判为日常动作，回到正常
                self.state = State.NORMAL
                self.lie_counter = 0

        elif self.state == State.FALLEN:
            if not is_lying and not voted:
                self.recover_counter += 1
                if self.recover_counter >= self.recover_frames:
                    self.state = State.NORMAL
                    self.alarm = False
                    self.recover_counter = 0
            else:
                self.recover_counter = 0

        return {
            "state": self.state.name,
            "alarm": self.alarm,
            "newly_triggered": newly_triggered,
            "voted_prob": sum(self.buf) / len(self.buf),
            "drop_recent": self.drop_recent,   # >0 表示近期检测到下坠
        }
