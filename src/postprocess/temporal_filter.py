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
    def __init__(self, fps: float = 30.0, prob_thr: float = 0.6,
                 vote_window: int = 9, vote_ratio: float = 0.6,
                 lie_seconds: float = 1.5, recover_frames: int = 15):
        self.fps = fps
        self.prob_thr = prob_thr
        self.vote_ratio = vote_ratio
        self.lie_frames = int(lie_seconds * fps)
        self.recover_frames = recover_frames

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

    def update(self, fall_prob: float, is_lying: bool) -> dict:
        """输入当前帧融合后的跌倒概率与"是否处于倒地姿态"，返回状态与报警标志。"""
        self.buf.append(fall_prob)
        voted = self._voted_fall()
        newly_triggered = False

        if self.state == State.NORMAL:
            if voted:
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
        }
