# -*- coding: utf-8 -*-
"""冒烟测试：无需真实数据 / 权重 / GPU 即可验证核心逻辑跑通并满足赛题约束。

    python -m pytest tests/ -v      或    python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.temporal.stgcn import LiteSTGCN, count_parameters
from src.temporal.tcn import LiteTCN
from src.pipeline import FallDetector
from src.postprocess.temporal_filter import FallStateMachine, State
from src.features.geometric import geometric_fall_score


def test_stgcn_shape_and_budget():
    net = LiteSTGCN(num_class=2)
    out = net(torch.randn(4, 3, 30, 17))
    assert out.shape == (4, 2)
    params_m = count_parameters(net) / 1e6
    assert params_m < 1.0, f"ST-GCN 参数 {params_m:.3f}M 过大"
    print(f"[ok] ST-GCN 输出 {tuple(out.shape)}, 参数 {params_m:.3f}M")


def test_tcn_shape():
    out = LiteTCN()(torch.randn(2, 3, 30, 17))
    assert out.shape == (2, 2)
    print("[ok] TCN 输出", tuple(out.shape))


def test_total_param_budget():
    """两阶段总参数须 <= 20M（YOLO11n-pose 标称 ~2.9M）。"""
    stgcn_m = count_parameters(LiteSTGCN()) / 1e6
    total = stgcn_m + 2.9
    assert total <= 20.0, f"总参数 {total:.2f}M 超过 20M 红线"
    print(f"[ok] 两阶段总参数 ~{total:.2f}M <= 20M")


def test_geometric_score():
    # 构造"倒地"骨骼：宽高比>1、躯干水平
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9
    kp[5] = [100, 200, 0.9]; kp[6] = [140, 205, 0.9]   # 肩
    kp[11] = [200, 210, 0.9]; kp[12] = [240, 212, 0.9]  # 髋（水平躯干）
    bbox = np.array([90, 190, 260, 230], dtype=np.float32)  # 宽>高
    g = geometric_fall_score([kp] * 5, [bbox] * 5)
    assert g["aspect_ratio"] > 1.0
    assert g["score"] > 0.3
    print(f"[ok] 几何跌倒分 {g['score']:.2f}, 宽高比 {g['aspect_ratio']:.2f}")


def test_state_machine_triggers():
    fsm = FallStateMachine(fps=30, prob_thr=0.6, lie_seconds=0.3)  # 9 帧倒地即确认
    fired = False
    for _ in range(40):
        out = fsm.update(0.9, is_lying=True)
        fired = fired or out["newly_triggered"]
    assert fired and fsm.state == State.FALLEN
    print("[ok] 状态机在持续倒地后触发报警")


def test_state_machine_ignores_transient():
    """快速坐下：短暂高分但不维持倒地 -> 不应报警。"""
    fsm = FallStateMachine(fps=30, prob_thr=0.6, lie_seconds=1.0)
    for _ in range(5):
        fsm.update(0.9, is_lying=True)     # 短暂
    for _ in range(20):
        fsm.update(0.1, is_lying=False)    # 随即恢复站立
    assert not fsm.alarm
    print("[ok] 瞬时动作(快速坐下)未误报")


def test_state_machine_requires_drop_for_static_lying():
    """真实管线传入竖直速度时，静态躺卧不应被当成跌倒。"""
    fsm = FallStateMachine(fps=30, prob_thr=0.6, lie_seconds=0.2)
    for _ in range(60):
        fsm.update(0.9, is_lying=True, vy=0.0)
    assert not fsm.alarm and fsm.state == State.NORMAL
    print("[ok] 静态躺卧未误报（缺少下坠证据）")


def test_state_machine_drop_then_lying_triggers():
    """出现下坠后持续倒地，应完成正常报警链路。"""
    fsm = FallStateMachine(fps=30, prob_thr=0.6, lie_seconds=0.2)
    fsm.update(0.9, is_lying=True, vy=0.6)
    fired = False
    for _ in range(20):
        out = fsm.update(0.9, is_lying=True, vy=0.0)
        fired = fired or out["newly_triggered"]
    assert fired and fsm.state == State.FALLEN
    print("[ok] 下坠后持续倒地触发报警")


def test_pipeline_end_to_end():
    det = FallDetector(dummy_pose=True, window=30, fps=30.0)
    for _ in range(60):
        res = det.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert isinstance(res, list) and "state" in res[0]
    print("[ok] 端到端流水线跑通, 输出字段:", list(res[0].keys()))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n全部 {len(fns)} 项冒烟测试通过")
