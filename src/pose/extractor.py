# -*- coding: utf-8 -*-
"""第一阶段：YOLO-Pose 人体关键点提取器。

封装 ultralytics YOLO-Pose（默认 yolo11n-pose，约 2.9M 参数），
从单帧图像中提取每个人的检测框与 COCO-17 关键点。
- 支持可见光与红外：红外多为单通道，这里统一转 3 通道灰度堆叠喂入，
  与训练时的红外增强保持一致。
- 无 GPU / 无权重时可用 dummy=True 生成结构一致的随机输出，便于流水线自测。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Person:
    bbox: np.ndarray        # (4,) x1,y1,x2,y2
    keypoints: np.ndarray   # (17, 3) x,y,score
    track_id: int = -1
    det_conf: float = 0.0


class PoseExtractor:
    def __init__(self, weights: str = "yolo11n-pose.pt", device: str = "cpu",
                 imgsz: int = 640, conf: float = 0.3, dummy: bool = False,
                 dummy_fall_at: int = 45):
        self.imgsz = imgsz
        self.conf = conf
        self.dummy = dummy
        self.device = device
        self.dummy_fall_at = dummy_fall_at   # dummy 模式：第几帧起进入"倒地"
        self._t = 0                          # dummy 模式内部帧计数
        self.model = None
        if not dummy:
            from ultralytics import YOLO
            self.model = YOLO(weights)

    @staticmethod
    def _to_3ch(frame: np.ndarray) -> np.ndarray:
        """红外/灰度单通道 -> 3 通道，保持与 YOLO 输入一致。"""
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        return frame

    def __call__(self, frame: np.ndarray):
        """返回当前帧的 Person 列表。"""
        frame = self._to_3ch(frame)
        if self.dummy:
            return self._dummy_persons(frame)

        res = self.model.track(frame, imgsz=self.imgsz, conf=self.conf,
                               persist=True, verbose=False)[0]
        persons = []
        if res.keypoints is None or res.boxes is None:
            return persons
        kps = res.keypoints.data.cpu().numpy()        # (n,17,3)
        boxes = res.boxes.xyxy.cpu().numpy()          # (n,4)
        confs = res.boxes.conf.cpu().numpy()
        ids = (res.boxes.id.int().cpu().numpy()
               if res.boxes.id is not None else -np.ones(len(boxes), dtype=int))
        for i in range(len(boxes)):
            persons.append(Person(bbox=boxes[i], keypoints=kps[i],
                                   track_id=int(ids[i]), det_conf=float(confs[i])))
        return persons

    def _dummy_persons(self, frame: np.ndarray):
        """脚本化的单人骨骼：前段站立，dummy_fall_at 帧后过渡为倒地，
        用于无摄像头/无权重时端到端演示报警链路。"""
        h, w = frame.shape[:2]
        t = self._t
        self._t += 1
        # 0 站立 -> 1 完全倒地 的过渡系数
        prog = float(np.clip((t - self.dummy_fall_at) / 10.0, 0.0, 1.0))

        cx = w * 0.5
        cy = h * (0.5 + 0.25 * prog)                 # 质心随倒地下移
        body_w = w * (0.12 + 0.28 * prog)            # 倒地后横向变宽
        body_h = h * (0.38 - 0.22 * prog)            # 倒地后纵向变矮

        kp = np.zeros((17, 3), dtype=np.float32)
        # 站立时躯干竖直(肩上髋下)，倒地后旋转到水平
        vert = np.array([0.0, -body_h * 0.4])
        horiz = np.array([body_w * 0.4, 0.0])
        torso = vert * (1 - prog) + horiz * prog
        sh = np.array([cx, cy]) - torso
        hip = np.array([cx, cy]) + torso
        kp[5, :2] = sh + [-body_w * 0.12, 0]; kp[6, :2] = sh + [body_w * 0.12, 0]
        kp[11, :2] = hip + [-body_w * 0.1, 0]; kp[12, :2] = hip + [body_w * 0.1, 0]
        kp[0, :2] = sh + [0, -body_h * 0.15]         # 头
        for j in (13, 14, 15, 16, 7, 8, 9, 10, 1, 2, 3, 4):
            kp[j, :2] = hip + np.random.randn(2) * 5
        kp[:, 2] = 0.7 + 0.2 * np.random.rand(17)

        bbox = np.array([cx - body_w * 0.6, cy - body_h * 0.6,
                         cx + body_w * 0.6, cy + body_h * 0.6], dtype=np.float32)
        return [Person(bbox=bbox, keypoints=kp, track_id=0, det_conf=0.9)]


def normalize_keypoints(kp: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """把关键点坐标按检测框归一化到 [-1,1]，消除人体在画面中的位置/尺度差异，
    这是时序模型跨机位、跨分辨率泛化的关键预处理。"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    out = kp.copy().astype(np.float32)
    out[:, 0] = (kp[:, 0] - cx) / (w / 2)
    out[:, 1] = (kp[:, 1] - cy) / (h / 2)
    # score 保留
    return out
