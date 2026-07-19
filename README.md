# Edge Fall Detection · 面向低算力端侧平台的实时跌倒检测

> 华为专项赛道 —— 队伍 **Ray**（宗家锐 2024302141121）
> 端到端纯视觉方案：**YOLO-Pose 轻量骨骼提取 + Lite ST-GCN 时序判别 + 几何先验后融合 + 状态机报警**

面向居家养老 / 医疗照护场景，在海思、瑞芯微等端侧 NPU 上，用**骨骼中间表征**替代
原始像素做时序动作理解，天然规避光照/纹理域差异，同时把整套方案压到 **~3M 参数、
12MB fp32**，端到端时延远低于 100ms 红线。

---

## 1. 为什么是"两阶段 + 骨骼中间表征"

| 赛题痛点 | 本方案对策 |
|---|---|
| 动作语义混淆（快速坐下/弯腰 vs 跌倒） | 时序 ST-GCN 学习**运动轨迹**而非单帧姿态；状态机要求"下坠 + 持续倒地"才报警 |
| 端侧算力/存储瓶颈 | 骨骼是极稀疏中间表征（17×3），时序头仅 **0.16M** 参数、1~2ms；无需 3D-CNN/Video-Transformer |
| 长尾场景（微光/夜间红外/遮挡/视角） | 骨骼表征对光照/纹理不敏感；训练期红外合成 + 遮挡/视角增强；几何先验兜底 |
| 红外支持 | 单通道红外统一转 3 通道灰度；骨骼提取与色彩无关 |

## 2. 目录结构

```
edge-fall-detection/
├── configs/default.yaml         # 全流程配置与赛题红线
├── src/
│   ├── pose/extractor.py        # 阶段1: YOLO-Pose 关键点提取(+dummy自测)
│   ├── temporal/stgcn.py        # 阶段2: Lite ST-GCN 时序判别头
│   ├── temporal/tcn.py          # 备选: 纯 1D 因果卷积(NPU 更友好)
│   ├── features/geometric.py    # 几何先验(躯干角/宽高比/下坠速度)
│   ├── postprocess/temporal_filter.py  # 状态机 + 滑动投票
│   ├── augment/ir_augment.py    # 红外/微光/遮挡/视角 数据增强
│   ├── datasets/skeleton_dataset.py    # 骨骼序列数据集(+合成)
│   └── pipeline.py              # 端到端 FallDetector
├── scripts/
│   ├── extract_skeletons.py     # 视频 -> 骨骼 npz
│   ├── train_temporal.py        # 训练时序头
│   ├── run_demo.py              # 视频/摄像头/合成 演示
│   ├── export_onnx.py           # 导出 ONNX(端侧部署)
│   └── benchmark.py             # 合规&性能核对
├── tests/test_smoke.py          # 冒烟测试(无需数据/权重/GPU)
└── docs/SOLUTION.md             # 端到端算法方案文档
```

## 3. 快速开始

```bash
conda activate yolo        # 需 torch + ultralytics

# ① 冒烟测试(无需任何数据/权重)
python tests/test_smoke.py

# ② 合规 & 性能核对(对照赛题红线)
python scripts/benchmark.py

# ③ 端到端演示 —— 合成一段"站立->跌倒"序列, 验证报警链路
python scripts/run_demo.py --synthetic
#    真实视频:
python scripts/run_demo.py --source your_video.mp4 --save out.mp4

# ④ 训练时序头(无真实数据时用合成样本跑通)
python scripts/train_temporal.py --synthetic 1536 --epochs 8
#    真实数据(先抽骨骼):
python scripts/extract_skeletons.py --videos data/raw --out data/skeletons/train
python scripts/train_temporal.py --data data/skeletons/train

# ⑤ 导出端侧 ONNX
python scripts/export_onnx.py --stage temporal --ckpt checkpoints/stgcn_fall.pt
python scripts/export_onnx.py --stage pose      --weights yolo11n-pose.pt
```

## 4. 实测结果（本仓库已跑通）

环境：Ubuntu 22.04 · RTX 5070 Ti · torch 2.11+cu128 · ultralytics 8.4.31

| 指标 | 阶段1 YOLO11n-Pose | 阶段2 Lite ST-GCN | 合计 | 赛题红线 |
|---|---|---|---|---|
| 参数量 | 2.87 M | 0.16 M | **3.03 M** | ≤ 20 M ✅ |
| fp32 权重 | 11.5 MB | 0.63 MB | **12.1 MB** | ≤ 80 MB ✅ |
| 单帧时延(GPU 参考) | 7.4 ms | 1.9 ms | **9.3 ms** | ≤ 100 ms ✅ |
| ONNX 产物 | — | 0.64 MB | — | NPU ≤ 20MB ✅ |

> 桌面 GPU 时延仅供比例参考；端侧 NPU（海思/瑞芯微）真实时延需目标板量化后实测。
> 全部两阶段合计参数仅占 20M 预算的 ~15%，为后续换用更强姿态主干留足空间。

## 5. 参考

- 数据集：[omnifall](https://huggingface.co/datasets/simplexsigil2/omnifall) · [fall-video-dataset](https://www.kaggle.com/datasets/payutch/fall-video-dataset)
- 文献：LFD-YOLO / BMR-YOLO / YOLO-fall（详见 `docs/SOLUTION.md`）

详细设计思路、可行性/先进性分析、部署链路见 **[docs/SOLUTION.md](docs/SOLUTION.md)**。
