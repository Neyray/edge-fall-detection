# Edge Fall Detection · 面向低算力端侧平台的实时跌倒检测

> 华为专项赛道 —— 队伍 **Ray**（宗家锐 2024302141121）
> 纯视觉方案：**YOLO-Pose 轻量骨骼提取 + Lite ST-GCN 时序判别 + 几何先验后融合 + 状态机报警**

面向居家养老 / 医疗照护场景，在海思、瑞芯微等端侧 NPU 上，用**骨骼中间表征**替代原始像素做
时序动作理解，天然规避光照/纹理域差异，同时把整套方案压到 **~3M 参数、12MB fp32**，端到端时延远低于 100ms 红线。

---

## 📖 三个文档，各看各的

| 想知道 | 看这个 |
|---|---|
| **我现在具体该做什么**（一步步敲命令到提交） | 👉 **[START_HERE.md](START_HERE.md)** |
| **方案怎么设计的**（答辩文档正文、痛点对策、创新点） | [docs/SOLUTION.md](docs/SOLUTION.md) |
| **数据怎么准备 + 模型怎么训练**（数据来源、划分、训练配置、复现命令） | [docs/真实数据处理流程.md](docs/真实数据处理流程.md) |

本 README 只讲**项目是什么、目录结构、怎么自测**。

---

## 1. 方案一句话

先用轻量姿态网络把每帧压成 **17 点骨骼**（与光照/纹理/背景解耦的稀疏中间表征），再在骨骼序列上做**廉价时空建模**判断跌倒。

```
摄像头一帧
  │
[阶段1] YOLO11n-Pose → 人框 + 17关键点(x,y,conf)      src/pose/extractor.py   (2.87M)
  │  按 track_id 攒最近 30 帧(≈1s)成滑动窗口
  ▼
[阶段2] 三路并联：
  ├ Lite ST-GCN 时序网络 → 跌倒概率                   src/temporal/stgcn.py   (0.16M)
  ├ 几何先验(躯干角/宽高比/下坠速度) → 跌倒分          src/features/geometric.py
  └ 融合(0.6:0.4) + 状态机(先下坠+持续倒地≥1.5s才报警) src/postprocess/temporal_filter.py
  ▼
本地实时报警（视频不出端，隐私闭环）                  src/pipeline.py
```

## 2. 目录结构

```
edge-fall-detectioin/
├── START_HERE.md                # ★ 行动清单：现在该做什么
├── README.md                    # 本文件：项目总览
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
│   ├── evaluate_temporal.py     # 评测(输出指标 JSON)
│   ├── run_demo.py              # 视频/摄像头/合成 演示
│   ├── export_onnx.py           # 导出 ONNX(端侧部署)
│   ├── make_synthetic_skeletons.py  # 生成合成骨骼(仅链路自测)
│   └── benchmark.py             # 合规&性能核对
├── docs/{SOLUTION,DATA}.md      # 方案设计 / 数据准备
└── tests/test_smoke.py          # 冒烟测试(无需数据/权重/GPU)
```

## 3. 环境与自测（无需任何数据/权重）

```bash
cd ~/projects/edge-fall-detectioin
conda activate yolo          # 需 torch + ultralytics

python tests/test_smoke.py   # 冒烟测试：核对模块/状态机/几何规则，应输出「全部 7 项通过 ✓」
python scripts/benchmark.py  # 对照赛题红线核对参数量/权重/时延
python scripts/run_demo.py --synthetic   # 合成一段「站立->跌倒」，验证报警链路
```

> 用**真实视频**训练/评测/出 demo 的完整流程 → 见 **[START_HERE.md](START_HERE.md)** 第 4 节。
> 合成数据（`data/skeletons/sample`）跑出的指标只能标注为「工程链路自测」，**不能当作真实精度**。

## 4. 合规与性能（本仓库已跑通、达标）

| 指标 | 阶段1 YOLO11n-Pose | 阶段2 Lite ST-GCN | 合计 | 赛题红线 |
|---|---|---|---|---|
| 参数量 | 2.87 M | 0.16 M | **3.03 M** | ≤ 20 M ✅ |
| fp32 权重 | 11.5 MB | 0.63 MB | **12.1 MB** | ≤ 80 MB ✅ |
| 单帧时延（GPU 参考） | 7.4 ms | 1.9 ms | **9.3 ms** | ≤ 100 ms ✅ |
| ONNX 产物 | — | 0.64 MB | — | NPU ≤ 20MB ✅ |

> 时延为桌面参考值，仅供比例说明；端侧 NPU（海思/瑞芯微）真实时延需目标板量化后实测。
> 两阶段合计参数仅占 20M 预算 ~15%，为升级更强姿态主干留足空间。

## 4.1 真实数据检测效果（Kaggle 骨骼验证集）

在 `payutch/fall-video-dataset` 真实骨骼上，按视频片段划分（零泄漏）+ 类别均衡后：

- **阶段2 时序头（窗口级）**：准确率 0.844 / 召回 0.840 / 精确 0.847 / F1 0.844
- **事件级消融**：几何融合把误报率 0.573→0.213，状态机再降到 0.12–0.15，"召回↔误报"可调

完整数据划分表与消融表见 [docs/SOLUTION.md](docs/SOLUTION.md) 第五节；复现流程见 [START_HERE.md](START_HERE.md)。
相关脚本：`scripts/csv2npz.py`（CSV→窗口）、`prepare_real.py`（选片+划分）、`ablation.py`（事件级消融）。

## 5. 参考

- 数据集：[omnifall](https://huggingface.co/datasets/simplexsigil2/omnifall) · [fall-video-dataset](https://www.kaggle.com/datasets/payutch/fall-video-dataset)
- 文献：LFD-YOLO / BMR-YOLO / YOLO-fall（详见 [docs/SOLUTION.md](docs/SOLUTION.md)）
