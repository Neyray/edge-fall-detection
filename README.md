# Edge Fall Detection · 面向低算力端侧平台的实时跌倒检测

> 华为专项赛道 —— 队伍 Ray（宗家锐 2024302141121）
> 纯视觉方案：YOLO-Pose 轻量骨骼提取 + Lite ST-GCN 时序判别 + 几何先验后融合 + 状态机报警

面向居家养老 / 医疗照护场景，在海思、瑞芯微等端侧 NPU 上，用骨骼中间表征替代原始像素做
时序动作理解，规避光照/纹理域差异，同时把整套方案压到约 3M 参数、12MB fp32，
桌面 GPU 参考时延为 9.3 ms；端侧 NPU 时延与整机存储占用需在目标板量化后实测。

方案设计、实测验证与部署细节见随附的《端到端算法解决方案文档》（PDF）。

## 1. 方案概述

先用轻量姿态网络把每帧压成 17 点骨骼（与光照/纹理/背景解耦的稀疏中间表征），
再在骨骼序列上做廉价时空建模判断跌倒。

```
摄像头一帧
  │
[阶段1] YOLO11n-Pose → 人框 + 17关键点(x,y,conf)      src/pose/extractor.py   (2.87M)
  │  按 track_id 攒最近 30 帧(≈1s)成滑动窗口
  ▼
[阶段2] 三路并联：
  ├ Lite ST-GCN 时序网络 → 跌倒概率                   src/temporal/stgcn.py   (0.16M)
  ├ 几何先验(躯干角/宽高比/下坠速度) → 跌倒分          src/features/geometric.py
  └ 融合(0.6:0.4) + 状态机(先下坠+持续倒地才报警)      src/postprocess/temporal_filter.py
  ▼
本地实时报警（视频不出端）                            src/pipeline.py
```

## 2. 目录结构

```
edge-fall-detection/
├── README.md                    # 本文件
├── requirements.txt             # 依赖
├── configs/default.yaml         # 全流程配置与赛题红线
├── src/
│   ├── pose/extractor.py        # 阶段1: YOLO-Pose 关键点提取(+dummy自测)
│   ├── temporal/stgcn.py        # 阶段2: Lite ST-GCN 时序判别头
│   ├── temporal/tcn.py          # 备选: 纯 1D 因果卷积(NPU 更友好)
│   ├── features/geometric.py    # 几何先验(躯干角/宽高比/下坠速度)
│   ├── postprocess/temporal_filter.py  # 状态机 + 滑动投票
│   ├── augment/ir_augment.py    # 类红外/微光/遮挡/视角 合成退化
│   ├── datasets/skeleton_dataset.py    # 骨骼序列数据集(+合成)
│   └── pipeline.py              # 端到端 FallDetector
├── scripts/
│   ├── csv2npz.py               # 骨骼 CSV -> 训练窗口
│   ├── prepare_real.py          # 选片 + 按片段划分 + 类别均衡
│   ├── train_temporal.py        # 训练时序头
│   ├── evaluate_temporal.py     # 窗口级评测(输出指标 JSON)
│   ├── ablation.py              # 事件级消融
│   ├── eval_longtail.py         # 长尾(类红外/微光/遮挡)鲁棒性评测
│   ├── build_ir_dataset.py      # 构建退化域微调数据集
│   ├── demo_video.py            # 端到端视频 demo(叠加骨骼/状态/报警)
│   ├── extract_skeletons.py     # 视频 -> 骨骼 npz
│   ├── run_demo.py              # 视频/摄像头/合成 演示
│   ├── export_onnx.py           # 导出 ONNX(端侧部署)
│   ├── make_synthetic_skeletons.py  # 生成合成骨骼(仅链路自测)
│   └── benchmark.py             # 合规与性能核对
├── checkpoints/                 # 训练权重(stgcn_fall.pt 等)
├── reports/                     # 各项评测结果 JSON
└── tests/test_smoke.py          # 冒烟测试(无需数据/权重/GPU)
```

## 3. 视频与数据来源（提交必读）

本项目使用的真实视频、骨骼 CSV、Demo 视频及长尾评测视频均来自同一个公开数据集：

- **数据集名称**：Fall Video Dataset（发布者 `payutch`）
- **官方下载页**：<https://www.kaggle.com/datasets/payutch/fall-video-dataset>
- **原始内容**：`Fall` / `No_Fall` 视频，以及数据集随附的 COCO-17 `Keypoints_CSV`
- **用途限制**：仅用于本次竞赛与研究验证；请从上述官方页面自行获取数据并遵守页面所列许可/使用条款。提交包不转发完整原始数据集。

数据划分和证据边界如下：

1. **阶段 2 训练/验证**：每类按固定随机种子 `seed=0` 选取 3000 个视频片段，再按片段划分为每类 2400 个训练片段与 600 个验证片段；滑窗后得到 16632 个训练窗口、4188 个验证窗口。同一视频片段不跨训练集和验证集。
2. **事件级评测**：`reports/ablation_big.json` 实际包含 1199 段有效留出片段（599 跌倒、600 日常），以该文件为准。
3. **端到端 Demo**：14 段原始视频同样来自上述 Kaggle 数据集，均未参与模型训练；完整文件名及剔除记录见 [`reports/demo_clip_list.md`](reports/demo_clip_list.md)。最初抽取的 `20240912_101626.mp4`、`B_D_0002.mp4` 因命中训练集，已从 Demo 评测中剔除。
4. **长尾评测**：60 段留出视频（30 跌倒、30 日常）仍来自上述可见光数据集；“类红外 / 微光 / 遮挡”是在这些可见光视频上施加的**合成图像退化**，并非真实红外相机采集，不能作为真实红外性能结论。
5. **微调素材**：`data/finetune_videos` 也抽样自上述 Kaggle 数据集；`yolo11n-pose-ir.pt` 是一次伪标签微调的负结果，未用于最终提交模型。
6. **合成内容**：`data/skeletons/sample` 及 `--synthetic` 只用于验证代码链路和报警状态机，不用于报告主精度。

简言之：主指标的真实来源是 Kaggle 可见光数据；合成部分仅包括退化增强和骨骼自测，两者在结果中分开标注。

为控制提交体积，精简提交包只附两段带叠加结果的代表视频：`demo_20240912_101331.mp4`（跌倒报警）和 `demo_B_D_0004.mp4`（日常活动不报警）。其余 12 段结果仍记录在 `reports/demo_e2e.json` 与清单中，原始视频请从数据集官方页面获取。

## 4. 环境与自测（无需任何数据/权重）

```bash
conda activate yolo          # 需 torch + ultralytics
python tests/test_smoke.py   # 冒烟测试：核对模块/状态机/几何规则，应输出「全部 9 项通过」
python scripts/benchmark.py  # 对照赛题红线核对参数量/权重/时延
python scripts/run_demo.py --synthetic   # 合成一段「站立->跌倒」，验证报警链路
```

真实视频的训练/评测/demo 复现命令见解决方案文档第五节。
合成数据（`data/skeletons/sample`）跑出的指标仅用于工程链路自测，不能当作真实精度。

## 5. 合规与性能

| 指标 | 阶段1 YOLO11n-Pose | 阶段2 Lite ST-GCN | 合计 | 赛题红线 |
|---|---|---|---|---|
| 参数量 | 2.87 M | 0.16 M | 3.03 M | ≤ 20 M（达标） |
| fp32 权重 | 11.5 MB | 0.63 MB | 12.1 MB | ≤ 80 MB（达标） |
| 单帧时延（桌面 GPU 参考） | 7.4 ms | 1.9 ms | 9.3 ms | 端侧 ≤ 100 ms（待板测） |
| ONNX 产物 | — | 0.64 MB | — | 整机 NPU ≤ 20 MB（待转换与板测） |

> 参数量与 fp32 权重大小已核对；时延为 RTX 4090 桌面参考值。端侧 NPU（海思/瑞芯微）真实时延、量化后精度与整机存储占用需在目标板实测。
> 两阶段合计参数仅占 20M 预算约 15%，为升级更强姿态主干留足空间。

## 5.1 真实数据检测效果（Kaggle 骨骼验证集）

在 `payutch/fall-video-dataset` 的数据集随附真实骨骼上，按视频片段划分（同一片段不跨集）并做类别均衡后：

- 阶段2 时序头（窗口级）：准确率 0.853 / 召回 0.843 / 精确 0.860 / F1 0.852
- 事件级消融（1199 段有效片段）：几何融合把误报率 0.615→0.217；状态机（倒地持续 0.15 s）再降到 0.082，召回由 0.977 降至 0.860，F1 由 0.890 变为 0.886，体现可控的召回—误报权衡

完整数据划分表、事件级消融表与长尾实测见解决方案文档第五节。
相关脚本：`scripts/csv2npz.py`、`prepare_real.py`、`ablation.py`、`eval_longtail.py`。

## 6. 参考

- 主数据集与全部提交视频来源：[Fall Video Dataset](https://www.kaggle.com/datasets/payutch/fall-video-dataset)
- 调研参考数据集：[OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall)（未作为本文主指标数据来源）
- 文献：LFD-YOLO / BMR-YOLO / YOLO-fall（详见解决方案文档参考文献）
