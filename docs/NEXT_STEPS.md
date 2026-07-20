# 初赛下一步建议

## 当前优先级

现在不建议大改技术路线。两阶段“骨骼提取 + 时序判别”的方案很贴合评分标准，真正缺的是证据：

1. 真实视频验证，而不是继续堆模型结构。
2. 一张小消融表：仅 ST-GCN、仅几何先验、融合、融合 + 状态机。
3. 一个可复现 demo：至少包含一段跌倒和几段容易误报的日常动作负样本。

## 优先采什么数据

初赛阶段，一份小而均衡的验证集就够用：

- 跌倒：20 到 50 段；
- 日常动作强负样本：40 到 100 段；
- 覆盖快速坐下、弯腰捡物、蹲下、躺下、走过镜头；
- 如果方便，补 5 到 10 段微光、遮挡、侧视角样本。

目录结构：

```text
data/raw/fall/*.mp4
data/raw/adl/*.mp4
```

然后运行：

```bash
conda activate yolo
python scripts/extract_skeletons.py --videos data/raw --out data/skeletons/train
python scripts/train_temporal.py --data data/skeletons/train --epochs 15
python scripts/evaluate_temporal.py --data data/skeletons/val --ckpt checkpoints/stgcn_fall.pt
```

## 是否需要租 GPU

你现在的无独显电脑可以完成：

- 用已抽取的骨骼 `.npz` 训练 ST-GCN/TCN；
- 合成链路自测；
- 评测、导出 ONNX；
- 整理 PPT 和报告证据。

CPU 不适合长期做：

- 对大量原始视频跑 YOLO-Pose 抽骨骼；
- 微调 YOLO-Pose；
- 反复生成端到端 demo 视频。

建议：**现在先不租 GPU**。先整理小样本视频，跑通目录、抽骨骼、训练、评测这条链路。
等你要批量处理公开视频或生成最终 demo 时，再短租 2 到 6 小时云 GPU 就够了。

初赛阶段不要从头训练 YOLO-Pose。直接用预训练 YOLO11n-Pose，重点训练时序头，把时间花在真实验证和 PPT 证据上。

## PPT 建议补充

把纯合成结果替换或补充为真实验证结果：

- 数据划分表：跌倒、日常动作、微光、遮挡各多少段；
- 事件级指标：召回率、精确率、误报率；
- 消融表：证明状态机能降低误报；
- 部署表：参数量、fp32 大小、ONNX 大小、CPU/GPU/NPU 实测状态。

合成数据结果请标注为“工程链路自测”，不要写成真实准确率。