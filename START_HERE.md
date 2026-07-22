# 从今天到初赛提交 —— 你只看这一个文件

> 队伍 Ray（宗家锐）· 华为专项赛道「面向低算力端侧平台基于视觉的实时跌倒检测」
> 初赛截止：**7 月 31 日**。初赛只要求交**解题文档（必选 PPT）**，代码/视频可选加分。
> 这个文件是「操作手册」：告诉你现在站在哪、每一步敲什么命令、按什么顺序做。
> 参考性的设计细节看 [docs/SOLUTION.md](docs/SOLUTION.md)，数据说明看 [docs/DATA.md](docs/DATA.md)。

---

## 0. 一句话理解你的项目

你做的是一个**两阶段纯视觉跌倒检测器**，思路是「先把每一帧的人压成 17 个骨骼点，再在骨骼序列上判断有没有跌倒」。
为什么这么做？因为骨骼点又小又稳（不受光照/红外/肤色/背景影响），刚好命中赛题三大痛点。

```
摄像头一帧图像
   │
[阶段1] YOLO11n-Pose  ──►  一个人框 + 17 个关键点(x,y,置信度)   ← src/pose/extractor.py
   │                        (2.87M 参数，端侧能跑)
   │  把最近 30 帧(约1秒)攒成一个"动作片段"
   ▼
[阶段2] 三路并联判断这段动作是不是跌倒：
   ├─ Lite ST-GCN 时序网络  给出"跌倒概率"        ← src/temporal/stgcn.py (0.16M 参数)
   ├─ 几何先验(躯干角/宽高比/下坠速度) 给出"跌倒分" ← src/features/geometric.py
   └─ 融合(0.6:0.4) → 状态机：先下坠+持续倒地才报警 ← src/postprocess/temporal_filter.py
   ▼
本地报警(不上传云端，保护隐私)                      ← src/pipeline.py 把上面全串起来
```

**记住三个数（PPT 合规分靠它）**：总参数 3.03M（红线≤20M）、fp32 权重 12.1MB（红线≤80MB）、时序头 ONNX 0.64MB（NPU≤20MB）。这些**已经达标且已跑通**。

---

## 1. 你现在的真实状态（进度已经很靠前了）

| 东西 | 状态 |
|---|---|
| 代码框架 | ✅ 全部跑通（冒烟测试 7/7 通过） |
| 合规三个数（参数/权重/时延） | ✅ 已达标，可直接写进 PPT |
| 设计文档 SOLUTION.md | ✅ 已含真实数据结果与消融（第五节） |
| **真实数据验证指标** | ✅ 已完成：Kaggle 真实骨骼，阶段2 窗口级 acc 0.844/召回 0.840 |
| **事件级消融表** | ✅ 已完成：证明几何融合+状态机降误报（见 `reports/ablation.json`） |
| PPT | ✅ `华为跌倒检测-方案答辩.pptx`（含真实数据消融页 + 端到端 demo 截图页，17 页） |
| **端到端 demo** | ✅ 已完成：16 段真实视频跑通完整链路，跌倒 7/8 报警、日常 1/8 误报，标注视频在 `reports/demo/` |

**结论**：合规 + 方案 + 消融 + **端到端真实视频 demo** 全部到位，**可以提交**。
剩余可选：接入红外增强跑长尾子集、录一段合并的 demo 展示视频。

> 下面第 4~6 节记录的是"从零拿数据到出指标"的完整流程（已跑通一遍）。想扩量重跑或换数据时照它做即可。
> 真实数据流程用到的脚本：`csv2npz.py`（CSV→窗口）、`prepare_real.py`（选片+划分）、`ablation.py`（事件级消融）。

> ⚠️ 重要提醒：这台机器是 **CPU（无独显）**。README 里写的 "RTX 5070 Ti" 是另一台机器/理想环境的说法。
> CPU 上抽骨骼会慢，但**只处理一个小验证集（几十段短视频）完全没问题**，不用租 GPU。

---

## 2. 对照评分维度，看清每一分从哪来

| 维度 | 分值 | 你现在 | 要补什么 |
|---|---|---|---|
| 方案合规与基础性能 | 30 | 参数/权重达标✅；红外靠灰度复制✅ | 用真实视频证明"能区分跌倒 vs 坐/弯腰"，别只有合成 |
| 技术方案设计与针对性优化 | 35 | SOLUTION.md 已覆盖三大痛点✅ | PPT 里把"两阶段+状态机+数据增强"讲清楚即可 |
| 方案创新性与先进性 | 20 | "骨骼中间表征解耦 + 三级融合"是亮点✅ | 对比 LFD-YOLO/YOLO-fall，突出你有显式时序+事件逻辑 |
| 落地实用性与文档完整度 | 15 | 部署链路已写✅ | **一段 demo 视频 + 一张消融表**，这里最容易丢分 |

所以你接下来的全部工作，就是围绕**"搞到一点真实视频 → 跑出真实指标 → 做消融表 → 出 demo → 填进 PPT"**。

---

## 3. 每次开工前的两条命令（背下来）

```bash
wsl                                   # 进 WSL（如果你在 Windows 终端）
cd ~/projects/edge-fall-detectioin
conda activate yolo                   # 激活环境，之后所有 python 命令都在这个环境里
```

先随时可以自检链路没坏：

```bash
python tests/test_smoke.py            # 应输出"全部 7 项冒烟测试通过 ✓"
python scripts/benchmark.py           # 核对参数/权重/时延红线
```

---

## 4. 主线任务：拿到真实数据 → 真实指标（按顺序做）

### 步骤 A｜搞到几十段视频（二选一，推荐 A2 更快）

**A1. 用公开数据集 Kaggle（需要 Kaggle 账号）**
1. 注册 kaggle.com → 头像 → Settings → API → Create New Token，下载 `kaggle.json`
2. 在 WSL 里：
   ```bash
   mkdir -p ~/.kaggle && cp /mnt/c/Users/宗家锐/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   pip install kaggle
   kaggle datasets download -d payutch/fall-video-dataset -p data/downloads --unzip
   ```
3. 从下载结果里**挑一小部分**（别全用）。

**A2. 自己/同学演一批（最快，不用账号，还算"自采数据"是加分项）**
- 用手机横屏拍，每段 3~8 秒即可：
  - **跌倒类 20~40 段**：往床垫/沙发上假摔（前扑、侧倒、后仰、慢慢滑倒各来几段，注意安全）
  - **日常类 40~80 段**：快速坐下、弯腰捡东西、下蹲、躺下、走过镜头（这些是最容易误报的"强负样本"，一定要拍够）
  - 有条件再补 5~10 段：拉窗帘的微光、有人挡一半的遮挡、从侧面拍的视角变化
- 拍完把视频拷进 WSL 并按类别分好文件夹（见步骤 B）。

> 目标不是数据多，而是**均衡 + 覆盖易混动作**。这正是赛题最看重的"区分跌倒 vs 坐/弯腰"。

### 步骤 B｜把视频摆成规定目录

```bash
mkdir -p data/raw/fall data/raw/adl
# 跌倒视频放 data/raw/fall/ ，日常动作放 data/raw/adl/
# 从 Windows 拷贝示例（把路径换成你的）：
# cp /mnt/c/Users/宗家锐/Downloads/我的跌倒视频/*.mp4 data/raw/fall/
# cp /mnt/c/Users/宗家锐/Downloads/我的日常视频/*.mp4 data/raw/adl/
ls data/raw/fall | wc -l && ls data/raw/adl | wc -l   # 看看各有几段
```

### 步骤 C｜抽骨骼（把视频变成模型能吃的 .npz）

```bash
python scripts/extract_skeletons.py --videos data/raw --out data/skeletons/all
```
CPU 上每段视频几秒~几十秒，几十段跑完喝杯水的功夫。跑完 `data/skeletons/all/` 里会是一堆 npz。

### 步骤 D｜分出验证集（约 20% 拿来评测）

```bash
mkdir -p data/skeletons/train data/skeletons/val
# 简单粗暴：每 5 个挑 1 个进验证集，其余进训练集
i=0; for f in data/skeletons/all/*.npz; do
  if [ $((i % 5)) -eq 0 ]; then cp "$f" data/skeletons/val/; else cp "$f" data/skeletons/train/; fi
  i=$((i+1)); done
ls data/skeletons/train | wc -l && ls data/skeletons/val | wc -l
```

### 步骤 E｜训练时序头 + 真实评测

```bash
python scripts/train_temporal.py --data data/skeletons/train --val data/skeletons/val --epochs 15
python scripts/evaluate_temporal.py --data data/skeletons/val --ckpt checkpoints/stgcn_fall.pt --out reports/eval_real.json
```
`reports/eval_real.json` 里的 **accuracy / recall / precision / f1 / 误报率** 就是你能**写进 PPT 的真实指标**了。

> 💡 初赛**不要从头训练 YOLO-Pose**（阶段1）。直接用预训练的 `yolo11n-pose.pt`，只训练上面这个轻量时序头（阶段2）。把时间花在真实验证和 PPT 证据上，而不是重训主干——CPU 也训不动主干。

### 步骤 F｜出一段 demo 视频（落地分靠它）

```bash
# 挑一段跌倒视频 + 一两段日常动作，看报警对不对
python scripts/run_demo.py --source data/raw/fall/某段.mp4 --ckpt checkpoints/stgcn_fall.pt --save reports/demo_fall.mp4
python scripts/run_demo.py --source data/raw/adl/某段坐下.mp4 --ckpt checkpoints/stgcn_fall.pt --save reports/demo_sit.mp4
```
目标画面：跌倒那段最后弹出报警；坐下/弯腰那段**不报警**。这两段对比截图放进 PPT 极加分。

---

## 5. 加分项：消融表（证明"状态机能降误报"）

在验证集上分别关掉某个模块跑一遍，填成一张表（这是拿"创新性/落地"分的关键证据）：

| 配置 | 召回率 | 精确率 | 误报率 |
|---|---|---|---|
| 仅 ST-GCN | | | |
| 仅 几何先验 | | | |
| 融合(ST-GCN+几何) | | | |
| 融合 + 状态机（完整方案） | | | |

> 消融的具体开关（改 `configs/default.yaml` 里的 `stgcn_weight/geo_weight` 与后处理参数，或写个小评测脚本）我可以帮你加。**做完步骤 A~E 有真实数据后再回来找我做这一步**。

---

## 6. 最后把证据填进 PPT / 文档

真实数据一出来，PPT 里补/替换这几页：
1. **数据划分表**：跌倒 X 段、日常 Y 段、微光/遮挡各几段（证明覆盖长尾）
2. **真实指标表**：用 `reports/eval_real.json` 的数字，**标注"真实验证集"**，别再用合成的 1.0
3. **消融表**：第 5 节那张，证明状态机降误报
4. **demo 截图**：跌倒报警 + 坐下不报警的对比
5. 合规三个数（3.03M / 12.1MB / 0.64MB）——这个现在就能写

> ⚠️ 合成数据（`data/skeletons/sample`）跑出来的 accuracy=1.0 **只能标成"工程链路自测"**，绝不能写成真实准确率，评委一眼看穿。

---

## 7. 到 7/31 的时间表（有 CPU、无 GPU 也够）

| 时间 | 做什么 |
|---|---|
| 第 1~2 天 | 步骤 A：搞到 60~120 段视频（自己演最快） |
| 第 3 天 | 步骤 B~D：摆目录、抽骨骼、分训练/验证集 |
| 第 4 天 | 步骤 E：训练 + 真实评测，拿到真实指标 |
| 第 5 天 | 步骤 F + 第 5 节：出 demo 视频 + 消融表（这步找我） |
| 第 6~8 天 | 第 6 节：真实证据填进 PPT，润色文档 |
| 第 9~11 天 | 检查合规、通读文档、留缓冲 |

---

## 8. 卡住了怎么办

- 命令报错 → 把**完整报错**贴给我。
- 不知道下一步 → 回到本文件第 4 节，看你做到哪一步了。
- 数据实在搞不到 → 告诉我，我帮你评估用公开数据集子集或纯自采的最小可行方案。

**你现在就差第一步：搞到几十段真实视频。其余链路我已经帮你验证能跑通了。**
