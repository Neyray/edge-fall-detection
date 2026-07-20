# 数据准备参考

> 本文件只讲**数据放哪、什么格式、去哪下**。
> "先做什么、后做什么"的完整操作流程见根目录 **[START_HERE.md](../START_HERE.md)** 第 4 节。

视频文件不随仓库提供。`data/` 默认被 git 忽略，大文件不要提交进仓库。

## 1. 目录格式（约定）

原始视频按类别分两个文件夹：

```text
data/raw/
  fall/*.mp4     # 跌倒视频
  adl/*.mp4      # 日常动作视频（坐下/弯腰/下蹲/躺下/走动等）
```

抽骨骼后得到滑动窗口 `.npz`，训练/验证分开放：

```text
data/skeletons/
  train/*.npz
  val/*.npz
  sample/*.npz   # make_synthetic_skeletons.py 生成，仅链路自测，非真实数据
```

## 2. 初赛需要多少数据

一份**小而均衡**的验证集即可，不必下载全量：

- 跌倒：20~50 段
- 日常动作强负样本：40~100 段（快速坐下、弯腰捡物、下蹲、躺下、走过镜头）
- 有条件再补 5~10 段：微光、遮挡、侧视角

> 关键不是数量多，而是**覆盖易混动作**——这正是赛题最看重的"区分跌倒 vs 坐/弯腰"。

## 3. 公开数据集怎么下

### Kaggle · fall-video-dataset
需要 Kaggle 账号 token（`~/.kaggle/kaggle.json`）：

```bash
chmod 600 ~/.kaggle/kaggle.json
conda activate yolo
pip install kaggle
kaggle datasets download -d payutch/fall-video-dataset -p data/downloads --unzip
find data/downloads -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' \)
```
下完从结果里**挑一小部分均衡样本**，复制到 `data/raw/fall` 与 `data/raw/adl`。

### HuggingFace · simplexsigil2/omnifall
适合查看标签和元数据；完整视频体积大，**初赛不要一开始就拉全量**，先把本地链路跑顺再按需下子集。

## 4. 自采数据（推荐，最快且算加分项）

手机横屏拍 3~8 秒短片，按上面第 2 节的数量和类别分好，直接放进 `data/raw/fall` 与 `data/raw/adl` 即可，无需账号。假摔请往床垫/沙发上做，注意安全。

## 5. 合成数据的定位

`make_synthetic_skeletons.py` 生成的是程序模拟骨骼，**只用于验证训练/评测链路能不能跑通**。它的指标（常见 accuracy=1.0）**不能写成真实精度**，PPT 里只能标注为"工程链路自测"。真实效果必须来自公开视频或自采视频验证集。
