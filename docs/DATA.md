# 数据准备说明

赛题方案文档里给的是公开数据集参考链接，但视频文件没有随仓库一起提供。公开视频、
自采视频和抽取后的骨骼片段都建议放在 `data/` 下；该目录默认被 git 忽略，避免把大文件提交进仓库。

## 初赛推荐路径

初赛不需要一上来下载和处理全量数据，先做一份小而均衡的验证集更划算：

1. 先准备少量样本：
   - 跌倒视频：20 到 50 段；
   - 日常动作负样本：40 到 100 段，包括快速坐下、弯腰捡物、蹲下、躺下、走过镜头；
   - 如果方便，再补 5 到 10 段微光、遮挡、侧视角样本。
2. 按下面目录放视频：

```text
data/raw/
  fall/*.mp4
  adl/*.mp4
```

3. 抽取骨骼窗口：

```bash
conda activate yolo
python scripts/extract_skeletons.py --videos data/raw --out data/skeletons/train
```

4. 划分一部分窗口或视频作为验证集，然后训练和评测：

```bash
python scripts/train_temporal.py --data data/skeletons/train --val data/skeletons/val --epochs 15
python scripts/evaluate_temporal.py --data data/skeletons/val --ckpt checkpoints/stgcn_fall.pt
python scripts/export_onnx.py --stage temporal --ckpt checkpoints/stgcn_fall.pt
```

## 公开数据集说明

### Kaggle fall-video-dataset

Kaggle 通常需要账号 token。把 `kaggle.json` 放到 `~/.kaggle/kaggle.json` 后执行：

```bash
chmod 600 ~/.kaggle/kaggle.json
conda activate yolo
kaggle datasets download -d payutch/fall-video-dataset -p data/downloads/kaggle --unzip
find data/downloads/kaggle -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' \)
```

然后从下载结果里挑一小部分均衡样本，复制或软链接到 `data/raw/fall` 和 `data/raw/adl`，
再运行 `extract_skeletons.py`。

### HuggingFace simplexsigil2/omnifall

HuggingFace 页面适合查看标签和元数据，完整视频体积可能较大。不要为了初赛一开始就拉全量；
先把本地训练、评测、导出链路跑顺，再按需要下载子集。

## 合成数据自测

只做工程链路自测时可以运行：

```bash
python scripts/make_synthetic_skeletons.py --out data/skeletons/sample --samples 40
python scripts/evaluate_temporal.py --data data/skeletons/sample --ckpt checkpoints/stgcn_fall.pt
```

注意：合成数据指标不能当作真实准确率写进 PPT。可以标注为“工程链路自测”，真实效果要用公开视频或自采视频验证。