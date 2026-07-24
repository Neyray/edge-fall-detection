# -*- coding: utf-8 -*-
"""导出端侧部署模型 (ONNX)，面向海思 / 瑞芯微 NPU 工具链。

- 阶段1 YOLO-Pose：直接用 ultralytics 导出 ONNX（再经厂商工具转 .om/.rknn）。
- 阶段2 Lite ST-GCN：torch.onnx 导出，opset 12，算子仅 Conv/BN/MatMul/Add，
  海思 (Ascend/Hi35xx) 与瑞芯微 (RKNN) 均可直接量化。

用法:
    python scripts/export_onnx.py --stage temporal --ckpt checkpoints/stgcn_fall.pt
    python scripts/export_onnx.py --stage pose --weights yolo11n-pose.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.temporal.stgcn import LiteSTGCN, count_parameters


def export_temporal(ckpt, out, window=30, allow_random=False):
    model = LiteSTGCN(num_class=2).eval()
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        acc = state.get("acc") if isinstance(state, dict) else None
        print(f"[load] {ckpt}" + (f"  (训练时最佳验证准确率 {acc:.3f})" if acc else ""))
    elif allow_random:
        print("[warn] 未加载权重，正在导出**随机初始化**模型——仅可用于结构/尺寸检查，"
              "绝不能用于部署或作为性能证据。")
    else:
        # 静默导出随机权重会让人误以为拿到了可用模型，这里直接失败。
        raise SystemExit(
            f"错误：找不到权重文件 {ckpt!r}，拒绝导出随机初始化模型。\n"
            f"       请先训练：python scripts/train_temporal.py --data <训练集> --val <验证集>\n"
            f"       或显式加 --allow-random 仅做结构检查（导出结果不可用于部署）。"
        )
    dummy = torch.randn(1, 3, window, 17)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    export_kw = dict(
        opset_version=12,
        input_names=["skeleton"], output_names=["logits"],
        dynamic_axes={"skeleton": {0: "batch"}, "logits": {0: "batch"}},
    )
    try:
        # torch>=2.5 默认走 dynamo 导出(需 onnxscript)，端侧工具链更常用稳定的
        # TorchScript 导出，这里显式指定以减少部署依赖。
        torch.onnx.export(model, dummy, out, dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(model, dummy, out, **export_kw)
    size_mb = os.path.getsize(out) / 1e6
    print(f"[onnx] {out}  params={count_parameters(model)/1e6:.3f}M  size={size_mb:.2f}MB")
    _check(out)


def export_pose(weights, out, imgsz=640):
    from ultralytics import YOLO
    model = YOLO(weights)
    path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    print(f"[onnx] pose 导出 -> {path}")


def _check(path):
    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[onnx] 模型校验通过")
    except ImportError:
        print("[onnx] 未安装 onnx，跳过校验 (pip install onnx)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pose", "temporal"], required=True)
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--ckpt", default="checkpoints/stgcn_fall.pt")
    ap.add_argument("--out", default="checkpoints/stgcn_fall.onnx")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--allow-random", action="store_true",
                    help="无权重时仍导出随机初始化模型（仅供结构/尺寸检查，不可用于部署）")
    args = ap.parse_args()

    if args.stage == "temporal":
        export_temporal(args.ckpt, args.out, args.window, allow_random=args.allow_random)
    else:
        export_pose(args.weights, args.out)


if __name__ == "__main__":
    main()
