#!/usr/bin/env python3
"""Minimal calibration + fake quantization + error analysis for local YOLOv5.

This script is intentionally small and educational. It references the official
TPU-MLIR workflow conceptually:

1. calibration: collect activation ranges on a dataset
2. quantization: apply symmetric fake quantization to weights/activations
3. error analysis: compare FP32 and fake-quant outputs layer by layer

Unlike TPU-MLIR, this implementation runs on the local PyTorch YOLOv5 model and
focuses on a minimal, readable PTQ pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent
YOLOV5_ROOT = Path("/home/jay/projs/yolov5")
DEFAULT_WEIGHTS = YOLOV5_ROOT / "yolov5s.pt"
DEFAULT_DATASET = REPO_ROOT / "data" / "images"
DEFAULT_WORKDIR = REPO_ROOT / "experiments" / "02_quant"
DEFAULT_CALI_TABLE = "yolov5s_mini_cali_table.json"
DEFAULT_REPORT = "yolov5s_mini_ptq_report.json"
DEFAULT_SUMMARY = "yolov5s_mini_ptq_summary.md"


@dataclass
class LayerError:
    name: str
    cosine: float
    mse: float
    mae: float
    max_abs_diff: float
    num_elements: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local PTQ pipeline for YOLOv5.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--cali-table", default=DEFAULT_CALI_TABLE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calib-num", type=int, default=100)
    parser.add_argument("--eval-num", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mode",
        choices=["all", "calibrate", "eval"],
        default="all",
        help="all: calibrate then eval, calibrate: calibration only, eval: use an existing calibration table",
    )
    return parser.parse_args()


def set_runtime_env() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def add_yolov5_repo_to_path() -> None:
    if str(YOLOV5_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOV5_ROOT))


def load_yolov5_model(weights: Path, device: str) -> nn.Module:
    add_yolov5_repo_to_path()
    from models.experimental import attempt_download
    from models.yolo import Detect, Model

    ckpt = torch.load(attempt_download(str(weights)), map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()
    if hasattr(model, "fuse"):
        model = model.fuse()
    model = model.eval()
    for module in model.modules():
        t = type(module)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            module.inplace = True
            if t is Detect and not isinstance(module.anchor_grid, list):
                delattr(module, "anchor_grid")
                setattr(module, "anchor_grid", [torch.zeros(1, device=device)] * module.nl)
        elif t is nn.Upsample and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    return model


def list_images(dataset: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [path for path in sorted(dataset.iterdir()) if path.suffix.lower() in exts]
    if not images:
        raise SystemExit(f"No images found in dataset: {dataset}")
    return images


def letterbox(image: np.ndarray, new_shape: int = 640, color: tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def preprocess_image(path: Path, imgsz: int) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Failed to read image: {path}")
    image = letterbox(image, new_shape=imgsz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)


def iter_top_level_modules(model: nn.Module):
    for name, module in model.named_modules():
        if not name.startswith("model."):
            continue
        if "." in name[len("model."):]:
            continue
        yield name, module


def iter_tensors(obj: Any, prefix: str) -> list[tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        return [(prefix, obj)]
    if isinstance(obj, (list, tuple)):
        items: list[tuple[str, torch.Tensor]] = []
        for idx, value in enumerate(obj):
            items.extend(iter_tensors(value, f"{prefix}.{idx}"))
        return items
    if isinstance(obj, dict):
        items: list[tuple[str, torch.Tensor]] = []
        for key, value in obj.items():
            items.extend(iter_tensors(value, f"{prefix}.{key}"))
        return items
    return []


def replace_tensors(obj: Any, prefix: str, fn) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(prefix, obj)
    if isinstance(obj, list):
        return [replace_tensors(value, f"{prefix}.{idx}", fn) for idx, value in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(replace_tensors(value, f"{prefix}.{idx}", fn) for idx, value in enumerate(obj))
    if isinstance(obj, dict):
        return {key: replace_tensors(value, f"{prefix}.{key}", fn) for key, value in obj.items()}
    return obj


class ActivationCollector:
    def __init__(self, model: nn.Module):
        self.stats: dict[str, dict[str, Any]] = {}
        self.handles = []
        for name, module in iter_top_level_modules(model):
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, module_name: str):
        def hook(_module, _inputs, output):
            for tensor_name, tensor in iter_tensors(output, module_name):
                data = tensor.detach().float()
                absmax = float(data.abs().max().item())
                item = self.stats.setdefault(
                    tensor_name,
                    {
                        "absmax": 0.0,
                        "shape": list(data.shape),
                        "dtype": str(data.dtype).replace("torch.", ""),
                        "samples": 0,
                    },
                )
                item["absmax"] = max(float(item["absmax"]), absmax)
                item["samples"] = int(item["samples"]) + 1
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class FeatureCapture:
    def __init__(self, model: nn.Module):
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, module in iter_top_level_modules(model):
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, module_name: str):
        def hook(_module, _inputs, output):
            for tensor_name, tensor in iter_tensors(output, module_name):
                self.outputs[tensor_name] = tensor.detach().float().cpu()
        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def symmetric_fake_quant_tensor(tensor: torch.Tensor, absmax: float) -> torch.Tensor:
    if absmax <= 1e-12:
        return tensor
    scale = absmax / 127.0
    q = torch.clamp(torch.round(tensor / scale), -127, 127)
    return q * scale


def quantize_conv_weights_inplace(model: nn.Module) -> None:
    for module in model.modules():
        if not isinstance(module, nn.Conv2d):
            continue
        weight = module.weight.data.float()
        flat = weight.abs().amax(dim=(1, 2, 3), keepdim=True)
        scale = torch.where(flat > 1e-12, flat / 127.0, torch.ones_like(flat))
        q = torch.clamp(torch.round(weight / scale), -127, 127)
        module.weight.data.copy_(q * scale)


def register_activation_fake_quant(model: nn.Module, cali_table: dict[str, dict[str, Any]]):
    handles = []
    for name, module in iter_top_level_modules(model):
        def make_hook(module_name: str):
            def hook(_module, _inputs, output):
                def quantize_tensor(tensor_name: str, tensor: torch.Tensor) -> torch.Tensor:
                    info = cali_table.get(tensor_name)
                    if info is None:
                        return tensor
                    return symmetric_fake_quant_tensor(tensor, float(info["absmax"]))
                return replace_tensors(output, module_name, quantize_tensor)
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 1e-12:
        return 1.0
    return float(torch.dot(a, b).item() / denom.item())


def compute_layer_error(a: torch.Tensor, b: torch.Tensor) -> LayerError:
    diff = a - b
    mse = float(torch.mean(diff * diff).item())
    mae = float(torch.mean(diff.abs()).item())
    max_abs = float(diff.abs().max().item())
    cos = cosine_similarity(a, b)
    return LayerError(name="", cosine=cos, mse=mse, mae=mae, max_abs_diff=max_abs, num_elements=a.numel())


def calibrate(model: nn.Module, images: list[Path], imgsz: int, device: str) -> dict[str, dict[str, Any]]:
    collector = ActivationCollector(model)
    with torch.no_grad():
        for path in images:
            tensor = preprocess_image(path, imgsz).to(device)
            _ = model(tensor)
    collector.close()
    return dict(sorted(collector.stats.items()))


def evaluate_quantized_model(
    model: nn.Module,
    cali_table: dict[str, dict[str, Any]],
    images: list[Path],
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    fp32_model = model
    quant_model = copy.deepcopy(model)
    quantize_conv_weights_inplace(quant_model)
    q_handles = register_activation_fake_quant(quant_model, cali_table)

    fp32_capture = FeatureCapture(fp32_model)
    quant_capture = FeatureCapture(quant_model)

    layer_errors_acc: dict[str, dict[str, float]] = {}
    output_errors: list[dict[str, Any]] = []

    with torch.no_grad():
        for path in images:
            tensor = preprocess_image(path, imgsz).to(device)

            fp32_capture.clear()
            quant_capture.clear()

            fp32_out = fp32_model(tensor)
            quant_out = quant_model(tensor)

            # Top-level model output comparison
            fp32_main = fp32_out[0] if isinstance(fp32_out, (tuple, list)) else fp32_out
            quant_main = quant_out[0] if isinstance(quant_out, (tuple, list)) else quant_out
            image_err = compute_layer_error(fp32_main.detach().cpu(), quant_main.detach().cpu())
            output_errors.append(
                {
                    "image": path.name,
                    "cosine": image_err.cosine,
                    "mse": image_err.mse,
                    "mae": image_err.mae,
                    "max_abs_diff": image_err.max_abs_diff,
                }
            )

            common_keys = sorted(set(fp32_capture.outputs) & set(quant_capture.outputs))
            for name in common_keys:
                err = compute_layer_error(fp32_capture.outputs[name], quant_capture.outputs[name])
                item = layer_errors_acc.setdefault(
                    name,
                    {
                        "cosine_sum": 0.0,
                        "mse_sum": 0.0,
                        "mae_sum": 0.0,
                        "max_abs_diff": 0.0,
                        "num_elements": 0,
                        "count": 0,
                    },
                )
                item["cosine_sum"] += err.cosine
                item["mse_sum"] += err.mse
                item["mae_sum"] += err.mae
                item["max_abs_diff"] = max(float(item["max_abs_diff"]), err.max_abs_diff)
                item["num_elements"] = err.num_elements
                item["count"] += 1

    fp32_capture.close()
    quant_capture.close()
    for handle in q_handles:
        handle.remove()

    layer_errors: list[dict[str, Any]] = []
    for name, item in layer_errors_acc.items():
        count = max(int(item["count"]), 1)
        layer_errors.append(
            {
                "name": name,
                "cosine": float(item["cosine_sum"]) / count,
                "mse": float(item["mse_sum"]) / count,
                "mae": float(item["mae_sum"]) / count,
                "max_abs_diff": float(item["max_abs_diff"]),
                "num_elements": int(item["num_elements"]),
            }
        )

    layer_errors.sort(key=lambda item: (item["cosine"], -item["mse"]))
    output_errors.sort(key=lambda item: (item["cosine"], -item["mse"]))
    return {
        "output_errors": output_errors,
        "layer_errors": layer_errors,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(path: Path, cali_table: dict[str, dict[str, Any]], report: dict[str, Any]) -> None:
    top_layers = report["layer_errors"][:15]
    worst_images = report["output_errors"][:10]
    lines = [
        "# Mini PTQ Summary",
        "",
        "## Calibration",
        "",
        f"- 校准 tensor 数量: `{len(cali_table)}`",
        f"- 最大激活阈值 tensor: `{max(cali_table.items(), key=lambda kv: kv[1]['absmax'])[0]}`",
        "",
        "## Worst Output Images",
        "",
    ]
    for item in worst_images:
        lines.append(
            f"- `{item['image']}`: cosine={item['cosine']:.6f}, mse={item['mse']:.6e}, "
            f"mae={item['mae']:.6e}, max_abs_diff={item['max_abs_diff']:.6e}"
        )
    lines.extend(["", "## Worst Layers", ""])
    for item in top_layers:
        lines.append(
            f"- `{item['name']}`: cosine={item['cosine']:.6f}, mse={item['mse']:.6e}, "
            f"mae={item['mae']:.6e}, max_abs_diff={item['max_abs_diff']:.6e}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_runtime_env()
    args.workdir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.dataset)
    calib_images = images[: min(args.calib_num, len(images))]
    eval_images = images[: min(args.eval_num, len(images))]

    model = load_yolov5_model(args.weights, args.device)
    cali_path = args.workdir / args.cali_table
    report_path = args.workdir / args.report
    summary_path = args.workdir / args.summary

    cali_table: dict[str, dict[str, Any]]
    if args.mode in {"all", "calibrate"}:
        cali_table = calibrate(model, calib_images, args.imgsz, args.device)
        write_json(
            cali_path,
            {
                "meta": {
                    "weights": str(args.weights),
                    "dataset": str(args.dataset),
                    "imgsz": args.imgsz,
                    "calib_num": len(calib_images),
                    "device": args.device,
                },
                "stats": cali_table,
            },
        )
        print(f"Wrote calibration table to: {cali_path.resolve()}")
        print(f"Calibrated tensors: {len(cali_table)}")
    else:
        payload = json.loads(cali_path.read_text(encoding="utf-8"))
        cali_table = payload["stats"]

    if args.mode in {"all", "eval"}:
        report = evaluate_quantized_model(model, cali_table, eval_images, args.imgsz, args.device)
        payload = {
            "meta": {
                "weights": str(args.weights),
                "dataset": str(args.dataset),
                "imgsz": args.imgsz,
                "eval_num": len(eval_images),
                "device": args.device,
            },
            **report,
        }
        write_json(report_path, payload)
        write_summary(summary_path, cali_table, report)
        print(f"Wrote PTQ report to: {report_path.resolve()}")
        print(f"Wrote PTQ summary to: {summary_path.resolve()}")
        if report["output_errors"]:
            worst = report["output_errors"][0]
            print(
                "Worst image: "
                f"{worst['image']} cosine={worst['cosine']:.6f} mse={worst['mse']:.6e}"
            )
        if report["layer_errors"]:
            worst_layer = report["layer_errors"][0]
            print(
                "Worst layer: "
                f"{worst_layer['name']} cosine={worst_layer['cosine']:.6f} mse={worst_layer['mse']:.6e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
