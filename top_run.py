#!/usr/bin/env python3
"""Minimal Top MLIR runner for the local YOLOv5 learning workflow.

This runner intentionally targets the canonical YOLOv5 Top MLIR generated in
this repo. It is not a generic TPU-MLIR runtime replacement. The goal is to
execute a small set of `top.*` ops directly from the textual MLIR so we can:

1. run `yolov5s_canonical.mlir`
2. dump the three output blobs
3. compare them against reference blobs before moving to calibration/quant
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from top_canonicalize import RETURN_RE, attr_map, parse_bool, parse_float_attr, parse_int_array, parse_op_line, parse_tensor_type


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MLIR = REPO_ROOT / "experiments" / "01_onnx_to_mlir" / "yolov5s_canonical.mlir"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "03_top_run"
DEFAULT_OUTPUT_NPZ = "yolov5s_top_run_outputs.npz"
DEFAULT_REF_INPUT = Path("/home/jay/projs/tpu-mlir/regression/regression_out/yolov5s_bm1684x_num_core_1/yolov5s_in_f32.npz")
DEFAULT_REF_TOP = Path("/home/jay/projs/tpu-mlir/regression/regression_out/yolov5s_bm1684x_num_core_1/yolov5s_top_outputs.npz")
DEFAULT_OUTPUT_ALIASES = ["350", "498", "646"]

LOC_RE = re.compile(r"^(#loc[\w\d]+)\s*=\s*loc\(\"([^\"]*)\"\)\s*$")
WEIGHT_FILE_RE = re.compile(r'module\.weight_file\s*=\s*"([^"]+)"')


@dataclass
class MlirProgram:
    weight_file: Path
    ops: list
    return_values: list[str]
    loc_table: dict[str, str]
    op_by_result: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal subset of Top MLIR for YOLOv5.")
    parser.add_argument("--mlir", type=Path, default=DEFAULT_MLIR, help="Input canonical Top MLIR.")
    parser.add_argument("--weights", type=Path, help="Optional override for module.weight_file.")
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_REF_INPUT, help="Input NPZ that contains the model input tensor.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for runner outputs.")
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ, help="Output NPZ filename.")
    parser.add_argument("--compare-npz", type=Path, default=DEFAULT_REF_TOP, help="Optional reference NPZ for blob comparison.")
    parser.add_argument("--output-aliases", default="350,498,646", help="Comma-separated aliases for returned outputs.")
    parser.add_argument("--dump-all", action="store_true", help="Dump all intermediate tensors in addition to returned outputs.")
    return parser.parse_args()


def load_program(mlir_path: Path, weight_override: Path | None = None) -> MlirProgram:
    text = mlir_path.read_text(encoding="utf-8")
    weight_match = WEIGHT_FILE_RE.search(text)
    if not weight_match and weight_override is None:
        raise ValueError(f"Failed to find module.weight_file in {mlir_path}")
    weight_file = weight_override or (mlir_path.parent / weight_match.group(1)).resolve()
    ops = []
    return_values: list[str] = []
    loc_table: dict[str, str] = {}
    op_by_result: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        op = parse_op_line(line)
        if op is not None:
            ops.append(op)
            op_by_result[op.result] = op
            continue
        loc_match = LOC_RE.match(line)
        if loc_match:
            loc_table[loc_match.group(1)] = loc_match.group(2)
            continue
        ret_match = RETURN_RE.match(line)
        if ret_match:
            return_values = [item.strip() for item in ret_match.group("values").split(",")]
    if not return_values:
        raise ValueError(f"Failed to parse return line in {mlir_path}")
    return MlirProgram(weight_file=weight_file, ops=ops, return_values=return_values, loc_table=loc_table, op_by_result=op_by_result)


def resolve_loc(loc: str, loc_table: dict[str, str]) -> str:
    return loc_table.get(loc, loc)


def sanitize_key(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", name.strip())
    return safe or "tensor"


def parse_input_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name] for name in data.files}


def to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(array))


def as_numpy(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def conv2d_run(inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, attrs: list[tuple[str, str]]) -> torch.Tensor:
    amap = attr_map(attrs)
    strides = tuple(parse_int_array(amap.get("strides", "[1, 1]")))
    dilations = tuple(parse_int_array(amap.get("dilations", "[1, 1]")))
    pads = parse_int_array(amap.get("pads", "[0, 0, 0, 0]"))
    groups = int(amap.get("group", "1 : i64").split(":", 1)[0].strip())
    if pads[0] == pads[2] and pads[1] == pads[3]:
        padding = (pads[0], pads[1])
        padded = inp
    else:
        padded = F.pad(inp, (pads[1], pads[3], pads[0], pads[2]))
        padding = (0, 0)
    return F.conv2d(padded, weight, bias, stride=strides, padding=padding, dilation=dilations, groups=groups)


def maxpool_run(inp: torch.Tensor, attrs: list[tuple[str, str]]) -> torch.Tensor:
    amap = attr_map(attrs)
    kernel = tuple(parse_int_array(amap.get("kernel_shape", "[1, 1]")))
    strides = tuple(parse_int_array(amap.get("strides", "[1, 1]")))
    pads = parse_int_array(amap.get("pads", "[0, 0, 0, 0]"))
    ceil_mode = parse_bool(amap.get("ceil_mode", "false"))
    if pads[0] == pads[2] and pads[1] == pads[3]:
        padding = (pads[0], pads[1])
        padded = inp
    else:
        padded = F.pad(inp, (pads[1], pads[3], pads[0], pads[2]), mode="constant", value=0.0)
        padding = (0, 0)
    return F.max_pool2d(padded, kernel_size=kernel, stride=strides, padding=padding, ceil_mode=ceil_mode)


def interp_run(inp: torch.Tensor, output_type: str, attrs: list[tuple[str, str]]) -> torch.Tensor:
    dims, _ = parse_tensor_type(output_type)
    if dims is None or len(dims) != 4 or dims[2] is None or dims[3] is None:
        raise ValueError(f"Interp expects static 4D output type, got: {output_type}")
    amap = attr_map(attrs)
    mode = amap.get("mode", '"nearest"').strip('"')
    coord_mode = amap.get("coord_mode", '"asymmetric"').strip('"')
    if mode == "nearest":
        return F.interpolate(inp, size=(dims[2], dims[3]), mode="nearest")
    align_corners = coord_mode != "asymmetric"
    return F.interpolate(inp, size=(dims[2], dims[3]), mode="bilinear", align_corners=align_corners)


def reshape_run(inp: torch.Tensor, attrs: list[tuple[str, str]]) -> torch.Tensor:
    amap = attr_map(attrs)
    shape = parse_int_array(amap["shape"])
    return inp.reshape(shape)


def permute_run(inp: torch.Tensor, attrs: list[tuple[str, str]]) -> torch.Tensor:
    amap = attr_map(attrs)
    order = parse_int_array(amap["order"])
    return inp.permute(*order).contiguous()


def execute_program(program: MlirProgram, inputs: dict[str, np.ndarray]) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
    weights = np.load(program.weight_file)
    env: dict[str, torch.Tensor | None] = {}
    outputs: dict[str, torch.Tensor] = {}
    ordered_input_values = list(inputs.values())
    input_index = 0
    for op in program.ops:
        if op.op == "top.None":
            env[op.result] = None
            continue
        if op.op == "top.Input":
            loc_name = resolve_loc(op.loc, program.loc_table)
            if loc_name in inputs:
                tensor = to_tensor(inputs[loc_name])
            elif len(inputs) == 1:
                tensor = to_tensor(ordered_input_values[0])
            else:
                tensor = to_tensor(ordered_input_values[input_index])
                input_index += 1
            env[op.result] = tensor
            outputs[loc_name] = tensor
            continue
        if op.op == "top.Weight":
            path = attr_map(op.attrs)["path"].strip('"')
            if path not in weights:
                raise KeyError(f"Weight `{path}` not found in {program.weight_file}")
            env[op.result] = to_tensor(weights[path])
            outputs[path] = env[op.result]
            continue

        args = [env[name] for name in op.operands]
        if op.op == "top.Conv":
            result = conv2d_run(args[0], args[1], args[2], op.attrs)
        elif op.op == "top.Sigmoid":
            result = torch.sigmoid(args[0])
        elif op.op == "top.Mul":
            result = args[0] * args[1]
        elif op.op == "top.Add":
            result = args[0] + args[1]
        elif op.op == "top.Concat":
            axis = int(attr_map(op.attrs)["axis"].split(":", 1)[0].strip())
            result = torch.cat(args, dim=axis)
        elif op.op == "top.MaxPool":
            result = maxpool_run(args[0], op.attrs)
        elif op.op == "top.Interp":
            result = interp_run(args[0], op.output_type, op.attrs)
        elif op.op == "top.Reshape":
            result = reshape_run(args[0], op.attrs)
        elif op.op == "top.Permute":
            result = permute_run(args[0], op.attrs)
        elif op.op == "top.SiLU":
            result = F.silu(args[0])
        else:
            raise NotImplementedError(f"Unsupported op for top_run.py: {op.op}")
        env[op.result] = result
        outputs[resolve_loc(op.loc, program.loc_table)] = result
    return env, outputs


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_flat = lhs.reshape(-1).astype(np.float64)
    rhs_flat = rhs.reshape(-1).astype(np.float64)
    lhs_norm = np.linalg.norm(lhs_flat)
    rhs_norm = np.linalg.norm(rhs_flat)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 1.0 if lhs_norm == rhs_norm else 0.0
    return float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))


def compare_arrays(actual: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = actual.astype(np.float64) - ref.astype(np.float64)
    return {
        "cosine": cosine_similarity(actual, ref),
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def official_yolov5_head_refs(ref_npz: Path) -> list[np.ndarray]:
    data = np.load(ref_npz)
    conv_keys = ["326_Conv", "474_Conv", "622_Conv"]
    hw = [80, 40, 20]
    refs: list[np.ndarray] = []
    for key, size in zip(conv_keys, hw):
        if key not in data:
            raise KeyError(f"Reference NPZ is missing `{key}`")
        conv = data[key]
        reshaped = conv.reshape(1, 3, 85, size, size).transpose(0, 1, 3, 4, 2)
        refs.append(reshaped)
    return refs


def save_outputs(
    path: Path,
    program: MlirProgram,
    env: dict[str, torch.Tensor | None],
    named_outputs: dict[str, torch.Tensor],
    aliases: list[str],
    dump_all: bool,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for index, result_name in enumerate(program.return_values):
        array = as_numpy(env[result_name])
        if array is None:
            continue
        payload[f"out{index}"] = array
        op = program.op_by_result[result_name]
        loc_name = resolve_loc(op.loc, program.loc_table)
        payload[sanitize_key(loc_name)] = array
        if index < len(aliases):
            payload[sanitize_key(aliases[index])] = array
    if dump_all:
        for key, value in named_outputs.items():
            array = as_numpy(value)
            if array is None:
                continue
            safe = sanitize_key(key)
            if safe not in payload:
                payload[safe] = array
    np.savez(path, **payload)
    return payload


def main() -> None:
    args = parse_args()
    aliases = [item.strip() for item in args.output_aliases.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_dir / args.output_npz

    program = load_program(args.mlir, args.weights.resolve() if args.weights else None)
    inputs = parse_input_npz(args.input_npz)
    env, named_outputs = execute_program(program, inputs)
    saved = save_outputs(output_npz, program, env, named_outputs, aliases, args.dump_all)

    print(f"Loaded MLIR: {args.mlir.resolve()}")
    print(f"Loaded weights: {program.weight_file}")
    print(f"Loaded input NPZ: {args.input_npz.resolve()}")
    print(f"Wrote output NPZ: {output_npz.resolve()}")

    for index, result_name in enumerate(program.return_values):
        op = program.op_by_result[result_name]
        loc_name = resolve_loc(op.loc, program.loc_table)
        array = as_numpy(env[result_name])
        print(f"output[{index}] {aliases[index] if index < len(aliases) else loc_name}: {loc_name} {tuple(array.shape)}")

    if args.compare_npz and args.compare_npz.exists():
        refs = official_yolov5_head_refs(args.compare_npz)
        print("Reference comparison:")
        for index, (result_name, ref) in enumerate(zip(program.return_values, refs)):
            actual = as_numpy(env[result_name])
            metrics = compare_arrays(actual, ref)
            alias = aliases[index] if index < len(aliases) else f"out{index}"
            print(
                f"  {alias}: cosine={metrics['cosine']:.6f}, "
                f"mse={metrics['mse']:.6f}, mae={metrics['mae']:.6f}, "
                f"max_abs_diff={metrics['max_abs_diff']:.6f}"
            )
    elif args.compare_npz:
        print(f"Reference NPZ not found, skipped compare: {args.compare_npz}")


if __name__ == "__main__":
    main()
