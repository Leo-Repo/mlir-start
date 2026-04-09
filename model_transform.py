#!/usr/bin/env python3
"""Standalone ONNX -> Top MLIR importer for the local YOLOv5 workflow.

This script intentionally does not call TPU-MLIR's Python package. It parses
an ONNX graph, materializes weights into an `.npz`, and emits a readable Top
MLIR text file that mirrors the structure used by TPU-MLIR's ONNX frontend.

The current implementation focuses on the operator subset commonly seen in a
YOLOv5 export:
  - Conv
  - Sigmoid
  - Mul / Add
  - Concat
  - MaxPool
  - Resize -> top.Interp
  - Reshape
  - Transpose -> top.Permute
  - Slice
  - Shape / Gather / Unsqueeze / Cast for shape subgraphs
  - Range / Expand for grid-generation subgraphs
  - ConstantOfShape for zero/one-filled shape tensors
  - Identity / Constant
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ONNX = REPO_ROOT / "models" / "yolov5s.onnx"
DEFAULT_WORKDIR = REPO_ROOT / "experiments" / "01_onnx_to_mlir"
DEFAULT_MLIR = "yolov5s.mlir"
DEFAULT_WEIGHT = "yolov5s_top_f32_all_origin_weight.npz"
DEFAULT_MODEL_NAME = "yolov5s"
DEFAULT_INPUT_SHAPES = "[[1,3,640,640]]"
DEFAULT_MEAN = "0.0,0.0,0.0"
DEFAULT_SCALE = "0.0039216,0.0039216,0.0039216"
DEFAULT_OUTPUT_NAMES = "350,498,646"
ROUND_MODE = "HalfAwayFromZero"


def import_onnx():
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper, shape_inference
    except ImportError as exc:
        raise SystemExit(
            "This script requires the `onnx` Python package in the current "
            "environment. Install or activate an environment that provides it."
        ) from exc
    return onnx, TensorProto, helper, numpy_helper, shape_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone ONNX -> Top MLIR converter for yolov5-style graphs."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-def", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--mlir", default=DEFAULT_MLIR)
    parser.add_argument("--weight-file", default=DEFAULT_WEIGHT)
    parser.add_argument(
        "--input-shapes",
        default=DEFAULT_INPUT_SHAPES,
        help="Python-literal list, for example [[1,3,640,640]].",
    )
    parser.add_argument("--mean", default=DEFAULT_MEAN)
    parser.add_argument("--scale", default=DEFAULT_SCALE)
    parser.add_argument(
        "--pixel-format",
        default="rgb",
        choices=["rgb", "bgr", "gray", "rgbd"],
    )
    parser.add_argument(
        "--output-names",
        default=DEFAULT_OUTPUT_NAMES,
        help="Comma-separated output names. Empty means use graph outputs.",
    )
    parser.add_argument("--resize-dims", default="")
    parser.add_argument("--keep-aspect-ratio", action="store_true", default=True)
    parser.add_argument(
        "--no-keep-aspect-ratio",
        action="store_false",
        dest="keep_aspect_ratio",
    )
    parser.add_argument(
        "--channel-format",
        default="nchw",
        choices=["nchw", "nhwc"],
    )
    parser.add_argument(
        "--pad-type",
        default="normal",
        choices=["normal", "center"],
    )
    parser.add_argument("--dump-summary", action="store_true")
    return parser.parse_args()


def sanitize_symbol(name: str, fallback: str) -> str:
    text = (name or fallback).replace(":", "_").replace("/", "_").replace(".", "_")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"v_{text}"
    return text


def to_bool_text(value: bool) -> str:
    return "true" if value else "false"


def to_int_array(values: list[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def to_float_array(values: list[float]) -> str:
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def parse_csv_numbers(text: str, caster) -> list[Any]:
    return [caster(part.strip()) for part in text.split(",") if part.strip()]


def parse_input_shapes(text: str) -> list[list[int]]:
    try:
        value = ast.literal_eval(text)
    except Exception as exc:
        raise SystemExit(f"Invalid --input-shapes: {text}") from exc
    if not isinstance(value, list) or not all(isinstance(v, list) for v in value):
        raise SystemExit("--input-shapes must look like [[1,3,640,640]]")
    return [[int(dim) for dim in shape] for shape in value]


def mlir_element_type(np_dtype: np.dtype) -> str:
    np_dtype = np.dtype(np_dtype)
    if np_dtype == np.float32:
        return "f32"
    if np_dtype == np.float16:
        return "f16"
    if np_dtype == np.float64:
        return "f64"
    if np_dtype == np.int8:
        return "i8"
    if np_dtype == np.uint8:
        return "ui8"
    if np_dtype == np.int16:
        return "i16"
    if np_dtype == np.uint16:
        return "ui16"
    if np_dtype == np.int32:
        return "i32"
    if np_dtype == np.uint32:
        return "ui32"
    if np_dtype == np.int64:
        return "i64"
    if np_dtype == np.uint64:
        return "ui64"
    if np_dtype == np.bool_:
        return "i1"
    raise SystemExit(f"Unsupported dtype for MLIR emission: {np_dtype}")


def tensor_type(shape: list[int] | None, dtype: np.dtype | str = np.float32) -> str:
    elem = dtype if isinstance(dtype, str) else mlir_element_type(np.dtype(dtype))
    if shape is None:
        return f"tensor<*x{elem}>"
    if len(shape) == 0:
        return f"tensor<{elem}>"
    dims = "x".join("?" if dim is None or dim < 0 else str(int(dim)) for dim in shape)
    return f"tensor<{dims}x{elem}>"


def parse_resize_dims(text: str, fallback_shape: list[int] | None) -> list[int]:
    if text:
        dims = parse_csv_numbers(text, int)
        if len(dims) != 2:
            raise SystemExit("--resize-dims must look like 640,640")
        return dims
    if fallback_shape and len(fallback_shape) >= 4:
        return [int(fallback_shape[-2]), int(fallback_shape[-1])]
    return []


@dataclass
class ValueRef:
    name: str
    type_str: str
    shape: list[int] | None
    dtype: np.dtype
    onnx_name: str


class MlirBuilder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lines: list[str] = []
        self.weight_map: dict[str, np.ndarray] = {}
        self.weight_values: dict[str, ValueRef] = {}
        self.value_map: dict[str, ValueRef] = {}
        self.none_value: ValueRef | None = None
        self.counter = 0
        self.loc_aliases: dict[str, str] = {}
        self.loc_order: list[str] = []

    def new_value(
        self,
        onnx_name: str,
        shape: list[int] | None,
        dtype: np.dtype | str = np.float32,
    ) -> ValueRef:
        self.counter += 1
        np_dtype = np.dtype(dtype) if not isinstance(dtype, str) else np.float32
        type_str = tensor_type(shape, dtype)
        return ValueRef(
            name=f"%{self.counter}",
            type_str=type_str,
            shape=shape,
            dtype=np_dtype,
            onnx_name=onnx_name,
        )

    def emit(self, text: str) -> None:
        self.lines.append(f"    {text}")

    def loc_ref(self, raw_name: str) -> str:
        name = raw_name or "unknown"
        if name not in self.loc_aliases:
            alias = f"#loc{len(self.loc_order) + 1}"
            self.loc_aliases[name] = alias
            self.loc_order.append(name)
        return self.loc_aliases[name]

    def loc_definitions(self) -> list[str]:
        defs = []
        for name in self.loc_order:
            alias = self.loc_aliases[name]
            escaped = name.replace("\\", "\\\\").replace('"', '\\"')
            defs.append(f'{alias} = loc("{escaped}")')
        return defs

    def ensure_none(self) -> ValueRef:
        if self.none_value is None:
            self.none_value = ValueRef("%0", "none", None, np.float32, "__none__")
            self.emit(f'%0 = "top.None"() : () -> none loc({self.loc_ref("none")})')
        return self.none_value

    def create_input(self, arg_name: str, arg_type: str, shape: list[int], loc_name: str) -> ValueRef:
        value = self.new_value(arg_name, shape, np.float32)
        attrs = {
            "channel_format": f'"{self.args.channel_format}"',
            "do_preprocess": "true",
            "keep_aspect_ratio": to_bool_text(self.args.keep_aspect_ratio),
            'keep_ratio_mode': '"letterbox"',
            "mean": to_float_array(parse_csv_numbers(self.args.mean, float)),
            "pad_type": f'"{self.args.pad_type}"',
            "pad_value": "0 : i64",
            "pixel_format": f'"{self.args.pixel_format}"',
            "scale": to_float_array(parse_csv_numbers(self.args.scale, float)),
            'yuv_type': '""',
        }
        resize_dims = parse_resize_dims(self.args.resize_dims, shape)
        if resize_dims:
            attrs["resize_dims"] = to_int_array(resize_dims)
        attr_text = ", ".join(f"{key} = {value}" for key, value in attrs.items())
        self.emit(
            f'{value.name} = "top.Input"({arg_name}) '
            f'{{{attr_text}}} : ({arg_type}) -> {value.type_str} loc({self.loc_ref(loc_name)})'
        )
        return value

    def create_weight(self, onnx_name: str, array: np.ndarray, loc_name: str | None = None) -> ValueRef:
        if onnx_name in self.weight_values:
            return self.weight_values[onnx_name]
        key = sanitize_symbol(onnx_name, f"weight_{len(self.weight_values)}")
        if key in self.weight_map:
            suffix = 1
            base = key
            while f"{base}_{suffix}" in self.weight_map:
                suffix += 1
            key = f"{base}_{suffix}"
        arr = np.asarray(array)
        shape = list(arr.shape)
        value = self.new_value(onnx_name, shape, arr.dtype)
        self.weight_map[key] = arr
        self.emit(
            f'{value.name} = "top.Weight"() {{path = "{key}"}} '
            f': () -> {value.type_str} loc({self.loc_ref(loc_name or onnx_name)})'
        )
        self.weight_values[onnx_name] = value
        return value

    def create_op(
        self,
        op_name: str,
        operands: list[ValueRef],
        attrs: dict[str, str],
        result_onnx_name: str,
        result_shape: list[int] | None,
        result_dtype: np.dtype | str = np.float32,
        loc_name: str | None = None,
    ) -> ValueRef:
        value = self.new_value(result_onnx_name, result_shape, result_dtype)
        operand_text = ", ".join(operand.name for operand in operands)
        input_types = ", ".join(operand.type_str for operand in operands)
        attr_text = ", ".join(f"{key} = {val}" for key, val in attrs.items())
        if attr_text:
            attr_text = f" {{{attr_text}}}"
        self.emit(
            f'{value.name} = "{op_name}"({operand_text}){attr_text} '
            f': ({input_types}) -> {value.type_str} loc({self.loc_ref(loc_name or result_onnx_name)})'
        )
        return value


class OnnxToTopImporter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.onnx, self.TensorProto, self.helper, self.numpy_helper, self.shape_inference = import_onnx()
        self.model = self.load_model()
        self.graph = self.model.graph
        self.initializers = {init.name: init for init in self.graph.initializer}
        self.initializer_names = set(self.initializers)
        self.const_tensors: dict[str, np.ndarray] = {
            init.name: np.asarray(self.numpy_helper.to_array(init))
            for init in self.graph.initializer
        }
        self.value_info = self.collect_value_info()
        self.input_overrides = parse_input_shapes(args.input_shapes)
        self.builder = MlirBuilder(args)

    def node_loc_name(self, node, *, default_output: str | None = None) -> str:
        if getattr(node, "name", ""):
            return str(node.name)
        if default_output:
            return str(default_output)
        if getattr(node, "output", None):
            for output_name in node.output:
                if output_name:
                    return str(output_name)
        return "unknown"

    def load_model(self):
        model_path = self.args.model_def
        if not model_path.exists():
            raise SystemExit(f"ONNX model not found: {model_path}")
        model = self.onnx.load(str(model_path))
        try:
            model = self.shape_inference.infer_shapes(model)
        except Exception:
            pass
        return model

    def collect_value_info(self) -> dict[str, tuple[list[int] | None, np.dtype]]:
        info: dict[str, tuple[list[int] | None, np.dtype]] = {}
        for value in list(self.graph.input) + list(self.graph.value_info) + list(self.graph.output):
            info[value.name] = self.tensor_info_from_value(value)
        for init in self.graph.initializer:
            array = self.numpy_helper.to_array(init)
            info[init.name] = (list(array.shape), array.dtype)
        return info

    def tensor_info_from_value(self, value) -> tuple[list[int] | None, np.dtype]:
        tensor_type_proto = value.type.tensor_type
        shape = []
        for dim in tensor_type_proto.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            else:
                shape.append(None)
        elem_type = tensor_type_proto.elem_type
        dtype = self.helper.tensor_dtype_to_np_dtype(elem_type)
        return (shape, np.dtype(dtype))

    def output_names(self) -> list[str]:
        if self.args.output_names.strip():
            return [name.strip() for name in self.args.output_names.split(",") if name.strip()]
        return [out.name for out in self.graph.output]

    def graph_inputs(self):
        raw_inputs = [value for value in self.graph.input if value.name not in self.initializer_names]
        for idx, value in enumerate(raw_inputs):
            shape, dtype = self.value_info.get(value.name, (None, np.float32))
            if idx < len(self.input_overrides):
                self.value_info[value.name] = (self.input_overrides[idx], dtype)
        return raw_inputs

    def selected_nodes(self, output_names: list[str]):
        producer = {}
        for node in self.graph.node:
            for out_name in node.output:
                producer[out_name] = node

        selected = []
        visited_nodes = set()
        visiting_values = set()

        def visit_value(value_name: str):
            if not value_name or value_name in visiting_values:
                return
            visiting_values.add(value_name)
            node = producer.get(value_name)
            if node is None:
                return
            node_id = id(node)
            if node_id in visited_nodes:
                return
            for input_name in node.input:
                visit_value(input_name)
            visited_nodes.add(node_id)
            selected.append(node)

        for output_name in output_names:
            visit_value(output_name)
        return selected

    def ensure_operand(self, name: str) -> ValueRef:
        if not name:
            return self.builder.ensure_none()
        if name in self.builder.value_map:
            return self.builder.value_map[name]
        if name in self.initializers:
            array = self.numpy_helper.to_array(self.initializers[name])
            value = self.builder.create_weight(name, array, loc_name=name)
            self.builder.value_map[name] = value
            return value
        raise SystemExit(f"Unknown ONNX value referenced before definition: {name}")

    def shape_of(self, name: str) -> list[int] | None:
        return self.value_info.get(name, (None, np.float32))[0]

    def dtype_of(self, name: str) -> np.dtype:
        return self.value_info.get(name, (None, np.float32))[1]

    def attr_dict(self, node) -> dict[str, Any]:
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = self.helper.get_attribute_value(attr)
        return attrs

    def constant_array(self, name: str) -> np.ndarray:
        if name in self.const_tensors:
            return self.const_tensors[name]
        raise SystemExit(f"Expected constant tensor, got dynamic value: {name}")

    def is_constant_value(self, name: str) -> bool:
        return bool(name) and name in self.const_tensors

    def materialize_constant_node(self, node) -> None:
        attrs = self.attr_dict(node)
        if "value" not in attrs:
            raise SystemExit(f"Constant node without tensor payload: {node.name or node.output[0]}")
        array = np.asarray(self.numpy_helper.to_array(attrs["value"]))
        value = self.builder.create_weight(
            node.output[0], array, loc_name=self.node_loc_name(node, default_output=node.output[0])
        )
        self.builder.value_map[node.output[0]] = value
        self.const_tensors[node.output[0]] = array
        self.value_info[node.output[0]] = (list(array.shape), array.dtype)

    def zero_bias(self, output_channels: int, base_name: str) -> ValueRef:
        array = np.zeros((output_channels,), dtype=np.float32)
        name = f"{base_name}_bias_auto"
        value = self.builder.create_weight(name, array, loc_name=name)
        self.builder.value_map[name] = value
        return value

    def materialize_const_output(self, output_name: str, array: np.ndarray, loc_name: str | None = None) -> None:
        arr = np.asarray(array)
        value = self.builder.create_weight(output_name, arr, loc_name=loc_name or output_name)
        self.builder.value_map[output_name] = value
        self.const_tensors[output_name] = arr
        self.value_info[output_name] = (list(arr.shape), arr.dtype)

    def emit_identity(self, node) -> None:
        src = self.ensure_operand(node.input[0])
        self.builder.value_map[node.output[0]] = src

    def emit_shape(self, node) -> None:
        input_shape = self.shape_of(node.input[0])
        if input_shape is None:
            raise SystemExit(f"Shape node needs known input shape: {node.name or node.output[0]}")
        attrs = self.attr_dict(node)
        start = int(attrs.get("start", 0))
        end = attrs.get("end")
        if end is None:
            sliced = input_shape[start:]
        else:
            sliced = input_shape[start:int(end)]
        values = [int(-1 if dim is None else dim) for dim in sliced]
        self.materialize_const_output(
            node.output[0], np.asarray(values, dtype=np.int64), loc_name=self.node_loc_name(node, default_output=node.output[0])
        )

    def emit_gather(self, node) -> None:
        data = self.constant_array(node.input[0])
        indices = self.constant_array(node.input[1]).astype(np.int64)
        attrs = self.attr_dict(node)
        axis = int(attrs.get("axis", 0))
        result = np.take(data, indices, axis=axis)
        self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))

    def emit_unsqueeze(self, node) -> None:
        data = self.constant_array(node.input[0])
        attrs = self.attr_dict(node)
        if len(node.input) > 1 and node.input[1]:
            axes = self.constant_array(node.input[1]).astype(np.int64).tolist()
        else:
            axes = [int(v) for v in attrs.get("axes", [])]
        result = data
        ndim = result.ndim
        normalized_axes = []
        for axis in axes:
            axis = int(axis)
            if axis < 0:
                axis += ndim + 1
            normalized_axes.append(axis)
            ndim += 1
        for axis in sorted(normalized_axes):
            result = np.expand_dims(result, axis=axis)
        self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))

    def emit_cast(self, node) -> None:
        data = self.constant_array(node.input[0])
        attrs = self.attr_dict(node)
        target_dtype = np.dtype(self.helper.tensor_dtype_to_np_dtype(int(attrs["to"])))
        self.materialize_const_output(
            node.output[0], data.astype(target_dtype, copy=False), loc_name=self.node_loc_name(node, default_output=node.output[0])
        )

    def emit_range(self, node) -> None:
        start = np.asarray(self.constant_array(node.input[0])).item()
        limit = np.asarray(self.constant_array(node.input[1])).item()
        delta = np.asarray(self.constant_array(node.input[2])).item()
        sample = np.asarray(start + limit + delta)
        result = np.arange(start, limit, delta, dtype=sample.dtype)
        self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))

    def emit_expand(self, node) -> None:
        data = self.constant_array(node.input[0])
        target_shape = [int(v) for v in self.constant_array(node.input[1]).tolist()]
        normalized = []
        for dim in target_shape:
            if dim == -1:
                normalized.append(1)
            else:
                normalized.append(dim)
        result = np.broadcast_to(data, normalized)
        self.materialize_const_output(node.output[0], np.asarray(result), loc_name=self.node_loc_name(node, default_output=node.output[0]))

    def emit_constant_of_shape(self, node) -> None:
        shape = [int(v) for v in self.constant_array(node.input[0]).astype(np.int64).tolist()]
        attrs = self.attr_dict(node)
        if "value" in attrs:
            fill_tensor = np.asarray(self.numpy_helper.to_array(attrs["value"]))
            fill_value = fill_tensor.reshape(-1)[0]
            dtype = fill_tensor.dtype
        else:
            fill_value = 0.0
            dtype = np.float32
        result = np.full(shape, fill_value, dtype=dtype)
        self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))

    def emit_sigmoid(self, node) -> None:
        if self.is_constant_value(node.input[0]):
            data = self.constant_array(node.input[0]).astype(np.float64)
            result = 1.0 / (1.0 + np.exp(-data))
            self.materialize_const_output(
                node.output[0], result.astype(data.dtype, copy=False), loc_name=self.node_loc_name(node, default_output=node.output[0])
            )
            return
        inp = self.ensure_operand(node.input[0])
        out = self.builder.create_op(
            "top.Sigmoid",
            [inp],
            {
                "bias": "0.0 : f64",
                "log": "false",
                "round_mode": f'"{ROUND_MODE}"',
                "scale": "1.0 : f64",
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_binary(self, node, op_name: str) -> None:
        if self.is_constant_value(node.input[0]) and self.is_constant_value(node.input[1]):
            lhs = self.constant_array(node.input[0])
            rhs = self.constant_array(node.input[1])
            if op_name == "top.Mul":
                result = np.multiply(lhs, rhs)
            elif op_name == "top.Add":
                result = np.add(lhs, rhs)
            else:
                raise SystemExit(f"Unsupported constant-fold binary op: {op_name}")
            self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))
            return
        lhs = self.ensure_operand(node.input[0])
        rhs = self.ensure_operand(node.input[1])
        is_scalar = "true" if (rhs.shape == [] or rhs.shape == [1] or lhs.shape == [] or lhs.shape == [1]) else "false"
        out = self.builder.create_op(
            op_name,
            [lhs, rhs],
            {
                "do_relu": "false",
                "is_scalar": is_scalar,
                "relu_limit": "-1.0 : f64",
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_concat(self, node) -> None:
        attrs = self.attr_dict(node)
        if all(self.is_constant_value(name) for name in node.input):
            arrays = [self.constant_array(name) for name in node.input]
            result = np.concatenate(arrays, axis=int(attrs.get("axis", 0)))
            self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))
            return
        ops = [self.ensure_operand(name) for name in node.input]
        out = self.builder.create_op(
            "top.Concat",
            ops,
            {
                "axis": f'{int(attrs.get("axis", 1))} : si32',
                "do_relu": "false",
                "only_merge": "false",
                "relu_limit": "-1.0 : f64",
                "round_mode": f'"{ROUND_MODE}"',
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_conv(self, node) -> None:
        attrs = self.attr_dict(node)
        data = self.ensure_operand(node.input[0])
        weight = self.ensure_operand(node.input[1])
        weight_shape = weight.shape or []
        if len(node.input) >= 3 and node.input[2]:
            bias = self.ensure_operand(node.input[2])
        else:
            if not weight_shape:
                raise SystemExit(f"Cannot infer bias size for Conv node: {node.name or node.output[0]}")
            bias = self.zero_bias(weight_shape[0], node.output[0])
        pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
        strides = [int(v) for v in attrs.get("strides", [1, 1])]
        dilations = [int(v) for v in attrs.get("dilations", [1, 1])]
        kernel_shape = [int(v) for v in attrs.get("kernel_shape", weight_shape[-2:])]
        out = self.builder.create_op(
            "top.Conv",
            [data, weight, bias],
            {
                "auto_pad": f'"{attrs.get("auto_pad", "NOTSET")}"',
                "dilations": to_int_array(dilations),
                "do_relu": "false",
                "dynweight_reorderd": "false",
                "group": f'{int(attrs.get("group", 1))} : i64',
                "kernel_shape": to_int_array(kernel_shape),
                "pads": to_int_array(pads),
                "relu_limit": "-1.0 : f64",
                "strides": to_int_array(strides),
                "weight_is_coeff": "1 : i64",
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_maxpool(self, node) -> None:
        attrs = self.attr_dict(node)
        inp = self.ensure_operand(node.input[0])
        out = self.builder.create_op(
            "top.MaxPool",
            [inp],
            {
                "auto_pad": f'"{attrs.get("auto_pad", "NOTSET")}"',
                "ceil_mode": to_bool_text(bool(attrs.get("ceil_mode", 0))),
                "count_include_pad": "false",
                "do_relu": "false",
                "first_round_mode": f'"{ROUND_MODE}"',
                "is_adaptive": "false",
                "keepdims": "true",
                "kernel_shape": to_int_array([int(v) for v in attrs.get("kernel_shape", [])]),
                "pad_value": "0 : i64",
                "pads": to_int_array([int(v) for v in attrs.get("pads", [0, 0, 0, 0])]),
                "relu_limit": "-1.0 : f64",
                "round_mode": f'"{ROUND_MODE}"',
                "strides": to_int_array([int(v) for v in attrs.get("strides", [1, 1])]),
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_resize(self, node) -> None:
        attrs = self.attr_dict(node)
        inp = self.ensure_operand(node.input[0])
        none_value = self.builder.ensure_none()
        scales = None
        if len(node.input) >= 3 and node.input[2]:
            scales = self.constant_array(node.input[2]).astype(np.float64)
        if scales is None and len(node.input) >= 4 and node.input[3]:
            sizes = self.constant_array(node.input[3]).astype(np.float64)
            input_shape = self.shape_of(node.input[0])
            if input_shape and len(input_shape) == len(sizes):
                scales = sizes / np.asarray(input_shape, dtype=np.float64)
        if scales is None or len(scales) < 2:
            raise SystemExit(f"Resize node requires constant scales or sizes: {node.name or node.output[0]}")
        out = self.builder.create_op(
            "top.Interp",
            [inp, none_value],
            {
                "coord_mode": f'"{attrs.get("coordinate_transformation_mode", "asymmetric")}"',
                "mode": f'"{attrs.get("mode", "nearest")}"',
                "scale_h": f"{float(scales[-2])} : f64",
                "scale_w": f"{float(scales[-1])} : f64",
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_reshape(self, node) -> None:
        if self.is_constant_value(node.input[0]) and self.is_constant_value(node.input[1]):
            data = self.constant_array(node.input[0])
            target_shape = [int(v) for v in self.constant_array(node.input[1]).tolist()]
            result = np.reshape(data, target_shape)
            self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))
            return
        inp = self.ensure_operand(node.input[0])
        shape_name = node.input[1]
        target_shape = [int(v) for v in self.constant_array(shape_name).tolist()]
        out = self.builder.create_op(
            "top.Reshape",
            [inp],
            {
                "flatten_start_dim": "-1 : i64",
                "shape": to_int_array(target_shape),
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_transpose(self, node) -> None:
        attrs = self.attr_dict(node)
        if self.is_constant_value(node.input[0]):
            data = self.constant_array(node.input[0])
            perm = [int(v) for v in attrs.get("perm", [])]
            result = np.transpose(data, axes=perm)
            self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))
            return
        inp = self.ensure_operand(node.input[0])
        perm = [int(v) for v in attrs.get("perm", [])]
        out = self.builder.create_op(
            "top.Permute",
            [inp],
            {"order": to_int_array(perm)},
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def emit_slice(self, node) -> None:
        if self.is_constant_value(node.input[0]):
            data = self.constant_array(node.input[0])
            starts = self.constant_array(node.input[1]).astype(np.int64).tolist()
            ends = self.constant_array(node.input[2]).astype(np.int64).tolist()
            axes = (
                self.constant_array(node.input[3]).astype(np.int64).tolist()
                if len(node.input) > 3 and node.input[3]
                else list(range(len(starts)))
            )
            steps = (
                self.constant_array(node.input[4]).astype(np.int64).tolist()
                if len(node.input) > 4 and node.input[4]
                else [1] * len(starts)
            )
            slices = [slice(None)] * data.ndim
            for axis, start, end, step in zip(axes, starts, ends, steps):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            result = data[tuple(slices)]
            self.materialize_const_output(node.output[0], result, loc_name=self.node_loc_name(node, default_output=node.output[0]))
            return
        inp = self.ensure_operand(node.input[0])
        none_value = self.builder.ensure_none()
        starts = self.constant_array(node.input[1]).astype(np.int64).tolist()
        ends = self.constant_array(node.input[2]).astype(np.int64).tolist()
        axes = (
            self.constant_array(node.input[3]).astype(np.int64).tolist()
            if len(node.input) > 3 and node.input[3]
            else list(range(len(starts)))
        )
        steps = (
            self.constant_array(node.input[4]).astype(np.int64).tolist()
            if len(node.input) > 4 and node.input[4]
            else [1] * len(starts)
        )
        out = self.builder.create_op(
            "top.Slice",
            [inp, none_value, none_value, none_value],
            {
                "axes": to_int_array(axes),
                "ends": to_int_array(ends),
                "hasparamConvert_axes": "[]",
                "offset": to_int_array(starts),
                "steps": to_int_array(steps),
            },
            node.output[0],
            self.shape_of(node.output[0]),
            self.dtype_of(node.output[0]),
            loc_name=self.node_loc_name(node, default_output=node.output[0]),
        )
        self.builder.value_map[node.output[0]] = out

    def convert_node(self, node) -> None:
        op_type = node.op_type
        if op_type == "Constant":
            self.materialize_constant_node(node)
            return
        if op_type == "Identity":
            self.emit_identity(node)
            return
        if op_type == "Conv":
            self.emit_conv(node)
            return
        if op_type == "Shape":
            self.emit_shape(node)
            return
        if op_type == "Gather":
            self.emit_gather(node)
            return
        if op_type == "Unsqueeze":
            self.emit_unsqueeze(node)
            return
        if op_type == "Cast":
            self.emit_cast(node)
            return
        if op_type == "Range":
            self.emit_range(node)
            return
        if op_type == "Expand":
            self.emit_expand(node)
            return
        if op_type == "ConstantOfShape":
            self.emit_constant_of_shape(node)
            return
        if op_type == "Sigmoid":
            self.emit_sigmoid(node)
            return
        if op_type == "Mul":
            self.emit_binary(node, "top.Mul")
            return
        if op_type == "Add":
            self.emit_binary(node, "top.Add")
            return
        if op_type == "Concat":
            self.emit_concat(node)
            return
        if op_type == "MaxPool":
            self.emit_maxpool(node)
            return
        if op_type == "Resize":
            self.emit_resize(node)
            return
        if op_type == "Reshape":
            self.emit_reshape(node)
            return
        if op_type == "Transpose":
            self.emit_transpose(node)
            return
        if op_type == "Slice":
            self.emit_slice(node)
            return
        raise SystemExit(
            f"Unsupported ONNX op `{op_type}` at node `{node.name or node.output[0]}`. "
            "Extend the importer for this op before continuing."
        )

    def build(self) -> tuple[str, dict[str, np.ndarray]]:
        output_names = self.output_names()
        inputs = self.graph_inputs()
        arg_defs = []
        for index, value in enumerate(inputs):
            shape, dtype = self.value_info[value.name]
            if shape is None:
                raise SystemExit(f"Input shape missing for `{value.name}`")
            arg_type = tensor_type(shape, dtype)
            arg_name = f"%arg{index}"
            arg_defs.append(f"{arg_name}: {arg_type} loc({self.builder.loc_ref(value.name)})")
        output_types = [tensor_type(self.shape_of(name), self.dtype_of(name)) for name in output_names]
        if len(output_types) == 1:
            result_sig = output_types[0]
            return_sig = output_types[0]
        else:
            result_sig = "(" + ", ".join(output_types) + ")"
            return_sig = ", ".join(output_types)

        header = (
            f'module @{self.args.model_name} attributes '
            f'{{module.chip = "ALL", module.platform = "ONNX", module.state = "TOP_F32", '
            f'module.top_run_mode = "STATIC", module.weight_file = "{self.args.weight_file}"}} {{'
        )
        lines = [header]
        lines.append(f"  func.func @main({', '.join(arg_defs)}) -> {result_sig} {{")
        self.builder.lines = []

        for index, value in enumerate(inputs):
            shape, _dtype = self.value_info[value.name]
            created = self.builder.create_input(f"%arg{index}", tensor_type(shape), shape, value.name)
            self.builder.value_map[value.name] = created

        for node in self.selected_nodes(output_names):
            self.convert_node(node)

        result_values = [self.ensure_operand(name) for name in output_names]
        result_names = ", ".join(value.name for value in result_values)
        self.builder.emit(f"return {result_names} : {return_sig} loc({self.builder.loc_ref('return')})")

        lines.extend(self.builder.lines)
        lines.append(f"  }} loc({self.builder.loc_ref('main')})")
        lines.append(f"}} loc({self.builder.loc_ref(self.args.model_name)})")
        lines.extend(self.builder.loc_definitions())
        return ("\n".join(lines) + "\n", self.builder.weight_map)

    def print_summary(self) -> None:
        counts: dict[str, int] = {}
        for node in self.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        print("ONNX op summary:")
        for op_type in sorted(counts):
            print(f"  {op_type}: {counts[op_type]}")


def main() -> int:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    importer = OnnxToTopImporter(args)
    if args.dump_summary:
        importer.print_summary()
    mlir_text, weights = importer.build()

    mlir_path = args.workdir / args.mlir
    weight_path = args.workdir / args.weight_file
    mlir_path.write_text(mlir_text, encoding="utf-8")
    np.savez(weight_path, **weights)

    print(f"Wrote Top MLIR to: {normalize_path(mlir_path)}")
    print(f"Wrote weights to: {normalize_path(weight_path)}")
    print(f"Selected outputs: {', '.join(importer.output_names())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
