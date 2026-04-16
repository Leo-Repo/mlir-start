#!/usr/bin/env python3
"""ONNX -> mini_top importer built on MLIR IR objects.

This is the native-track counterpart to the archived text emitter in
`legacy_python_frontend/model_transform.py`.

Design goals:
- keep the importer in Python for fast iteration, similar to TPU-MLIR's frontend
- stop emitting raw MLIR text by string concatenation
- construct IR through `mlir.ir` objects and generic `Operation.create`
- only support the YOLOv5 operator subset we currently study
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "yolov5s.onnx"
DEFAULT_WORKDIR = REPO_ROOT / "experiments" / "04_mini_top_import"
DEFAULT_MLIR = "yolov5s_mini_top.mlir"
DEFAULT_WEIGHT = "yolov5s_mini_top_weights.npz"
DEFAULT_OUTPUTS = "350,498,646"


def import_onnx():
    try:
        import onnx
        from onnx import TensorProto, numpy_helper, shape_inference
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This importer requires the `onnx` package. Activate an environment "
            "that provides ONNX before running it."
        ) from exc
    return onnx, TensorProto, numpy_helper, shape_inference


def import_mlir():
    try:
        from mlir.ir import (
            ArrayAttr,
            Attribute,
            Context,
            FloatAttr,
            FunctionType,
            InsertionPoint,
            IntegerAttr,
            Location,
            Module,
            Operation,
            StringAttr,
            Type,
        )
        from mlir.dialects import func
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This importer requires MLIR Python bindings (`mlir.ir`). "
            "Build or activate an LLVM/MLIR Python environment first."
        ) from exc
    return {
        "ArrayAttr": ArrayAttr,
        "Attribute": Attribute,
        "Context": Context,
        "FloatAttr": FloatAttr,
        "FunctionType": FunctionType,
        "InsertionPoint": InsertionPoint,
        "IntegerAttr": IntegerAttr,
        "Location": Location,
        "Module": Module,
        "Operation": Operation,
        "StringAttr": StringAttr,
        "Type": Type,
        "func": func,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a YOLOv5-style ONNX model into mini_top using MLIR IR objects."
    )
    parser.add_argument("--model-def", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--mlir", default=DEFAULT_MLIR)
    parser.add_argument("--weight-file", default=DEFAULT_WEIGHT)
    parser.add_argument(
        "--output-names",
        default=DEFAULT_OUTPUTS,
        help="Comma-separated output names. Empty means graph outputs.",
    )
    parser.add_argument("--dump-summary", action="store_true")
    return parser.parse_args()


def mlir_elem_type(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return "f32"
    if dtype == np.float16:
        return "f16"
    if dtype == np.float64:
        return "f64"
    if dtype == np.int64:
        return "i64"
    if dtype == np.int32:
        return "i32"
    if dtype == np.int16:
        return "i16"
    if dtype == np.int8:
        return "i8"
    if dtype == np.bool_:
        return "i1"
    raise SystemExit(f"Unsupported dtype for mini_top importer: {dtype}")


def tensor_type_str(shape: list[int] | None, dtype: np.dtype = np.float32) -> str:
    elem = mlir_elem_type(dtype)
    if shape is None:
        return f"tensor<*x{elem}>"
    if len(shape) == 0:
        return f"tensor<{elem}>"
    dims = "x".join("?" if dim is None or dim < 0 else str(int(dim)) for dim in shape)
    return f"tensor<{dims}x{elem}>"


def onnx_dtype_to_numpy(elem_type: int, tensor_proto) -> np.dtype:
    mapping = {
        tensor_proto.FLOAT: np.float32,
        tensor_proto.FLOAT16: np.float16,
        tensor_proto.DOUBLE: np.float64,
        tensor_proto.INT64: np.int64,
        tensor_proto.INT32: np.int32,
        tensor_proto.INT16: np.int16,
        tensor_proto.INT8: np.int8,
        tensor_proto.BOOL: np.bool_,
    }
    try:
        return np.dtype(mapping[elem_type])
    except KeyError as exc:
        raise SystemExit(f"Unsupported ONNX elem_type: {elem_type}") from exc


def value_info_shape(v) -> list[int] | None:
    tensor_type = v.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    return dims


def broadcast_shape(lhs: list[int] | None, rhs: list[int] | None) -> list[int] | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    result: list[int] = []
    for a, b in zip(reversed(lhs), reversed(rhs)):
        if a == 1:
            result.append(b)
        elif b == 1 or a == b:
            result.append(a)
        else:
            result.append(-1)
    longer = lhs if len(lhs) > len(rhs) else rhs
    result.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
    return list(reversed(result))


def conv_out_dim(input_dim: int, kernel: int, stride: int, pad0: int, pad1: int, dilation: int) -> int:
    effective_kernel = dilation * (kernel - 1) + 1
    return ((input_dim + pad0 + pad1 - effective_kernel) // stride) + 1


@dataclass
class MlirValue:
    value: Any
    shape: list[int] | None
    dtype: np.dtype


class OnnxToMiniTopImporter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        onnx, tensor_proto, numpy_helper, shape_inference = import_onnx()
        self.onnx = onnx
        self.tensor_proto = tensor_proto
        self.numpy_helper = numpy_helper
        self.shape_inference = shape_inference
        self.mlir = import_mlir()

        self.model = self._load_model(args.model_def)
        self.graph = self.model.graph
        self.initializers = {
            init.name: self.numpy_helper.to_array(init) for init in self.graph.initializer
        }
        self.initializer_names = set(self.initializers)
        self.producer_of: dict[str, Any] = {}
        self.value_shapes: dict[str, list[int] | None] = {}
        self.value_dtypes: dict[str, np.dtype] = {}
        self.constant_cache: dict[str, np.ndarray] = {}
        self.values: dict[str, MlirValue] = {}
        self.weight_values: dict[str, MlirValue] = {}
        self.weight_arrays: dict[str, np.ndarray] = {}

        self._collect_model_metadata()
        self.output_names = self._resolve_output_names(args.output_names)

    def _load_model(self, path: Path):
        model = self.onnx.load(str(path))
        try:
            model = self.shape_inference.infer_shapes(model)
        except Exception:
            pass
        return model

    def _collect_model_metadata(self) -> None:
        for init in self.graph.initializer:
            arr = self.numpy_helper.to_array(init)
            self.value_shapes[init.name] = list(arr.shape)
            self.value_dtypes[init.name] = arr.dtype
        for value in list(self.graph.input) + list(self.graph.value_info) + list(self.graph.output):
            self.value_shapes[value.name] = value_info_shape(value)
            self.value_dtypes[value.name] = onnx_dtype_to_numpy(
                value.type.tensor_type.elem_type, self.tensor_proto
            )
        for node in self.graph.node:
            for out in node.output:
                self.producer_of[out] = node

    def _resolve_output_names(self, raw: str) -> list[str]:
        if raw.strip():
            return [name.strip() for name in raw.split(",") if name.strip()]
        return [out.name for out in self.graph.output]

    def selected_nodes(self) -> list[Any]:
        ordered: list[Any] = []
        visited_nodes: set[int] = set()

        def visit_value(value_name: str) -> None:
            if not value_name or value_name in self.initializer_names:
                return
            node = self.producer_of.get(value_name)
            if node is None:
                return
            node_id = id(node)
            if node_id in visited_nodes:
                return
            visited_nodes.add(node_id)
            for inp in node.input:
                visit_value(inp)
            ordered.append(node)

        for out_name in self.output_names:
            visit_value(out_name)
        return ordered

    def node_loc_name(self, node: Any) -> str:
        if getattr(node, "name", ""):
            return node.name
        return node.output[0]

    def attr_dict(self, node: Any) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for attr in node.attribute:
            attrs[attr.name] = self.onnx.helper.get_attribute_value(attr)
        return attrs

    def ensure_mlir_type(self, shape: list[int] | None, dtype: np.dtype = np.float32):
        return self.mlir["Type"].parse(tensor_type_str(shape, dtype))

    def i64_attr(self, value: int):
        i64 = self.mlir["Type"].parse("i64")
        return self.mlir["IntegerAttr"].get(i64, int(value))

    def f64_attr(self, value: float):
        f64 = self.mlir["Type"].parse("f64")
        return self.mlir["FloatAttr"].get(f64, float(value))

    def array_i64_attr(self, values: list[int]):
        return self.mlir["ArrayAttr"].get([self.i64_attr(v) for v in values])

    def string_attr(self, value: str):
        return self.mlir["StringAttr"].get(value)

    def create_op(
        self,
        name: str,
        result_shapes: list[list[int] | None],
        operands: list[Any],
        attrs: dict[str, Any],
        result_dtypes: list[np.dtype] | None = None,
        loc_name: str | None = None,
    ):
        result_dtypes = result_dtypes or [np.float32] * len(result_shapes)
        result_types = [
            self.ensure_mlir_type(shape, dtype)
            for shape, dtype in zip(result_shapes, result_dtypes, strict=True)
        ]
        loc = self.mlir["Location"].name(loc_name or name)
        op = self.mlir["Operation"].create(
            name, results=result_types, operands=operands, attributes=attrs, loc=loc
        )
        return list(op.results)

    def shape_of(self, name: str) -> list[int] | None:
        return self.value_shapes.get(name)

    def dtype_of(self, name: str) -> np.dtype:
        return self.value_dtypes.get(name, np.dtype(np.float32))

    def set_value_info(self, name: str, shape: list[int] | None, dtype: np.dtype | None = None) -> None:
        self.value_shapes[name] = shape
        if dtype is not None:
            self.value_dtypes[name] = np.dtype(dtype)

    def ensure_operand(self, name: str) -> MlirValue:
        if name in self.values:
            return self.values[name]
        if name in self.initializers:
            return self.materialize_weight(name, self.initializers[name])
        raise KeyError(f"Unknown operand: {name}")

    def materialize_weight(self, key: str, array: np.ndarray) -> MlirValue:
        if key in self.weight_values:
            return self.weight_values[key]
        results = self.create_op(
            "mini_top.weight",
            [list(array.shape)],
            [],
            {"weight_key": self.string_attr(key)},
            [array.dtype],
            loc_name=key,
        )
        value = MlirValue(results[0], list(array.shape), array.dtype)
        self.weight_values[key] = value
        self.values[key] = value
        self.weight_arrays[key] = array
        return value

    def zero_bias(self, channels: int, key: str) -> MlirValue:
        return self.materialize_weight(f"{key}_auto_bias", np.zeros((channels,), dtype=np.float32))

    def constant_array(self, name: str) -> np.ndarray | None:
        if name in self.constant_cache:
            return self.constant_cache[name]
        if name in self.initializers:
            arr = self.initializers[name]
            self.constant_cache[name] = arr
            return arr

        node = self.producer_of.get(name)
        if node is None:
            return None
        attrs = self.attr_dict(node)
        result: np.ndarray | None = None

        if node.op_type == "Constant":
            result = self.numpy_helper.to_array(attrs["value"])
        elif node.op_type == "Shape":
            shape = self.shape_of(node.input[0])
            if shape is not None and all(dim is not None and dim >= 0 for dim in shape):
                result = np.asarray(shape, dtype=np.int64)
        elif node.op_type == "Gather":
            data = self.constant_array(node.input[0])
            indices = self.constant_array(node.input[1])
            if data is not None and indices is not None:
                result = np.take(data, indices.astype(np.int64), axis=int(attrs.get("axis", 0)))
        elif node.op_type == "Unsqueeze":
            data = self.constant_array(node.input[0])
            if data is not None:
                axes = attrs.get("axes")
                if axes is None and len(node.input) > 1:
                    axes_arr = self.constant_array(node.input[1])
                    axes = [] if axes_arr is None else [int(v) for v in axes_arr.tolist()]
                axes = [] if axes is None else [int(v) for v in axes]
                result = data
                for axis in sorted(axes):
                    result = np.expand_dims(result, axis)
        elif node.op_type == "Cast":
            data = self.constant_array(node.input[0])
            if data is not None:
                result = data.astype(onnx_dtype_to_numpy(int(attrs["to"]), self.tensor_proto), copy=False)
        elif node.op_type == "Concat":
            arrays = [self.constant_array(inp) for inp in node.input]
            if all(arr is not None for arr in arrays):
                result = np.concatenate(arrays, axis=int(attrs.get("axis", 0)))

        if result is not None:
            self.constant_cache[name] = result
        return result

    def infer_conv_shape(
        self,
        input_shape: list[int] | None,
        weight_shape: list[int] | None,
        pads: list[int],
        strides: list[int],
        dilations: list[int],
    ) -> list[int] | None:
        if input_shape is None or weight_shape is None or len(input_shape) != 4 or len(weight_shape) != 4:
            return self.shape_of("")
        n = input_shape[0]
        c_out = weight_shape[0]
        h = conv_out_dim(input_shape[2], weight_shape[2], strides[0], pads[0], pads[2], dilations[0])
        w = conv_out_dim(input_shape[3], weight_shape[3], strides[1], pads[1], pads[3], dilations[1])
        return [n, c_out, h, w]

    def infer_concat_shape(self, shapes: list[list[int] | None], axis: int) -> list[int] | None:
        if not shapes or any(shape is None for shape in shapes):
            return None
        base = list(shapes[0])
        total = 0
        for shape in shapes:
            total += shape[axis]
        base[axis] = total
        return base

    def infer_maxpool_shape(
        self, input_shape: list[int] | None, kernel: list[int], strides: list[int], pads: list[int]
    ) -> list[int] | None:
        if input_shape is None or len(input_shape) != 4:
            return None
        return [
            input_shape[0],
            input_shape[1],
            conv_out_dim(input_shape[2], kernel[0], strides[0], pads[0], pads[2], 1),
            conv_out_dim(input_shape[3], kernel[1], strides[1], pads[1], pads[3], 1),
        ]

    def infer_reshape_shape(self, input_shape: list[int] | None, target: list[int]) -> list[int]:
        if input_shape is None:
            return target
        output = list(target)
        known_product = 1
        unknown_index = None
        input_product = math.prod(int(v) for v in input_shape)
        for index, dim in enumerate(output):
            if dim == 0:
                output[index] = input_shape[index]
                known_product *= int(output[index])
            elif dim == -1:
                unknown_index = index
            else:
                known_product *= int(dim)
        if unknown_index is not None and known_product != 0:
            output[unknown_index] = int(input_product // known_product)
        return output

    def infer_transpose_shape(self, input_shape: list[int] | None, order: list[int]) -> list[int] | None:
        if input_shape is None:
            return None
        return [input_shape[index] for index in order]

    def infer_interp_shape(
        self, input_shape: list[int] | None, target_h: int, target_w: int
    ) -> list[int] | None:
        if input_shape is None or len(input_shape) != 4:
            return None
        return [input_shape[0], input_shape[1], target_h, target_w]

    def convert_constant_only(self, node: Any) -> bool:
        if self.constant_array(node.output[0]) is None:
            return False
        result = self.constant_array(node.output[0])
        self.set_value_info(node.output[0], list(result.shape), result.dtype)
        return True

    def analyze_node(self, node: Any) -> None:
        attrs = self.attr_dict(node)
        if node.op_type in {"Constant", "Shape", "Gather", "Unsqueeze", "Cast"}:
            self.convert_constant_only(node)
            return
        if node.op_type == "Conv":
            input_shape = self.shape_of(node.input[0])
            weight_shape = self.shape_of(node.input[1])
            pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
            strides = [int(v) for v in attrs.get("strides", [1, 1])]
            dilations = [int(v) for v in attrs.get("dilations", [1, 1])]
            self.set_value_info(
                node.output[0],
                self.infer_conv_shape(input_shape, weight_shape, pads, strides, dilations),
                self.dtype_of(node.input[0]),
            )
        elif node.op_type in {"Sigmoid", "Identity"}:
            self.set_value_info(node.output[0], self.shape_of(node.input[0]), self.dtype_of(node.input[0]))
        elif node.op_type in {"Mul", "Add"}:
            lhs_shape = self.shape_of(node.input[0])
            rhs_shape = self.shape_of(node.input[1])
            self.set_value_info(
                node.output[0],
                broadcast_shape(lhs_shape, rhs_shape),
                np.result_type(self.dtype_of(node.input[0]), self.dtype_of(node.input[1])),
            )
        elif node.op_type == "Concat":
            if not self.convert_constant_only(node):
                self.set_value_info(
                    node.output[0],
                    self.infer_concat_shape([self.shape_of(name) for name in node.input], int(attrs.get("axis", 1))),
                    self.dtype_of(node.input[0]),
                )
        elif node.op_type == "MaxPool":
            self.set_value_info(
                node.output[0],
                self.infer_maxpool_shape(
                    self.shape_of(node.input[0]),
                    [int(v) for v in attrs.get("kernel_shape", [1, 1])],
                    [int(v) for v in attrs.get("strides", [1, 1])],
                    [int(v) for v in attrs.get("pads", [0, 0, 0, 0])],
                ),
                self.dtype_of(node.input[0]),
            )
        elif node.op_type == "Resize":
            inp_shape = self.shape_of(node.input[0])
            sizes = self.constant_array(node.input[3]) if len(node.input) >= 4 and node.input[3] else None
            scales = self.constant_array(node.input[2]) if len(node.input) >= 3 and node.input[2] else None
            if sizes is not None:
                target_h, target_w = int(sizes[-2]), int(sizes[-1])
            elif scales is not None and inp_shape is not None:
                target_h = int(round(inp_shape[-2] * float(scales[-2])))
                target_w = int(round(inp_shape[-1] * float(scales[-1])))
            else:
                target_h = target_w = -1
            self.set_value_info(
                node.output[0],
                self.infer_interp_shape(inp_shape, target_h, target_w),
                self.dtype_of(node.input[0]),
            )
        elif node.op_type == "Reshape":
            target_arr = self.constant_array(node.input[1])
            target_shape = None if target_arr is None else [int(v) for v in target_arr.tolist()]
            self.set_value_info(
                node.output[0],
                None if target_shape is None else self.infer_reshape_shape(self.shape_of(node.input[0]), target_shape),
                self.dtype_of(node.input[0]),
            )
        elif node.op_type == "Transpose":
            self.set_value_info(
                node.output[0],
                self.infer_transpose_shape(self.shape_of(node.input[0]), [int(v) for v in attrs.get("perm", [])]),
                self.dtype_of(node.input[0]),
            )
        else:
            raise SystemExit(f"Unsupported ONNX op in shape analysis: {node.op_type} ({self.node_loc_name(node)})")

    def analyze_selected_nodes(self) -> None:
        for node in self.selected_nodes():
            self.analyze_node(node)

    def emit_conv(self, node: Any) -> None:
        attrs = self.attr_dict(node)
        data = self.ensure_operand(node.input[0])
        weight = self.ensure_operand(node.input[1])
        if len(node.input) >= 3 and node.input[2]:
            bias = self.ensure_operand(node.input[2])
        else:
            bias = self.zero_bias(weight.shape[0], node.output[0])
        pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
        strides = [int(v) for v in attrs.get("strides", [1, 1])]
        dilations = [int(v) for v in attrs.get("dilations", [1, 1])]
        out_shape = self.infer_conv_shape(data.shape, weight.shape, pads, strides, dilations)
        self.set_value_info(node.output[0], out_shape, data.dtype)
        result = self.create_op(
            "mini_top.conv",
            [out_shape],
            [data.value, weight.value, bias.value],
            {},
            [data.dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.values[node.output[0]] = MlirValue(result, out_shape, data.dtype)

    def emit_sigmoid(self, node: Any) -> None:
        inp = self.ensure_operand(node.input[0])
        result = self.create_op(
            "mini_top.sigmoid",
            [inp.shape],
            [inp.value],
            {},
            [inp.dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.set_value_info(node.output[0], inp.shape, inp.dtype)
        self.values[node.output[0]] = MlirValue(result, inp.shape, inp.dtype)

    def emit_binary(self, node: Any, op_name: str) -> None:
        lhs = self.ensure_operand(node.input[0])
        rhs = self.ensure_operand(node.input[1])
        out_shape = broadcast_shape(lhs.shape, rhs.shape)
        out_dtype = np.result_type(lhs.dtype, rhs.dtype)
        result = self.create_op(
            op_name,
            [out_shape],
            [lhs.value, rhs.value],
            {},
            [out_dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.set_value_info(node.output[0], out_shape, out_dtype)
        self.values[node.output[0]] = MlirValue(result, out_shape, out_dtype)

    def emit_concat(self, node: Any) -> None:
        attrs = self.attr_dict(node)
        operands = [self.ensure_operand(name) for name in node.input]
        axis = int(attrs.get("axis", 1))
        out_shape = self.infer_concat_shape([item.shape for item in operands], axis)
        result = self.create_op(
            "mini_top.concat",
            [out_shape],
            [item.value for item in operands],
            {"axis": self.i64_attr(axis)},
            [operands[0].dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.set_value_info(node.output[0], out_shape, operands[0].dtype)
        self.values[node.output[0]] = MlirValue(result, out_shape, operands[0].dtype)

    def emit_maxpool(self, node: Any) -> None:
        attrs = self.attr_dict(node)
        inp = self.ensure_operand(node.input[0])
        kernel = [int(v) for v in attrs.get("kernel_shape", [1, 1])]
        strides = [int(v) for v in attrs.get("strides", [1, 1])]
        pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
        out_shape = self.infer_maxpool_shape(inp.shape, kernel, strides, pads)
        result = self.create_op(
            "mini_top.maxpool",
            [out_shape],
            [inp.value],
            {
                "kernel_shape": self.array_i64_attr(kernel),
                "strides": self.array_i64_attr(strides),
                "pads": self.array_i64_attr(pads),
            },
            [inp.dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.set_value_info(node.output[0], out_shape, inp.dtype)
        self.values[node.output[0]] = MlirValue(result, out_shape, inp.dtype)

    def emit_resize(self, node: Any) -> None:
        attrs = self.attr_dict(node)
        inp = self.ensure_operand(node.input[0])
        sizes = None
        scales = None
        if len(node.input) >= 4 and node.input[3]:
            sizes = self.constant_array(node.input[3])
        if len(node.input) >= 3 and node.input[2]:
            scales = self.constant_array(node.input[2])
        if sizes is not None:
            target_h = int(sizes[-2])
            target_w = int(sizes[-1])
        elif scales is not None and inp.shape is not None:
            target_h = int(round(inp.shape[-2] * float(scales[-2])))
            target_w = int(round(inp.shape[-1] * float(scales[-1])))
        else:
            raise SystemExit(f"Resize needs constant sizes/scales: {self.node_loc_name(node)}")
        out_shape = self.infer_interp_shape(inp.shape, target_h, target_w)
        result = self.create_op(
            "mini_top.interp",
            [out_shape],
            [inp.value],
            {
                "target_h": self.i64_attr(target_h),
                "target_w": self.i64_attr(target_w),
                "mode": self.string_attr(str(attrs.get("mode", b"nearest")).replace("b'", "").replace("'", "")),
            },
            [inp.dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.set_value_info(node.output[0], out_shape, inp.dtype)
        self.values[node.output[0]] = MlirValue(result, out_shape, inp.dtype)

    def emit_reshape(self, node: Any) -> None:
        inp = self.ensure_operand(node.input[0])
        target_arr = self.constant_array(node.input[1])
        if target_arr is None:
            raise SystemExit(f"Reshape shape input must be constant for now: {self.node_loc_name(node)}")
        target_shape = [int(v) for v in target_arr.tolist()]
        out_shape = self.infer_reshape_shape(inp.shape, target_shape)
        result = self.create_op(
            "mini_top.reshape",
            [out_shape],
            [inp.value],
            {"shape": self.array_i64_attr(target_shape)},
            [inp.dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.set_value_info(node.output[0], out_shape, inp.dtype)
        self.values[node.output[0]] = MlirValue(result, out_shape, inp.dtype)

    def emit_transpose(self, node: Any) -> None:
        attrs = self.attr_dict(node)
        inp = self.ensure_operand(node.input[0])
        order = [int(v) for v in attrs.get("perm", [])]
        out_shape = self.infer_transpose_shape(inp.shape, order)
        result = self.create_op(
            "mini_top.permute",
            [out_shape],
            [inp.value],
            {"order": self.array_i64_attr(order)},
            [inp.dtype],
            loc_name=self.node_loc_name(node),
        )[0]
        self.set_value_info(node.output[0], out_shape, inp.dtype)
        self.values[node.output[0]] = MlirValue(result, out_shape, inp.dtype)

    def emit_identity(self, node: Any) -> None:
        self.values[node.output[0]] = self.ensure_operand(node.input[0])
        self.set_value_info(node.output[0], self.shape_of(node.input[0]), self.dtype_of(node.input[0]))

    def convert_node(self, node: Any) -> None:
        if node.op_type in {"Constant", "Shape", "Gather", "Unsqueeze", "Cast"}:
            self.convert_constant_only(node)
            return
        if node.op_type == "Conv":
            self.emit_conv(node)
        elif node.op_type == "Sigmoid":
            self.emit_sigmoid(node)
        elif node.op_type == "Mul":
            self.emit_binary(node, "mini_top.mul")
        elif node.op_type == "Add":
            self.emit_binary(node, "mini_top.add")
        elif node.op_type == "Concat":
            if not self.convert_constant_only(node):
                self.emit_concat(node)
        elif node.op_type == "MaxPool":
            self.emit_maxpool(node)
        elif node.op_type == "Resize":
            self.emit_resize(node)
        elif node.op_type == "Reshape":
            self.emit_reshape(node)
        elif node.op_type == "Transpose":
            self.emit_transpose(node)
        elif node.op_type == "Identity":
            self.emit_identity(node)
        else:
            raise SystemExit(
                f"Unsupported ONNX op for mini_top importer: {node.op_type} ({self.node_loc_name(node)})"
            )

    def build_module(self):
        ctx = self.mlir["Context"]()
        ctx.allow_unregistered_dialects = True
        with ctx, self.mlir["Location"].unknown():
            module = self.mlir["Module"].create()
            module.operation.attributes["mini_top.weight_file"] = self.string_attr(
                str((self.args.workdir / self.args.weight_file).resolve())
            )

            graph_inputs = [
                value for value in self.graph.input if value.name not in self.initializer_names
            ]
            self.analyze_selected_nodes()
            input_types = [
                self.ensure_mlir_type(self.shape_of(inp.name), self.dtype_of(inp.name))
                for inp in graph_inputs
            ]
            output_types = [
                self.ensure_mlir_type(self.shape_of(name), self.dtype_of(name))
                for name in self.output_names
            ]

            selected = self.selected_nodes()
            with self.mlir["InsertionPoint"](module.body):
                function_type = self.mlir["FunctionType"].get(input_types, output_types)
                func_op = self.mlir["func"].FuncOp("main", function_type)
                entry = func_op.add_entry_block(
                    arg_locs=[self.mlir["Location"].name(inp.name) for inp in graph_inputs]
                )
                with self.mlir["InsertionPoint"](entry):
                    for inp, arg in zip(graph_inputs, entry.arguments, strict=True):
                        self.values[inp.name] = MlirValue(arg, self.shape_of(inp.name), self.dtype_of(inp.name))
                    for node in selected:
                        self.convert_node(node)

                    output_values = [self.values[name] for name in self.output_names]
                    self.mlir["func"].ReturnOp([item.value for item in output_values])
            return module

    def dump_summary(self) -> None:
        print("mini_top importer summary")
        print(f"  model: {self.args.model_def}")
        print(f"  outputs: {', '.join(self.output_names)}")
        print(f"  graph nodes selected: {len(self.selected_nodes())}")
        print(f"  initializers: {len(self.initializers)}")

    def run(self) -> tuple[str, Path]:
        if self.args.dump_summary:
            self.dump_summary()
        self.args.workdir.mkdir(parents=True, exist_ok=True)
        module = self.build_module()
        mlir_text = module.operation.get_asm(enable_debug_info=True)
        mlir_path = self.args.workdir / self.args.mlir
        weight_path = self.args.workdir / self.args.weight_file
        mlir_path.write_text(mlir_text)
        np.savez(weight_path, **self.weight_arrays)
        return mlir_text, mlir_path


def main() -> None:
    args = parse_args()
    importer = OnnxToMiniTopImporter(args)
    _, mlir_path = importer.run()
    print(f"mini_top MLIR written to: {mlir_path}")


if __name__ == "__main__":
    main()
