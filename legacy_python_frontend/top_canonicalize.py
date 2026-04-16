#!/usr/bin/env python3
"""Local Top MLIR canonicalize pass for the YOLOv5 learning workflow.

This is intentionally narrower than TPU-MLIR's full pass pipeline. It only
implements a few canonicalization patterns that are useful for the locally
generated `yolov5s.mlir`:

- sanitize byte-string attributes such as `"b'nearest'" -> "nearest"`
- infer static result shapes for `top.MaxPool`, `top.Interp`, `top.Concat`,
  `top.Reshape`, and `top.Permute`
- remove no-op `top.Reshape`
- remove identity `top.Permute`

The goal is to keep a readable "canonicalized" Top MLIR without depending on
`tpuc-opt`.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


OP_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%[\w\d]+)\s*=\s*"(?P<op>[^"]+)"'
    r'\((?P<operands>.*?)\)(?P<attrs>\s*\{.*\})?\s*:\s*'
    r'\((?P<input_types>.*?)\)\s*->\s*(?P<output_type>.+?)\s+loc\((?P<loc>[^)]+)\)\s*$'
)
RETURN_RE = re.compile(r"^(?P<indent>\s*)return\s+(?P<values>.+?)\s*:\s*(?P<types>.+?)\s+loc\((?P<loc>[^)]+)\)\s*$")
FUNC_RE = re.compile(r"^(?P<prefix>\s*func\.func @main\(.*?\) -> )(?P<sig>\(.+\)|tensor<.+>)\s*\{\s*$")
TENSOR_RE = re.compile(r"^tensor<(?P<body>.+)>$")


@dataclass
class OpLine:
    indent: str
    result: str
    op: str
    operands: list[str]
    attrs: list[tuple[str, str]]
    input_types: list[str]
    output_type: str
    loc: str

    def to_line(self) -> str:
        operand_text = ", ".join(self.operands)
        attr_text = ""
        if self.attrs:
            attr_text = " {" + ", ".join(f"{key} = {value}" for key, value in self.attrs) + "}"
        input_text = ", ".join(self.input_types)
        return (
            f'{self.indent}{self.result} = "{self.op}"({operand_text}){attr_text} : '
            f'({input_text}) -> {self.output_type} loc({self.loc})'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize locally generated Top MLIR.")
    parser.add_argument("--input", type=Path, required=True, help="Input raw Top MLIR file.")
    parser.add_argument("--output", type=Path, required=True, help="Output canonicalized Top MLIR file.")
    return parser.parse_args()


def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth_square = 0
    depth_angle = 0
    depth_brace = 0
    in_string = False
    escape = False
    for char in text:
        if in_string:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            current.append(char)
            continue
        if char == "[":
            depth_square += 1
        elif char == "]":
            depth_square -= 1
        elif char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        if char == sep and depth_square == 0 and depth_angle == 0 and depth_brace == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_attrs(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] != "{" or stripped[-1] != "}":
        raise ValueError(f"Malformed attr block: {text}")
    body = stripped[1:-1].strip()
    if not body:
        return []
    result: list[tuple[str, str]] = []
    for item in split_top_level(body):
        key, value = item.split("=", 1)
        result.append((key.strip(), value.strip()))
    return result


def parse_op_line(line: str) -> OpLine | None:
    match = OP_RE.match(line)
    if not match:
        return None
    operands = [item for item in split_top_level(match.group("operands")) if item]
    input_types = [item for item in split_top_level(match.group("input_types")) if item]
    return OpLine(
        indent=match.group("indent"),
        result=match.group("result"),
        op=match.group("op"),
        operands=operands,
        attrs=parse_attrs(match.group("attrs")),
        input_types=input_types,
        output_type=match.group("output_type").strip(),
        loc=match.group("loc").strip(),
    )


def parse_tensor_type(type_str: str) -> tuple[list[int | None] | None, str]:
    if type_str == "none":
        return None, "none"
    match = TENSOR_RE.match(type_str.strip())
    if not match:
        raise ValueError(f"Unsupported type syntax: {type_str}")
    body = match.group("body")
    if "x" not in body:
        return [], body
    dims_text, elem = body.rsplit("x", 1)
    dims: list[int | None] = []
    for dim in dims_text.split("x"):
        dim = dim.strip()
        if dim == "?":
            dims.append(None)
        else:
            dims.append(int(dim))
    return dims, elem


def tensor_type(dims: list[int | None] | None, elem: str) -> str:
    if dims is None:
        return "none"
    if len(dims) == 0:
        return f"tensor<{elem}>"
    return "tensor<" + "x".join("?" if dim is None else str(int(dim)) for dim in dims) + f"x{elem}>"


def parse_int_array(text: str) -> list[int]:
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError(f"Expected integer array attr, got: {text}")
    body = stripped[1:-1].strip()
    if not body:
        return []
    return [int(part.strip()) for part in body.split(",")]


def parse_bool(text: str) -> bool:
    stripped = text.strip()
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    raise ValueError(f"Expected bool attr, got: {text}")


def parse_float_attr(text: str) -> float:
    return float(text.split(":", 1)[0].strip())


def attr_map(attrs: list[tuple[str, str]]) -> dict[str, str]:
    return {key: value for key, value in attrs}


def sanitize_attr_value(value: str) -> str:
    match = re.fullmatch(r'"b\'([^\']*)\'"', value.strip())
    if match:
        return f'"{match.group(1)}"'
    return value


def conv_like_out_dim(
    input_dim: int | None,
    kernel: int,
    stride: int,
    pad_before: int,
    pad_after: int,
    ceil_mode: bool = False,
) -> int | None:
    if input_dim is None:
        return None
    numerator = int(input_dim) + int(pad_before) + int(pad_after) - int(kernel)
    if ceil_mode:
        return math.floor((numerator + int(stride) - 1) / int(stride)) + 1
    return math.floor(numerator / int(stride)) + 1


def infer_maxpool_type(input_type: str, attrs: list[tuple[str, str]]) -> str:
    dims, elem = parse_tensor_type(input_type)
    if dims is None or len(dims) != 4:
        return input_type
    amap = attr_map(attrs)
    kernel = parse_int_array(amap.get("kernel_shape", "[1, 1]"))
    pads = parse_int_array(amap.get("pads", "[0, 0, 0, 0]"))
    strides = parse_int_array(amap.get("strides", "[1, 1]"))
    ceil_mode = parse_bool(amap.get("ceil_mode", "false"))
    out = [
        dims[0],
        dims[1],
        conv_like_out_dim(dims[2], kernel[0], strides[0], pads[0], pads[2], ceil_mode),
        conv_like_out_dim(dims[3], kernel[1], strides[1], pads[1], pads[3], ceil_mode),
    ]
    return tensor_type(out, elem)


def infer_interp_type(input_type: str, attrs: list[tuple[str, str]]) -> str:
    dims, elem = parse_tensor_type(input_type)
    if dims is None or len(dims) != 4:
        return input_type
    amap = attr_map(attrs)
    scale_h = parse_float_attr(amap.get("scale_h", "1.0"))
    scale_w = parse_float_attr(amap.get("scale_w", "1.0"))
    out = list(dims)
    if out[2] is not None:
        out[2] = int(round(out[2] * scale_h))
    if out[3] is not None:
        out[3] = int(round(out[3] * scale_w))
    return tensor_type(out, elem)


def infer_concat_type(input_types: list[str], attrs: list[tuple[str, str]], fallback: str) -> str:
    tensors = [parse_tensor_type(item) for item in input_types]
    dims_list = [dims for dims, _ in tensors if dims is not None]
    if not dims_list:
        return fallback
    elem = tensors[0][1]
    rank = len(dims_list[0])
    amap = attr_map(attrs)
    axis = int(amap.get("axis", "1 : si32").split(":", 1)[0].strip())
    if axis < 0:
        axis += rank
    out = list(dims_list[0])
    for dim_index in range(rank):
        if dim_index == axis:
            total = 0
            for dims in dims_list:
                if dims[dim_index] is None:
                    total = None
                    break
                total += int(dims[dim_index])
            out[dim_index] = total
            continue
        ref = out[dim_index]
        for dims in dims_list[1:]:
            if ref is None or dims[dim_index] is None:
                ref = ref if ref == dims[dim_index] else None
            elif int(ref) != int(dims[dim_index]):
                ref = None
        out[dim_index] = ref
    return tensor_type(out, elem)


def infer_reshape_type(input_type: str, attrs: list[tuple[str, str]]) -> str:
    dims, elem = parse_tensor_type(input_type)
    if dims is None:
        return input_type
    amap = attr_map(attrs)
    target = parse_int_array(amap["shape"])
    out: list[int | None] = []
    known_product = 1
    unknown_index = None
    for idx, dim in enumerate(target):
        if dim == 0 and idx < len(dims):
            out.append(dims[idx])
            if dims[idx] is not None:
                known_product *= int(dims[idx])
        elif dim == -1:
            out.append(None)
            unknown_index = idx
        else:
            out.append(int(dim))
            known_product *= int(dim)
    if unknown_index is not None and all(dim is not None for dim in dims):
        input_product = 1
        for dim in dims:
            input_product *= int(dim)
        if known_product != 0:
            out[unknown_index] = input_product // known_product
    return tensor_type(out, elem)


def infer_permute_type(input_type: str, attrs: list[tuple[str, str]]) -> str:
    dims, elem = parse_tensor_type(input_type)
    if dims is None:
        return input_type
    amap = attr_map(attrs)
    order = parse_int_array(amap["order"])
    if len(order) != len(dims):
        return input_type
    out = [dims[index] for index in order]
    return tensor_type(out, elem)


def canonicalize_ops(ops: list[OpLine]) -> tuple[list[OpLine], dict[str, str]]:
    replacements: dict[str, str] = {}
    result_types: dict[str, str] = {}
    kept: list[OpLine] = []

    def resolve(value: str) -> str:
        while value in replacements:
            value = replacements[value]
        return value

    for op in ops:
        op.operands = [resolve(operand) for operand in op.operands]
        rewritten_input_types: list[str] = []
        for operand, old_type in zip(op.operands, op.input_types):
            rewritten_input_types.append(result_types.get(operand, old_type))
        op.input_types = rewritten_input_types
        op.attrs = [(key, sanitize_attr_value(value)) for key, value in op.attrs]
        if op.op == "top.MaxPool" and op.input_types:
            op.output_type = infer_maxpool_type(op.input_types[0], op.attrs)
        elif op.op == "top.Interp" and op.input_types:
            op.output_type = infer_interp_type(op.input_types[0], op.attrs)
        elif op.op == "top.Concat":
            op.output_type = infer_concat_type(op.input_types, op.attrs, op.output_type)
        elif op.op == "top.Reshape" and op.input_types:
            op.output_type = infer_reshape_type(op.input_types[0], op.attrs)
        elif op.op == "top.Permute" and op.input_types:
            op.output_type = infer_permute_type(op.input_types[0], op.attrs)

        if op.op == "top.Reshape" and op.input_types and op.output_type == op.input_types[0]:
            replacements[op.result] = op.operands[0]
            continue
        if op.op == "top.Permute":
            amap = attr_map(op.attrs)
            order = parse_int_array(amap.get("order", "[]"))
            if order == list(range(len(order))) and op.input_types and op.output_type == op.input_types[0]:
                replacements[op.result] = op.operands[0]
                continue

        result_types[op.result] = op.output_type
        kept.append(op)

    for op in kept:
        op.operands = [resolve(operand) for operand in op.operands]
        op.input_types = [result_types.get(operand, old_type) for operand, old_type in zip(op.operands, op.input_types)]
    return kept, replacements


def canonicalize_mlir(text: str) -> str:
    raw_lines = text.splitlines()
    parsed_ops: list[OpLine] = []
    op_line_indexes: list[int] = []
    return_index: int | None = None
    func_index: int | None = None
    func_prefix: str | None = None

    for index, line in enumerate(raw_lines):
        op = parse_op_line(line)
        if op is not None:
            parsed_ops.append(op)
            op_line_indexes.append(index)
            continue
        if RETURN_RE.match(line):
            return_index = index
            continue
        func_match = FUNC_RE.match(line)
        if func_match:
            func_index = index
            func_prefix = func_match.group("prefix")

    new_ops, replacements = canonicalize_ops(parsed_ops)

    def resolve(value: str) -> str:
        while value in replacements:
            value = replacements[value]
        return value

    new_lines = list(raw_lines)
    for index in op_line_indexes:
        new_lines[index] = None
    for op, index in zip(new_ops, op_line_indexes, strict=False):
        new_lines[index] = op.to_line()

    if return_index is not None:
        match = RETURN_RE.match(raw_lines[return_index])
        assert match is not None
        values = [resolve(item.strip()) for item in match.group("values").split(",")]
        types: list[str] = []
        for value in values:
            for op in new_ops:
                if op.result == value:
                    types.append(op.output_type)
                    break
        type_sig = ", ".join(types) if len(types) > 1 else (types[0] if types else match.group("types"))
        new_lines[return_index] = f'{match.group("indent")}return {", ".join(values)} : {type_sig} loc({match.group("loc")})'
        if func_index is not None and func_prefix is not None and types:
            func_sig = "(" + ", ".join(types) + ")" if len(types) > 1 else types[0]
            new_lines[func_index] = f"{func_prefix}{func_sig} {{"

    compacted = [line for line in new_lines if line is not None]
    return "\n".join(compacted) + "\n"


def main() -> int:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    canonical = canonicalize_mlir(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical, encoding="utf-8")
    print(f"Wrote canonical Top MLIR to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
