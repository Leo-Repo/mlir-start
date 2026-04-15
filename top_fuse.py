#!/usr/bin/env python3
"""Local Top MLIR fusion pass for the YOLOv5 learning workflow.

This pass intentionally operates after `top_canonicalize.py` and only covers a
small, conservative subset of graph rewrites that are easy to reason about for
the current project. The first implemented rule is:

  top.Conv -> top.Sigmoid -> top.Mul

where `top.Mul` consumes the Conv result and the corresponding
Sigmoid(Conv) result. This pattern is rewritten into the more official form:

  top.Conv -> top.SiLU
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from top_canonicalize import OpLine, parse_op_line


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "experiments" / "01_onnx_to_mlir" / "yolov5s_canonical.mlir"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "01_onnx_to_mlir" / "yolov5s_fused.mlir"


@dataclass
class FusionRecord:
    conv_result: str
    sigmoid_result: str
    mul_result: str
    conv_loc: str
    sigmoid_loc: str
    mul_loc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse YOLOv5-specific Top MLIR patterns.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input canonical Top MLIR.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output fused Top MLIR.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only analyze and summarize fusible patterns without writing output MLIR.",
    )
    parser.add_argument(
        "--dump-patterns",
        action="store_true",
        help="Print every matched fusible pattern record.",
    )
    return parser.parse_args()


def build_use_map(ops: list[OpLine]) -> dict[str, list[int]]:
    uses: dict[str, list[int]] = {}
    for index, op in enumerate(ops):
        for operand in op.operands:
            if not operand.startswith("%"):
                continue
            uses.setdefault(operand, []).append(index)
    return uses


def fuse_conv_silu(ops: list[OpLine]) -> tuple[dict[int, OpLine | None], list[FusionRecord]]:
    producer = {op.result: index for index, op in enumerate(ops)}
    uses = build_use_map(ops)
    fused_records: list[FusionRecord] = []
    rewritten_by_index: dict[int, OpLine | None] = {index: op for index, op in enumerate(ops)}

    for index, op in enumerate(ops):
        if rewritten_by_index.get(index) is None:
            continue
        if op.op != "top.Mul" or len(op.operands) != 2:
            continue

        conv_operand = None
        sigmoid_operand = None
        if op.operands[0] in producer and ops[producer[op.operands[0]]].op == "top.Sigmoid":
            sigmoid_operand = op.operands[0]
            conv_operand = op.operands[1]
        elif op.operands[1] in producer and ops[producer[op.operands[1]]].op == "top.Sigmoid":
            sigmoid_operand = op.operands[1]
            conv_operand = op.operands[0]

        if sigmoid_operand is None or conv_operand is None:
            continue

        sigmoid_index = producer.get(sigmoid_operand)
        conv_index = producer.get(conv_operand)
        if sigmoid_index is None or conv_index is None:
            continue

        sigmoid_op = ops[sigmoid_index]
        conv_op = ops[conv_index]
        if conv_op.op != "top.Conv":
            continue
        if len(sigmoid_op.operands) != 1 or sigmoid_op.operands[0] != conv_op.result:
            continue
        sigmoid_uses = uses.get(sigmoid_op.result, [])
        if sigmoid_uses != [index]:
            continue

        silu_op = OpLine(
            indent=op.indent,
            result=op.result,
            op="top.SiLU",
            operands=[conv_op.result],
            attrs=[],
            input_types=[conv_op.output_type],
            output_type=op.output_type,
            loc=op.loc,
        )
        rewritten_by_index[sigmoid_index] = None
        rewritten_by_index[index] = silu_op
        fused_records.append(
            FusionRecord(
                conv_result=conv_op.result,
                sigmoid_result=sigmoid_op.result,
                mul_result=op.result,
                conv_loc=conv_op.loc,
                sigmoid_loc=sigmoid_op.loc,
                mul_loc=op.loc,
            )
        )

    return rewritten_by_index, fused_records


def fuse_mlir(text: str) -> tuple[str, list[FusionRecord]]:
    raw_lines = text.splitlines()
    parsed_ops: list[OpLine] = []
    op_line_indexes: list[int] = []
    for index, line in enumerate(raw_lines):
        op = parse_op_line(line)
        if op is None:
            continue
        parsed_ops.append(op)
        op_line_indexes.append(index)

    rewritten_by_index, fused_records = fuse_conv_silu(parsed_ops)
    new_lines = list(raw_lines)
    for parsed_index, line_index in enumerate(op_line_indexes):
        rewritten = rewritten_by_index.get(parsed_index)
        new_lines[line_index] = None if rewritten is None else rewritten.to_line()
    fused_text = "\n".join(line for line in new_lines if line is not None) + "\n"
    return fused_text, fused_records


def print_summary(records: list[FusionRecord], *, dump_patterns: bool) -> None:
    print(f"Detected fusible Conv+Sigmoid+Mul patterns: {len(records)}")
    if not records:
        return
    print("Pattern kind summary:")
    print(f"  Conv+Sigmoid+Mul -> Conv+SiLU: {len(records)}")
    if dump_patterns:
        print("Matched patterns:")
        for index, record in enumerate(records, start=1):
            print(
                f"  [{index}] {record.conv_result} + {record.sigmoid_result} -> {record.mul_result} "
                f"(locs: {record.conv_loc}, {record.sigmoid_loc}, {record.mul_loc})"
            )
    else:
        preview = min(10, len(records))
        print("First matches:")
        for record in records[:preview]:
            print(
                "  "
                f"{record.conv_result} + {record.sigmoid_result} -> {record.mul_result} "
                f"(locs: {record.conv_loc}, {record.sigmoid_loc}, {record.mul_loc})"
            )
        if len(records) > preview:
            print(f"  ... and {len(records) - preview} more")


def main() -> int:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    fused_text, records = fuse_mlir(text)
    print_summary(records, dump_patterns=args.dump_patterns)
    if args.summary_only:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fused_text, encoding="utf-8")
    print(f"Wrote fused Top MLIR to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
