#!/usr/bin/env python3
"""Prepare .npy tensors for mini-top-run.

This helper intentionally stays small: it converts existing tensor containers
into the one-file-per-input format used by the C++ runner. Image decoding and
letterbox preprocessing can be added later without making the runner depend on
OpenCV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NPZ tensors to .npy inputs.")
    parser.add_argument("--npz", type=Path, required=True, help="Source .npz file.")
    parser.add_argument("--key", help="Tensor key to extract. Defaults to the first key.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="input.npy")
    parser.add_argument(
        "--list-file",
        default="inputs.txt",
        help="Input list file written next to the .npy tensor.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)
    key = args.key or data.files[0]
    if key not in data:
        raise SystemExit(f"Key `{key}` not found in {args.npz}. Available: {data.files}")
    output = args.output_dir / args.output_name
    np.save(output, np.ascontiguousarray(data[key].astype(np.float32, copy=False)))
    (args.output_dir / args.list_file).write_text(str(output.resolve()) + "\n", encoding="utf-8")
    print(f"Wrote input tensor: {output.resolve()}")
    print(f"Wrote input list: {(args.output_dir / args.list_file).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
