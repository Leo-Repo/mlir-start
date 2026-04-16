# MLIR Native Skeleton

This subtree is the first LLVM/MLIR-native skeleton for the project.

It intentionally does not replace the existing Python prototype yet. Instead it
gives us a minimal place to move core concepts into the MLIR ecosystem:

- TableGen-defined ops
- a real dialect in C++
- a native rewrite pass
- an `mlir-opt`-style driver

## Current Scope

The first slice is deliberately small:

- Dialect: `mini_top`
- Ops:
  - `mini_top.conv`
  - `mini_top.sigmoid`
  - `mini_top.mul`
  - `mini_top.silu`
  - `mini_top.reshape`
  - `mini_top.permute`
- Pass:
  - `--mini-top-fuse-silu`
  - `--mini-top-canonicalize-layout`
  - `--mini-top-lower-activations`

This is enough to mirror the first important pattern we already studied in
YOLOv5:

`Conv + Sigmoid + Mul -> SiLU`

The project is now intentionally split into two layers:

- root Python scripts:
  - importer / exploration / reference execution
  - frozen as reference-oracle code except for bug fixes
- `mlir_native/`:
  - the forward-looking MLIR/LLVM-native implementation track
  - where new pass/lowering work should land

## Build

```bash
cmake -S /home/jay/projs/mlir_start/mlir_native \
  -B /home/jay/projs/mlir_start/mlir_native/build \
  -DMLIR_DIR=/home/jay/projs/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/jay/projs/llvm-project/build/lib/cmake/llvm

cmake --build /home/jay/projs/mlir_start/mlir_native/build -j
```

## Try It

```bash
/home/jay/projs/mlir_start/mlir_native/build/bin/mini-top-opt \
  /home/jay/projs/mlir_start/mlir_native/examples/silu_pattern.mlir \
  --mini-top-fuse-silu
```

If the pass runs as expected, the `mini_top.mul` should be rewritten to
`mini_top.silu`.

To observe YOLOv5-style layout canonicalization:

```bash
/home/jay/projs/mlir_start/mlir_native/build/bin/mini-top-opt \
  /home/jay/projs/mlir_start/mlir_native/examples/layout_patterns.mlir \
  --mini-top-canonicalize-layout
```

This removes:

- no-op `mini_top.reshape`
- identity `mini_top.permute`

To observe the first lowering step:

```bash
/home/jay/projs/mlir_start/mlir_native/build/bin/mini-top-opt \
  /home/jay/projs/mlir_start/mlir_native/examples/silu_pattern.mlir \
  --mini-top-fuse-silu \
  --mini-top-lower-activations
```

This keeps `mini_top.conv` untouched for now, but lowers `mini_top.silu` into
`arith` + `math` ops.

## What Moved From Python

The native line now contains the first small batch of YOLOv5-relevant rules
that originally lived in the Python prototypes:

- `Conv + Sigmoid + Mul -> SiLU`
- remove no-op `reshape`
- remove identity `permute`

This is the current pattern migration direction for the project:

1. validate the idea in Python
2. move the stable rule into `mlir_native`
3. keep Python as an oracle, not the main compiler path

## IR-Object Importer

The project now also contains a first native-track ONNX importer prototype:

- [`python/onnx_to_mini_top.py`](/home/jay/projs/mlir_start/mlir_native/python/onnx_to_mini_top.py)

This importer is important because it changes the frontend style:

- old archived frontend:
  - parsed ONNX and concatenated `.mlir` text
- new native-track frontend:
  - parses ONNX
  - uses `mlir.ir` objects plus `Operation.create(...)`
  - builds `mini_top` IR in memory

Current scope is intentionally limited to the YOLOv5 subset we already study:

- `Conv`
- `Sigmoid`
- `Mul`
- `Add`
- `Concat`
- `MaxPool`
- `Resize`
- `Reshape`
- `Transpose`
- shape-subgraph constant folding for `Shape/Gather/Unsqueeze/Cast`

It also materializes model weights into a separate `.npz` file and emits
`mini_top.weight` ops so the importer stays close to the "external weights +
graph IR" style used by real compiler frontends.

Note:

- this script requires both `onnx` and upstream MLIR Python bindings
- the current workspace build does not yet provide those bindings by default
- so the script is checked in, syntax-checked, and ready to use once the MLIR
  Python environment is available
