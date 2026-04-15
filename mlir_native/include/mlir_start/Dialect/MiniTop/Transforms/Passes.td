#ifndef MLIR_START_DIALECT_MINITOP_TRANSFORMS_PASSES
#define MLIR_START_DIALECT_MINITOP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MiniTopFuseSiLU : Pass<"mini-top-fuse-silu", "mlir::func::FuncOp"> {
  let summary = "Fuse mini_top.sigmoid + mini_top.mul into mini_top.silu";
  let description = [{
    A first native rewrite pass for the project. It mirrors the SiLU pattern
    already studied in YOLOv5 and replaces:

      %s = mini_top.sigmoid %x
      %y = mini_top.mul %x, %s

    with:

      %y = mini_top.silu %x
  }];
  let constructor = "mlir::mlir_start::mini_top::createFuseSiLUPass()";
}

def MiniTopCanonicalizeLayout : Pass<"mini-top-canonicalize-layout", "mlir::func::FuncOp"> {
  let summary = "Canonicalize layout-only MiniTop ops";
  let description = [{
    Migrates two YOLOv5-relevant Python canonicalization rules into the native track:

    - remove no-op `mini_top.reshape`
    - remove identity `mini_top.permute`

    This mirrors the local Python `top_canonicalize.py` behavior, but now lives
    as a real MLIR rewrite pass.
  }];
  let constructor = "mlir::mlir_start::mini_top::createCanonicalizeLayoutPass()";
}

def MiniTopLowerActivations : Pass<"mini-top-lower-activations", "mlir::func::FuncOp"> {
  let summary = "Lower mini_top activation ops to lower built-in dialects";
  let description = [{
    First lowering step for the native subproject.

    Current scope:
    - mini_top.sigmoid -> arith/math ops
    - mini_top.mul     -> arith.mulf
    - mini_top.silu    -> arith/math ops

    This keeps `mini_top.conv` in place for now, but moves activation semantics
    into lower standard dialects so the project can start observing a true
    lowering boundary.
  }];
  let constructor = "mlir::mlir_start::mini_top::createLowerActivationsPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect"
  ];
}

#endif
