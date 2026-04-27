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

    This mirrors the local Python `legacy_python_frontend/top_canonicalize.py`
    behavior, but now lives
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

def MiniTopFuseConvSiLU : Pass<"mini-top-fuse-conv-silu", "mlir::func::FuncOp"> {
  let summary = "Fuse mini_top.conv feeding mini_top.silu into mini_top.conv_silu";
  let description = [{
    Models the first hardware-oriented fused layer boundary for this project.
    The pass expects the existing SiLU pass to have already rewritten
    sigmoid/mul patterns, but it also naturally handles importer-emitted
    conv -> silu patterns.
  }];
  let constructor = "mlir::mlir_start::mini_top::createFuseConvSiLUPass()";
}

def MiniTopInsertInt8Quant : Pass<"mini-top-insert-int8-quant", "mlir::func::FuncOp"> {
  let summary = "Insert first-cut symmetric int8 quantization markers";
  let description = [{
    Rewrites mini_top.conv_silu into quantize -> qconv_silu -> dequantize.
    Calibration table parsing is intentionally kept out of this first native
    skeleton; the pass exposes scale options so generated IR has the final
    quantized shape and can be executed by mini-top-run.
  }];
  let constructor = "mlir::mlir_start::mini_top::createInsertInt8QuantPass()";
  let options = [
    Option<"inputScale", "input-scale", "double", /*default=*/"1.0",
           "Activation scale to attach to inserted quantize/qconv_silu ops">,
    Option<"outputScale", "output-scale", "double", /*default=*/"1.0",
           "Output scale to attach to qconv_silu/dequantize ops">,
    Option<"calibrationTable", "calibration-table", "std::string", /*default=*/"\"\"",
           "Optional mini-top-run calibration JSON; first version derives a conservative global scale from absmax values">
  ];
}

def MiniTopLowerConvSiLUToGPU : Pass<"mini-top-lower-conv-silu-to-gpu", "mlir::func::FuncOp"> {
  let summary = "Mark conv_silu ops with a GPU lowering demonstration op";
  let description = [{
    First GPU lowering demonstration. It creates a gpu.launch_func-compatible
    marker region around conv_silu/qconv_silu candidates without attempting a
    full executable binary pipeline.
  }];
  let constructor = "mlir::mlir_start::mini_top::createLowerConvSiLUToGPUPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect"
  ];
}

#endif
