#ifndef MLIR_START_DIALECT_MINITOP_TRANSFORMS_PASSES_H
#define MLIR_START_DIALECT_MINITOP_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include <memory>

namespace mlir::mlir_start::mini_top {

std::unique_ptr<Pass> createFuseSiLUPass();
std::unique_ptr<Pass> createFuseConvSiLUPass();
std::unique_ptr<Pass> createCanonicalizeLayoutPass();
std::unique_ptr<Pass> createLowerActivationsPass();
std::unique_ptr<Pass> createInsertInt8QuantPass();
std::unique_ptr<Pass> createLowerConvSiLUToGPUPass();

#define GEN_PASS_DECL
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

} // namespace mlir::mlir_start::mini_top

#endif
