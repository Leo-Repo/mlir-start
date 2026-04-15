#ifndef MLIR_START_DIALECT_MINITOP_TRANSFORMS_PASSES_H
#define MLIR_START_DIALECT_MINITOP_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::mlir_start::mini_top {

std::unique_ptr<Pass> createFuseSiLUPass();
std::unique_ptr<Pass> createCanonicalizeLayoutPass();
std::unique_ptr<Pass> createLowerActivationsPass();

#define GEN_PASS_DECL
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

} // namespace mlir::mlir_start::mini_top

#endif
