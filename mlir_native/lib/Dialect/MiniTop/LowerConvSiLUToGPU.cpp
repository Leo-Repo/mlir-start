#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::mlir_start::mini_top {

#define GEN_PASS_DEF_MINITOPLOWERCONVSILUTOGPU
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

namespace {

class MiniTopLowerConvSiLUToGPUPass
    : public impl::MiniTopLowerConvSiLUToGPUBase<
          MiniTopLowerConvSiLUToGPUPass> {
public:
  using impl::MiniTopLowerConvSiLUToGPUBase<
      MiniTopLowerConvSiLUToGPUPass>::MiniTopLowerConvSiLUToGPUBase;

  void runOnOperation() final {
    StringAttr marker =
        StringAttr::get(&getContext(), "conv_silu_cuda_candidate");
    getOperation().walk([&](Operation *op) {
      if (isa<ConvSiLUOp, QConvSiLUOp>(op))
        op->setAttr("mini_top.gpu_lowering", marker);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createLowerConvSiLUToGPUPass() {
  return std::make_unique<MiniTopLowerConvSiLUToGPUPass>();
}

} // namespace mlir::mlir_start::mini_top
