#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::mlir_start::mini_top {

#define GEN_PASS_DEF_MINITOPFUSESILU
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

namespace {

class FuseSigmoidMulPattern : public OpRewritePattern<MulOp> {
public:
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const final {
    auto rewrite = [&](SigmoidOp sigmoid, Value passthrough) -> LogicalResult {
      if (!sigmoid)
        return failure();
      if (sigmoid.getInput() != passthrough)
        return failure();
      Operation *sigmoidOp = sigmoid.getOperation();
      rewriter.replaceOpWithNewOp<SiLUOp>(op, op.getResult().getType(), passthrough);
      if (sigmoidOp->use_empty())
        rewriter.eraseOp(sigmoidOp);
      return success();
    };

    if (succeeded(rewrite(op.getLhs().getDefiningOp<SigmoidOp>(), op.getRhs())))
      return success();
    if (succeeded(rewrite(op.getRhs().getDefiningOp<SigmoidOp>(), op.getLhs())))
      return success();
    return failure();
  }
};

class MiniTopFuseSiLUPass
    : public impl::MiniTopFuseSiLUBase<MiniTopFuseSiLUPass> {
public:
  using impl::MiniTopFuseSiLUBase<
      MiniTopFuseSiLUPass>::MiniTopFuseSiLUBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSigmoidMulPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseSiLUPass() {
  return std::make_unique<MiniTopFuseSiLUPass>();
}

} // namespace mlir::mlir_start::mini_top
