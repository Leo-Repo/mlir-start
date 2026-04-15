#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::mlir_start::mini_top {

#define GEN_PASS_DEF_MINITOPCANONICALIZELAYOUT
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

namespace {

class ElideNoOpReshapePattern : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getInput().getType() != op.getResult().getType())
      return failure();
    rewriter.replaceOp(op.getOperation(), ValueRange{op.getInput()});
    return success();
  }
};

class ElideIdentityPermutePattern : public OpRewritePattern<PermuteOp> {
public:
  using OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getInput().getType() != op.getResult().getType())
      return failure();

    auto order = op.getOrder();
    for (auto [index, dimAttr] : llvm::enumerate(order)) {
      auto dim = llvm::dyn_cast<IntegerAttr>(dimAttr);
      if (!dim || dim.getInt() != static_cast<int64_t>(index))
        return failure();
    }

    rewriter.replaceOp(op.getOperation(), ValueRange{op.getInput()});
    return success();
  }
};

class MiniTopCanonicalizeLayoutPass
    : public impl::MiniTopCanonicalizeLayoutBase<MiniTopCanonicalizeLayoutPass> {
public:
  using impl::MiniTopCanonicalizeLayoutBase<
      MiniTopCanonicalizeLayoutPass>::MiniTopCanonicalizeLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ElideNoOpReshapePattern, ElideIdentityPermutePattern>(
        &getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCanonicalizeLayoutPass() {
  return std::make_unique<MiniTopCanonicalizeLayoutPass>();
}

} // namespace mlir::mlir_start::mini_top
