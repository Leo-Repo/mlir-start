#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::mlir_start::mini_top {

#define GEN_PASS_DEF_MINITOPFUSECONVSILU
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

namespace {

class FuseConvSiLUPattern : public OpRewritePattern<SiLUOp> {
public:
  using OpRewritePattern<SiLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SiLUOp op,
                                PatternRewriter &rewriter) const final {
    auto conv = op.getInput().getDefiningOp<ConvOp>();
    if (!conv)
      return failure();

    SmallVector<NamedAttribute> attrs;
    for (NamedAttribute attr : conv->getAttrs())
      attrs.push_back(attr);

    OperationState state(op.getLoc(), ConvSiLUOp::getOperationName());
    state.addOperands({conv.getInput(), conv.getFilter(), conv.getBias()});
    state.addTypes(op.getResult().getType());
    state.addAttributes(attrs);
    Operation *fused = rewriter.create(state);
    rewriter.replaceOp(op, fused->getResults());
    if (conv->use_empty())
      rewriter.eraseOp(conv);
    return success();
  }
};

class MiniTopFuseConvSiLUPass
    : public impl::MiniTopFuseConvSiLUBase<MiniTopFuseConvSiLUPass> {
public:
  using impl::MiniTopFuseConvSiLUBase<
      MiniTopFuseConvSiLUPass>::MiniTopFuseConvSiLUBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConvSiLUPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseConvSiLUPass() {
  return std::make_unique<MiniTopFuseConvSiLUPass>();
}

} // namespace mlir::mlir_start::mini_top
