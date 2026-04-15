#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::mlir_start::mini_top {

#define GEN_PASS_DEF_MINITOPLOWERACTIVATIONS
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

namespace {

static Value createFloatLikeConstant(PatternRewriter &rewriter, Location loc,
                                     Type type, double value) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    auto attr = rewriter.getFloatAttr(floatType, value);
    return rewriter.create<arith::ConstantOp>(loc, type, attr);
  }

  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementType = dyn_cast<FloatType>(tensorType.getElementType());
    if (!elementType || !tensorType.hasStaticShape())
      return {};
    auto attr =
        DenseElementsAttr::get(tensorType, rewriter.getFloatAttr(elementType, value));
    return rewriter.create<arith::ConstantOp>(loc, tensorType, attr);
  }

  return {};
}

static Value emitSigmoidExpr(PatternRewriter &rewriter, Location loc, Value input) {
  Type type = input.getType();
  Value negOne = createFloatLikeConstant(rewriter, loc, type, -1.0);
  Value one = createFloatLikeConstant(rewriter, loc, type, 1.0);
  if (!negOne || !one)
    return {};

  Value neg = rewriter.create<arith::MulFOp>(loc, input, negOne);
  Value exp = rewriter.create<math::ExpOp>(loc, type, neg);
  Value denom = rewriter.create<arith::AddFOp>(loc, one, exp);
  return rewriter.create<arith::DivFOp>(loc, one, denom);
}

class LowerSigmoidPattern : public OpRewritePattern<SigmoidOp> {
public:
  using OpRewritePattern<SigmoidOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SigmoidOp op,
                                PatternRewriter &rewriter) const final {
    Value lowered = emitSigmoidExpr(rewriter, op.getLoc(), op.getInput());
    if (!lowered)
      return failure();
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

class LowerMulPattern : public OpRewritePattern<MulOp> {
public:
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const final {
    Type type = op.getResult().getType();
    if (!isa<FloatType>(type) &&
        !(isa<RankedTensorType>(type) &&
          isa<FloatType>(cast<RankedTensorType>(type).getElementType())))
      return failure();
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

class LowerSiLUPattern : public OpRewritePattern<SiLUOp> {
public:
  using OpRewritePattern<SiLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SiLUOp op,
                                PatternRewriter &rewriter) const final {
    Value sigmoid = emitSigmoidExpr(rewriter, op.getLoc(), op.getInput());
    if (!sigmoid)
      return failure();
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getInput(), sigmoid);
    return success();
  }
};

class MiniTopLowerActivationsPass
    : public impl::MiniTopLowerActivationsBase<MiniTopLowerActivationsPass> {
public:
  using impl::MiniTopLowerActivationsBase<
      MiniTopLowerActivationsPass>::MiniTopLowerActivationsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSigmoidPattern, LowerMulPattern, LowerSiLUPattern>(
        &getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerActivationsPass() {
  return std::make_unique<MiniTopLowerActivationsPass>();
}

} // namespace mlir::mlir_start::mini_top
