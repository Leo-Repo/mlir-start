#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdlib>
#include <optional>

namespace mlir::mlir_start::mini_top {

#define GEN_PASS_DEF_MINITOPINSERTINT8QUANT
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h.inc"

namespace {

static Type withElementType(Type type, Type elementType) {
  if (auto ranked = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(ranked.getShape(), elementType);
  if (auto unranked = dyn_cast<UnrankedTensorType>(type))
    return UnrankedTensorType::get(elementType);
  return type;
}

static void copyConvAttrs(Operation *source, OperationState &state) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name == "input_scale" || name == "output_scale")
      continue;
    state.addAttribute(attr.getName(), attr.getValue());
  }
}

static std::optional<double> readGlobalScaleFromCalibration(StringRef path) {
  if (path.empty())
    return std::nullopt;
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::nullopt;
  StringRef text = (*buffer)->getBuffer();
  double maxAbs = 0.0;
  size_t pos = 0;
  while ((pos = text.find("\"absmax\"", pos)) != StringRef::npos) {
    size_t colon = text.find(':', pos);
    if (colon == StringRef::npos)
      break;
    StringRef tail = text.drop_front(colon + 1).ltrim();
    char *end = nullptr;
    std::string number = tail.take_until([](char c) {
                              return c == ',' || c == '}' || c == '\n';
                            }).str();
    double value = std::strtod(number.c_str(), &end);
    if (end != number.c_str())
      maxAbs = std::max(maxAbs, std::abs(value));
    pos = colon + 1;
  }
  if (maxAbs <= 1e-12)
    return std::nullopt;
  return maxAbs / 127.0;
}

class QuantizeConvSiLUPattern : public OpRewritePattern<ConvSiLUOp> {
public:
  QuantizeConvSiLUPattern(MLIRContext *context, double inputScale,
                          double outputScale)
      : OpRewritePattern<ConvSiLUOp>(context), inputScale(inputScale),
        outputScale(outputScale) {}

  LogicalResult matchAndRewrite(ConvSiLUOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type i8 = rewriter.getIntegerType(8);
    Type qInputType = withElementType(op.getInput().getType(), i8);
    Type qOutputType = withElementType(op.getResult().getType(), i8);
    auto inputScaleAttr = rewriter.getF64FloatAttr(inputScale);
    auto outputScaleAttr = rewriter.getF64FloatAttr(outputScale);

    OperationState qState(loc, QuantizeOp::getOperationName());
    qState.addOperands(op.getInput());
    qState.addTypes(qInputType);
    qState.addAttribute("scale", inputScaleAttr);
    Operation *qInput = rewriter.create(qState);

    OperationState qconvState(loc, QConvSiLUOp::getOperationName());
    qconvState.addOperands(
        {qInput->getResult(0), op.getFilter(), op.getBias(), op.getFilter()});
    qconvState.addTypes(qOutputType);
    copyConvAttrs(op.getOperation(), qconvState);
    qconvState.addAttribute("input_scale", inputScaleAttr);
    qconvState.addAttribute("output_scale", outputScaleAttr);
    Operation *qconv = rewriter.create(qconvState);

    OperationState dqState(loc, DequantizeOp::getOperationName());
    dqState.addOperands(qconv->getResult(0));
    dqState.addTypes(op.getResult().getType());
    dqState.addAttribute("scale", outputScaleAttr);
    Operation *dq = rewriter.create(dqState);

    rewriter.replaceOp(op, dq->getResults());
    return success();
  }

private:
  double inputScale;
  double outputScale;
};

class MiniTopInsertInt8QuantPass
    : public impl::MiniTopInsertInt8QuantBase<MiniTopInsertInt8QuantPass> {
public:
  using impl::MiniTopInsertInt8QuantBase<
      MiniTopInsertInt8QuantPass>::MiniTopInsertInt8QuantBase;

  void runOnOperation() final {
    double inScale = inputScale;
    double outScale = outputScale;
    if (std::optional<double> tableScale =
            readGlobalScaleFromCalibration(calibrationTable)) {
      inScale = *tableScale;
      outScale = *tableScale;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<QuantizeConvSiLUPattern>(&getContext(), inScale, outScale);
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createInsertInt8QuantPass() {
  return std::make_unique<MiniTopInsertInt8QuantPass>();
}

} // namespace mlir::mlir_start::mini_top
