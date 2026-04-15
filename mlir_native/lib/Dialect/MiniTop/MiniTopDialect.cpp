#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"

using namespace mlir;
using namespace mlir::mlir_start::mini_top;

MiniTopDialect::MiniTopDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MiniTopDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.cpp.inc"
      >();
}
