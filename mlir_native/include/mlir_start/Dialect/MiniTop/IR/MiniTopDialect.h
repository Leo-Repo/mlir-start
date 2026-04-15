#ifndef MLIR_START_DIALECT_MINITOP_IR_MINITOPDIALECT_H
#define MLIR_START_DIALECT_MINITOP_IR_MINITOPDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir::mlir_start::mini_top {

class MiniTopDialect : public Dialect {
public:
  explicit MiniTopDialect(MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "mini_top"; }
};

} // namespace mlir::mlir_start::mini_top

#endif
