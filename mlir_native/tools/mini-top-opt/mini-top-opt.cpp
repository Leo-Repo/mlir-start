#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"
#include "mlir_start/Dialect/MiniTop/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect,
                  mlir::mlir_start::mini_top::MiniTopDialect>();

  mlir::registerAllPasses();
  mlir::mlir_start::mini_top::registerMiniTopPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mini-top optimizer\n", registry));
}
