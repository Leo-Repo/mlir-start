set(LLVM_TARGET_DEFINITIONS MiniTopOps.td)
mlir_tablegen(MiniTopOps.h.inc -gen-op-decls)
mlir_tablegen(MiniTopOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRStartMiniTopOpsIncGen)
