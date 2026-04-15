add_mlir_dialect_library(MLIRStartMiniTop
  MiniTopDialect.cpp
  MiniTopOps.cpp
  CanonicalizeLayout.cpp
  FuseSiLU.cpp
  LowerActivations.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir_start/Dialect/MiniTop

  DEPENDS
    MLIRStartMiniTopOpsIncGen
    MLIRStartMiniTopPassIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRFuncDialect
    MLIRMathDialect
    MLIRPass
    MLIRRewrite
    MLIRTransforms
)
