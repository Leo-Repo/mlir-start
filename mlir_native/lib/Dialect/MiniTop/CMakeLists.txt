add_mlir_dialect_library(MLIRStartMiniTop
  MiniTopDialect.cpp
  MiniTopOps.cpp
  CanonicalizeLayout.cpp
  FuseConvSiLU.cpp
  FuseSiLU.cpp
  InsertInt8Quant.cpp
  LowerActivations.cpp
  LowerConvSiLUToGPU.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir_start/Dialect/MiniTop

  DEPENDS
    MLIRStartMiniTopOpsIncGen
    MLIRStartMiniTopPassIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRMathDialect
    MLIRPass
    MLIRRewrite
    MLIRTransforms
)
