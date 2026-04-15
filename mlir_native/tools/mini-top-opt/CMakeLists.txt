set(LIBS
  MLIRArithDialect
  MLIRMathDialect
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRFuncDialect
  MLIRStartMiniTop
)

add_llvm_executable(mini-top-opt
  mini-top-opt.cpp
)

llvm_update_compile_flags(mini-top-opt)
target_link_libraries(mini-top-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mini-top-opt)
