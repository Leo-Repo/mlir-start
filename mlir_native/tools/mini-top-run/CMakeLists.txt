set(LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRStartMiniTop
  MLIRSupport
)

add_llvm_executable(mini-top-run
  mini-top-run.cpp
)

llvm_update_compile_flags(mini-top-run)
target_link_libraries(mini-top-run PRIVATE ${LIBS})

mlir_check_all_link_libraries(mini-top-run)
