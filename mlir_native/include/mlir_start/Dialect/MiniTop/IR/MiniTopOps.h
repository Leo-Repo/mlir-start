#ifndef MLIR_START_DIALECT_MINITOP_IR_MINITOPOPS_H
#define MLIR_START_DIALECT_MINITOP_IR_MINITOPOPS_H

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir_start/Dialect/MiniTop/IR/MiniTopDialect.h"

#define GET_OP_CLASSES
#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h.inc"

#endif
