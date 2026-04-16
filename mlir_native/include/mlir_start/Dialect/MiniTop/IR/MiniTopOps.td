#ifndef MLIR_START_DIALECT_MINITOP_IR_MINITOPOPS
#define MLIR_START_DIALECT_MINITOP_IR_MINITOPOPS

include "mlir/IR/OpBase.td"

def MiniTop_Dialect : Dialect {
  let name = "mini_top";
  let cppNamespace = "::mlir::mlir_start::mini_top";
  let description = [{
    Minimal YOLOv5-oriented learning dialect for the mlir_start project.
  }];
}

class MiniTop_Op<string mnemonic, list<Trait> traits = []> :
    Op<MiniTop_Dialect, mnemonic, traits>;

def MiniTop_WeightOp : MiniTop_Op<"weight"> {
  let summary = "External weight reference op";
  let arguments = (ins StrAttr:$weight_key);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$weight_key attr-dict `:` type($output)";
}

def MiniTop_ConvOp : MiniTop_Op<"conv"> {
  let summary = "Minimal convolution placeholder op";
  let arguments = (ins AnyTensor:$input, AnyTensor:$filter, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input `,` $filter `,` $bias attr-dict `:` type($input) `,` type($filter) `,` type($bias) `->` type($output)";
}

def MiniTop_SigmoidOp : MiniTop_Op<"sigmoid"> {
  let summary = "Minimal sigmoid op";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def MiniTop_MulOp : MiniTop_Op<"mul"> {
  let summary = "Minimal elementwise mul op";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

def MiniTop_AddOp : MiniTop_Op<"add"> {
  let summary = "Minimal elementwise add op";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

def MiniTop_SiLUOp : MiniTop_Op<"silu"> {
  let summary = "Minimal SiLU op";
  let description = [{
    This op exists so we can model the first native rewrite we care about:
    conv + sigmoid + mul -> silu
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def MiniTop_ReshapeOp : MiniTop_Op<"reshape"> {
  let summary = "Minimal reshape op";
  let arguments = (ins AnyTensor:$input, ArrayAttr:$shape);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def MiniTop_PermuteOp : MiniTop_Op<"permute"> {
  let summary = "Minimal permute op";
  let arguments = (ins AnyTensor:$input, ArrayAttr:$order);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def MiniTop_ConcatOp : MiniTop_Op<"concat"> {
  let summary = "Minimal concat op";
  let arguments = (ins Variadic<AnyTensor>:$inputs, I64Attr:$axis);
  let results = (outs AnyTensor:$output);
}

def MiniTop_MaxPoolOp : MiniTop_Op<"maxpool"> {
  let summary = "Minimal maxpool op";
  let arguments = (ins AnyTensor:$input, ArrayAttr:$kernel_shape, ArrayAttr:$strides, ArrayAttr:$pads);
  let results = (outs AnyTensor:$output);
}

def MiniTop_InterpOp : MiniTop_Op<"interp"> {
  let summary = "Minimal interpolation op";
  let arguments = (ins AnyTensor:$input, I64Attr:$target_h, I64Attr:$target_w, StrAttr:$mode);
  let results = (outs AnyTensor:$output);
}

#endif
