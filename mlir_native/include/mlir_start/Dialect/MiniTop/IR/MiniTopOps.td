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
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$bias,
    OptionalAttr<ArrayAttr>:$strides,
    OptionalAttr<ArrayAttr>:$pads,
    OptionalAttr<ArrayAttr>:$dilations,
    OptionalAttr<I64Attr>:$group
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input `,` $filter `,` $bias attr-dict `:` type($input) `,` type($filter) `,` type($bias) `->` type($output)";
}

def MiniTop_ConvSiLUOp : MiniTop_Op<"conv_silu"> {
  let summary = "Fused convolution plus SiLU activation";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$bias,
    OptionalAttr<ArrayAttr>:$strides,
    OptionalAttr<ArrayAttr>:$pads,
    OptionalAttr<ArrayAttr>:$dilations,
    OptionalAttr<I64Attr>:$group
  );
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

def MiniTop_QuantizeOp : MiniTop_Op<"quantize"> {
  let summary = "Symmetric int8 quantize marker";
  let arguments = (ins AnyTensor:$input, F64Attr:$scale);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def MiniTop_DequantizeOp : MiniTop_Op<"dequantize"> {
  let summary = "Symmetric int8 dequantize marker";
  let arguments = (ins AnyTensor:$input, F64Attr:$scale);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def MiniTop_QConvOp : MiniTop_Op<"qconv"> {
  let summary = "Quantized convolution placeholder op";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$bias,
    AnyTensor:$weight_scale,
    F64Attr:$input_scale,
    F64Attr:$output_scale,
    OptionalAttr<ArrayAttr>:$strides,
    OptionalAttr<ArrayAttr>:$pads,
    OptionalAttr<ArrayAttr>:$dilations,
    OptionalAttr<I64Attr>:$group
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input `,` $filter `,` $bias `,` $weight_scale attr-dict `:` type($input) `,` type($filter) `,` type($bias) `,` type($weight_scale) `->` type($output)";
}

def MiniTop_QConvSiLUOp : MiniTop_Op<"qconv_silu"> {
  let summary = "Quantized fused convolution plus SiLU placeholder op";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$bias,
    AnyTensor:$weight_scale,
    F64Attr:$input_scale,
    F64Attr:$output_scale,
    OptionalAttr<ArrayAttr>:$strides,
    OptionalAttr<ArrayAttr>:$pads,
    OptionalAttr<ArrayAttr>:$dilations,
    OptionalAttr<I64Attr>:$group
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input `,` $filter `,` $bias `,` $weight_scale attr-dict `:` type($input) `,` type($filter) `,` type($bias) `,` type($weight_scale) `->` type($output)";
}

#endif
