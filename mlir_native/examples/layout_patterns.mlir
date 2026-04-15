func.func @main(%arg0: tensor<1x3x80x80xf32>) -> tensor<1x3x80x80xf32> {
  %0 = mini_top.reshape %arg0 {shape = [1, 3, 80, 80]} : tensor<1x3x80x80xf32> -> tensor<1x3x80x80xf32>
  %1 = mini_top.permute %0 {order = [0, 1, 2, 3]} : tensor<1x3x80x80xf32> -> tensor<1x3x80x80xf32>
  return %1 : tensor<1x3x80x80xf32>
}
