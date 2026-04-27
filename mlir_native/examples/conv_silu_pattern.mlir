module {
  func.func @main(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x1x2x2xf32>, %arg2: tensor<1xf32>) -> tensor<1x1x2x2xf32> {
    %0 = mini_top.conv %arg0, %arg1, %arg2 {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1], group = 1 : i64} : tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>, tensor<1xf32> -> tensor<1x1x2x2xf32>
    %1 = mini_top.sigmoid %0 : tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    %2 = mini_top.mul %0, %1 : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    return %2 : tensor<1x1x2x2xf32>
  }
}
