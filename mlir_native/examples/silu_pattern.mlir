func.func @main(
  %input: tensor<1x3x640x640xf32>,
  %filter: tensor<32x3x6x6xf32>,
  %bias: tensor<32xf32>
) -> tensor<1x32x320x320xf32> {
  %0 = "mini_top.conv"(%input, %filter, %bias) : (tensor<1x3x640x640xf32>, tensor<32x3x6x6xf32>, tensor<32xf32>) -> tensor<1x32x320x320xf32>
  %1 = "mini_top.sigmoid"(%0) : (tensor<1x32x320x320xf32>) -> tensor<1x32x320x320xf32>
  %2 = "mini_top.mul"(%0, %1) : (tensor<1x32x320x320xf32>, tensor<1x32x320x320xf32>) -> tensor<1x32x320x320xf32>
  return %2 : tensor<1x32x320x320xf32>
}
