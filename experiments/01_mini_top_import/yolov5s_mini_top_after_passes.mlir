module attributes {mini_top.weight_file = "/home/jay/projs/mlir_start/experiments/01_mini_top_import/yolov5s_mini_top_weights.npz"} {
  func.func @main(%arg0: tensor<1x3x640x640xf32>) -> (tensor<1x3x80x80x85xf32>, tensor<1x3x40x40x85xf32>, tensor<1x3x20x20x85xf32>) {
    %0 = mini_top.weight "model.0.conv.weight" : tensor<32x3x6x6xf32>
    %1 = mini_top.weight "model.0.conv.bias" : tensor<32xf32>
    %2 = mini_top.conv_silu %arg0, %0, %1 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [2, 2, 2, 2], strides = [2, 2]} : tensor<1x3x640x640xf32>, tensor<32x3x6x6xf32>, tensor<32xf32> -> tensor<1x32x320x320xf32>
    %3 = mini_top.weight "model.1.conv.weight" : tensor<64x32x3x3xf32>
    %4 = mini_top.weight "model.1.conv.bias" : tensor<64xf32>
    %5 = mini_top.conv_silu %2, %3, %4 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x32x320x320xf32>, tensor<64x32x3x3xf32>, tensor<64xf32> -> tensor<1x64x160x160xf32>
    %6 = mini_top.weight "model.2.cv1.conv.weight" : tensor<32x64x1x1xf32>
    %7 = mini_top.weight "model.2.cv1.conv.bias" : tensor<32xf32>
    %8 = mini_top.conv_silu %5, %6, %7 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x64x160x160xf32>, tensor<32x64x1x1xf32>, tensor<32xf32> -> tensor<1x32x160x160xf32>
    %9 = mini_top.weight "model.2.m.0.cv1.conv.weight" : tensor<32x32x1x1xf32>
    %10 = mini_top.weight "model.2.m.0.cv1.conv.bias" : tensor<32xf32>
    %11 = mini_top.conv_silu %8, %9, %10 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x32x160x160xf32>, tensor<32x32x1x1xf32>, tensor<32xf32> -> tensor<1x32x160x160xf32>
    %12 = mini_top.weight "model.2.m.0.cv2.conv.weight" : tensor<32x32x3x3xf32>
    %13 = mini_top.weight "model.2.m.0.cv2.conv.bias" : tensor<32xf32>
    %14 = mini_top.conv_silu %11, %12, %13 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x32x160x160xf32>, tensor<32x32x3x3xf32>, tensor<32xf32> -> tensor<1x32x160x160xf32>
    %15 = mini_top.add %8, %14 : tensor<1x32x160x160xf32>, tensor<1x32x160x160xf32> -> tensor<1x32x160x160xf32>
    %16 = mini_top.weight "model.2.cv2.conv.weight" : tensor<32x64x1x1xf32>
    %17 = mini_top.weight "model.2.cv2.conv.bias" : tensor<32xf32>
    %18 = mini_top.conv_silu %5, %16, %17 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x64x160x160xf32>, tensor<32x64x1x1xf32>, tensor<32xf32> -> tensor<1x32x160x160xf32>
    %19 = "mini_top.concat"(%15, %18) <{axis = 1 : i64}> : (tensor<1x32x160x160xf32>, tensor<1x32x160x160xf32>) -> tensor<1x64x160x160xf32>
    %20 = mini_top.weight "model.2.cv3.conv.weight" : tensor<64x64x1x1xf32>
    %21 = mini_top.weight "model.2.cv3.conv.bias" : tensor<64xf32>
    %22 = mini_top.conv_silu %19, %20, %21 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x64x160x160xf32>, tensor<64x64x1x1xf32>, tensor<64xf32> -> tensor<1x64x160x160xf32>
    %23 = mini_top.weight "model.3.conv.weight" : tensor<128x64x3x3xf32>
    %24 = mini_top.weight "model.3.conv.bias" : tensor<128xf32>
    %25 = mini_top.conv_silu %22, %23, %24 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x64x160x160xf32>, tensor<128x64x3x3xf32>, tensor<128xf32> -> tensor<1x128x80x80xf32>
    %26 = mini_top.weight "model.4.cv1.conv.weight" : tensor<64x128x1x1xf32>
    %27 = mini_top.weight "model.4.cv1.conv.bias" : tensor<64xf32>
    %28 = mini_top.conv_silu %25, %26, %27 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x80x80xf32>, tensor<64x128x1x1xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %29 = mini_top.weight "model.4.m.0.cv1.conv.weight" : tensor<64x64x1x1xf32>
    %30 = mini_top.weight "model.4.m.0.cv1.conv.bias" : tensor<64xf32>
    %31 = mini_top.conv_silu %28, %29, %30 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x64x80x80xf32>, tensor<64x64x1x1xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %32 = mini_top.weight "model.4.m.0.cv2.conv.weight" : tensor<64x64x3x3xf32>
    %33 = mini_top.weight "model.4.m.0.cv2.conv.bias" : tensor<64xf32>
    %34 = mini_top.conv_silu %31, %32, %33 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x64x80x80xf32>, tensor<64x64x3x3xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %35 = mini_top.add %28, %34 : tensor<1x64x80x80xf32>, tensor<1x64x80x80xf32> -> tensor<1x64x80x80xf32>
    %36 = mini_top.weight "model.4.m.1.cv1.conv.weight" : tensor<64x64x1x1xf32>
    %37 = mini_top.weight "model.4.m.1.cv1.conv.bias" : tensor<64xf32>
    %38 = mini_top.conv_silu %35, %36, %37 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x64x80x80xf32>, tensor<64x64x1x1xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %39 = mini_top.weight "model.4.m.1.cv2.conv.weight" : tensor<64x64x3x3xf32>
    %40 = mini_top.weight "model.4.m.1.cv2.conv.bias" : tensor<64xf32>
    %41 = mini_top.conv_silu %38, %39, %40 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x64x80x80xf32>, tensor<64x64x3x3xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %42 = mini_top.add %35, %41 : tensor<1x64x80x80xf32>, tensor<1x64x80x80xf32> -> tensor<1x64x80x80xf32>
    %43 = mini_top.weight "model.4.cv2.conv.weight" : tensor<64x128x1x1xf32>
    %44 = mini_top.weight "model.4.cv2.conv.bias" : tensor<64xf32>
    %45 = mini_top.conv_silu %25, %43, %44 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x80x80xf32>, tensor<64x128x1x1xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %46 = "mini_top.concat"(%42, %45) <{axis = 1 : i64}> : (tensor<1x64x80x80xf32>, tensor<1x64x80x80xf32>) -> tensor<1x128x80x80xf32>
    %47 = mini_top.weight "model.4.cv3.conv.weight" : tensor<128x128x1x1xf32>
    %48 = mini_top.weight "model.4.cv3.conv.bias" : tensor<128xf32>
    %49 = mini_top.conv_silu %46, %47, %48 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x80x80xf32>, tensor<128x128x1x1xf32>, tensor<128xf32> -> tensor<1x128x80x80xf32>
    %50 = mini_top.weight "model.5.conv.weight" : tensor<256x128x3x3xf32>
    %51 = mini_top.weight "model.5.conv.bias" : tensor<256xf32>
    %52 = mini_top.conv_silu %49, %50, %51 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x128x80x80xf32>, tensor<256x128x3x3xf32>, tensor<256xf32> -> tensor<1x256x40x40xf32>
    %53 = mini_top.weight "model.6.cv1.conv.weight" : tensor<128x256x1x1xf32>
    %54 = mini_top.weight "model.6.cv1.conv.bias" : tensor<128xf32>
    %55 = mini_top.conv_silu %52, %53, %54 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<128x256x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %56 = mini_top.weight "model.6.m.0.cv1.conv.weight" : tensor<128x128x1x1xf32>
    %57 = mini_top.weight "model.6.m.0.cv1.conv.bias" : tensor<128xf32>
    %58 = mini_top.conv_silu %55, %56, %57 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %59 = mini_top.weight "model.6.m.0.cv2.conv.weight" : tensor<128x128x3x3xf32>
    %60 = mini_top.weight "model.6.m.0.cv2.conv.bias" : tensor<128xf32>
    %61 = mini_top.conv_silu %58, %59, %60 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x3x3xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %62 = mini_top.add %55, %61 : tensor<1x128x40x40xf32>, tensor<1x128x40x40xf32> -> tensor<1x128x40x40xf32>
    %63 = mini_top.weight "model.6.m.1.cv1.conv.weight" : tensor<128x128x1x1xf32>
    %64 = mini_top.weight "model.6.m.1.cv1.conv.bias" : tensor<128xf32>
    %65 = mini_top.conv_silu %62, %63, %64 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %66 = mini_top.weight "model.6.m.1.cv2.conv.weight" : tensor<128x128x3x3xf32>
    %67 = mini_top.weight "model.6.m.1.cv2.conv.bias" : tensor<128xf32>
    %68 = mini_top.conv_silu %65, %66, %67 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x3x3xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %69 = mini_top.add %62, %68 : tensor<1x128x40x40xf32>, tensor<1x128x40x40xf32> -> tensor<1x128x40x40xf32>
    %70 = mini_top.weight "model.6.m.2.cv1.conv.weight" : tensor<128x128x1x1xf32>
    %71 = mini_top.weight "model.6.m.2.cv1.conv.bias" : tensor<128xf32>
    %72 = mini_top.conv_silu %69, %70, %71 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %73 = mini_top.weight "model.6.m.2.cv2.conv.weight" : tensor<128x128x3x3xf32>
    %74 = mini_top.weight "model.6.m.2.cv2.conv.bias" : tensor<128xf32>
    %75 = mini_top.conv_silu %72, %73, %74 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x3x3xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %76 = mini_top.add %69, %75 : tensor<1x128x40x40xf32>, tensor<1x128x40x40xf32> -> tensor<1x128x40x40xf32>
    %77 = mini_top.weight "model.6.cv2.conv.weight" : tensor<128x256x1x1xf32>
    %78 = mini_top.weight "model.6.cv2.conv.bias" : tensor<128xf32>
    %79 = mini_top.conv_silu %52, %77, %78 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<128x256x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %80 = "mini_top.concat"(%76, %79) <{axis = 1 : i64}> : (tensor<1x128x40x40xf32>, tensor<1x128x40x40xf32>) -> tensor<1x256x40x40xf32>
    %81 = mini_top.weight "model.6.cv3.conv.weight" : tensor<256x256x1x1xf32>
    %82 = mini_top.weight "model.6.cv3.conv.bias" : tensor<256xf32>
    %83 = mini_top.conv_silu %80, %81, %82 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<256x256x1x1xf32>, tensor<256xf32> -> tensor<1x256x40x40xf32>
    %84 = mini_top.weight "model.7.conv.weight" : tensor<512x256x3x3xf32>
    %85 = mini_top.weight "model.7.conv.bias" : tensor<512xf32>
    %86 = mini_top.conv_silu %83, %84, %85 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x256x40x40xf32>, tensor<512x256x3x3xf32>, tensor<512xf32> -> tensor<1x512x20x20xf32>
    %87 = mini_top.weight "model.8.cv1.conv.weight" : tensor<256x512x1x1xf32>
    %88 = mini_top.weight "model.8.cv1.conv.bias" : tensor<256xf32>
    %89 = mini_top.conv_silu %86, %87, %88 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<256x512x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %90 = mini_top.weight "model.8.m.0.cv1.conv.weight" : tensor<256x256x1x1xf32>
    %91 = mini_top.weight "model.8.m.0.cv1.conv.bias" : tensor<256xf32>
    %92 = mini_top.conv_silu %89, %90, %91 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x20x20xf32>, tensor<256x256x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %93 = mini_top.weight "model.8.m.0.cv2.conv.weight" : tensor<256x256x3x3xf32>
    %94 = mini_top.weight "model.8.m.0.cv2.conv.bias" : tensor<256xf32>
    %95 = mini_top.conv_silu %92, %93, %94 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x256x20x20xf32>, tensor<256x256x3x3xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %96 = mini_top.add %89, %95 : tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32> -> tensor<1x256x20x20xf32>
    %97 = mini_top.weight "model.8.cv2.conv.weight" : tensor<256x512x1x1xf32>
    %98 = mini_top.weight "model.8.cv2.conv.bias" : tensor<256xf32>
    %99 = mini_top.conv_silu %86, %97, %98 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<256x512x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %100 = "mini_top.concat"(%96, %99) <{axis = 1 : i64}> : (tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32>) -> tensor<1x512x20x20xf32>
    %101 = mini_top.weight "model.8.cv3.conv.weight" : tensor<512x512x1x1xf32>
    %102 = mini_top.weight "model.8.cv3.conv.bias" : tensor<512xf32>
    %103 = mini_top.conv_silu %100, %101, %102 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<512x512x1x1xf32>, tensor<512xf32> -> tensor<1x512x20x20xf32>
    %104 = mini_top.weight "model.9.cv1.conv.weight" : tensor<256x512x1x1xf32>
    %105 = mini_top.weight "model.9.cv1.conv.bias" : tensor<256xf32>
    %106 = mini_top.conv_silu %103, %104, %105 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<256x512x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %107 = "mini_top.maxpool"(%106) <{kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]}> : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32>
    %108 = "mini_top.maxpool"(%107) <{kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]}> : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32>
    %109 = "mini_top.maxpool"(%108) <{kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]}> : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xf32>
    %110 = "mini_top.concat"(%106, %107, %108, %109) <{axis = 1 : i64}> : (tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32>) -> tensor<1x1024x20x20xf32>
    %111 = mini_top.weight "model.9.cv2.conv.weight" : tensor<512x1024x1x1xf32>
    %112 = mini_top.weight "model.9.cv2.conv.bias" : tensor<512xf32>
    %113 = mini_top.conv_silu %110, %111, %112 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x1024x20x20xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32> -> tensor<1x512x20x20xf32>
    %114 = mini_top.weight "model.10.conv.weight" : tensor<256x512x1x1xf32>
    %115 = mini_top.weight "model.10.conv.bias" : tensor<256xf32>
    %116 = mini_top.conv_silu %113, %114, %115 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<256x512x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %117 = "mini_top.interp"(%116) <{mode = "nearest", target_h = 40 : i64, target_w = 40 : i64}> : (tensor<1x256x20x20xf32>) -> tensor<1x256x40x40xf32>
    %118 = "mini_top.concat"(%117, %83) <{axis = 1 : i64}> : (tensor<1x256x40x40xf32>, tensor<1x256x40x40xf32>) -> tensor<1x512x40x40xf32>
    %119 = mini_top.weight "model.13.cv1.conv.weight" : tensor<128x512x1x1xf32>
    %120 = mini_top.weight "model.13.cv1.conv.bias" : tensor<128xf32>
    %121 = mini_top.conv_silu %118, %119, %120 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x40x40xf32>, tensor<128x512x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %122 = mini_top.weight "model.13.m.0.cv1.conv.weight" : tensor<128x128x1x1xf32>
    %123 = mini_top.weight "model.13.m.0.cv1.conv.bias" : tensor<128xf32>
    %124 = mini_top.conv_silu %121, %122, %123 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %125 = mini_top.weight "model.13.m.0.cv2.conv.weight" : tensor<128x128x3x3xf32>
    %126 = mini_top.weight "model.13.m.0.cv2.conv.bias" : tensor<128xf32>
    %127 = mini_top.conv_silu %124, %125, %126 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x3x3xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %128 = mini_top.weight "model.13.cv2.conv.weight" : tensor<128x512x1x1xf32>
    %129 = mini_top.weight "model.13.cv2.conv.bias" : tensor<128xf32>
    %130 = mini_top.conv_silu %118, %128, %129 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x40x40xf32>, tensor<128x512x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %131 = "mini_top.concat"(%127, %130) <{axis = 1 : i64}> : (tensor<1x128x40x40xf32>, tensor<1x128x40x40xf32>) -> tensor<1x256x40x40xf32>
    %132 = mini_top.weight "model.13.cv3.conv.weight" : tensor<256x256x1x1xf32>
    %133 = mini_top.weight "model.13.cv3.conv.bias" : tensor<256xf32>
    %134 = mini_top.conv_silu %131, %132, %133 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<256x256x1x1xf32>, tensor<256xf32> -> tensor<1x256x40x40xf32>
    %135 = mini_top.weight "model.14.conv.weight" : tensor<128x256x1x1xf32>
    %136 = mini_top.weight "model.14.conv.bias" : tensor<128xf32>
    %137 = mini_top.conv_silu %134, %135, %136 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<128x256x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %138 = "mini_top.interp"(%137) <{mode = "nearest", target_h = 80 : i64, target_w = 80 : i64}> : (tensor<1x128x40x40xf32>) -> tensor<1x128x80x80xf32>
    %139 = "mini_top.concat"(%138, %49) <{axis = 1 : i64}> : (tensor<1x128x80x80xf32>, tensor<1x128x80x80xf32>) -> tensor<1x256x80x80xf32>
    %140 = mini_top.weight "model.17.cv1.conv.weight" : tensor<64x256x1x1xf32>
    %141 = mini_top.weight "model.17.cv1.conv.bias" : tensor<64xf32>
    %142 = mini_top.conv_silu %139, %140, %141 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x80x80xf32>, tensor<64x256x1x1xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %143 = mini_top.weight "model.17.m.0.cv1.conv.weight" : tensor<64x64x1x1xf32>
    %144 = mini_top.weight "model.17.m.0.cv1.conv.bias" : tensor<64xf32>
    %145 = mini_top.conv_silu %142, %143, %144 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x64x80x80xf32>, tensor<64x64x1x1xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %146 = mini_top.weight "model.17.m.0.cv2.conv.weight" : tensor<64x64x3x3xf32>
    %147 = mini_top.weight "model.17.m.0.cv2.conv.bias" : tensor<64xf32>
    %148 = mini_top.conv_silu %145, %146, %147 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x64x80x80xf32>, tensor<64x64x3x3xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %149 = mini_top.weight "model.17.cv2.conv.weight" : tensor<64x256x1x1xf32>
    %150 = mini_top.weight "model.17.cv2.conv.bias" : tensor<64xf32>
    %151 = mini_top.conv_silu %139, %149, %150 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x80x80xf32>, tensor<64x256x1x1xf32>, tensor<64xf32> -> tensor<1x64x80x80xf32>
    %152 = "mini_top.concat"(%148, %151) <{axis = 1 : i64}> : (tensor<1x64x80x80xf32>, tensor<1x64x80x80xf32>) -> tensor<1x128x80x80xf32>
    %153 = mini_top.weight "model.17.cv3.conv.weight" : tensor<128x128x1x1xf32>
    %154 = mini_top.weight "model.17.cv3.conv.bias" : tensor<128xf32>
    %155 = mini_top.conv_silu %152, %153, %154 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x80x80xf32>, tensor<128x128x1x1xf32>, tensor<128xf32> -> tensor<1x128x80x80xf32>
    %156 = mini_top.weight "model.24.m.0.weight" : tensor<255x128x1x1xf32>
    %157 = mini_top.weight "model.24.m.0.bias" : tensor<255xf32>
    %158 = mini_top.conv %155, %156, %157 {dilations = [1, 1], group = 1 : i64, pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x80x80xf32>, tensor<255x128x1x1xf32>, tensor<255xf32> -> tensor<1x255x80x80xf32>
    %159 = mini_top.reshape %158 {shape = [1, 3, 85, 80, 80]} : tensor<1x255x80x80xf32> -> tensor<1x3x85x80x80xf32>
    %160 = mini_top.permute %159 {order = [0, 1, 3, 4, 2]} : tensor<1x3x85x80x80xf32> -> tensor<1x3x80x80x85xf32>
    %161 = mini_top.weight "model.18.conv.weight" : tensor<128x128x3x3xf32>
    %162 = mini_top.weight "model.18.conv.bias" : tensor<128xf32>
    %163 = mini_top.conv_silu %155, %161, %162 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x128x80x80xf32>, tensor<128x128x3x3xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %164 = "mini_top.concat"(%163, %137) <{axis = 1 : i64}> : (tensor<1x128x40x40xf32>, tensor<1x128x40x40xf32>) -> tensor<1x256x40x40xf32>
    %165 = mini_top.weight "model.20.cv1.conv.weight" : tensor<128x256x1x1xf32>
    %166 = mini_top.weight "model.20.cv1.conv.bias" : tensor<128xf32>
    %167 = mini_top.conv_silu %164, %165, %166 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<128x256x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %168 = mini_top.weight "model.20.m.0.cv1.conv.weight" : tensor<128x128x1x1xf32>
    %169 = mini_top.weight "model.20.m.0.cv1.conv.bias" : tensor<128xf32>
    %170 = mini_top.conv_silu %167, %168, %169 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %171 = mini_top.weight "model.20.m.0.cv2.conv.weight" : tensor<128x128x3x3xf32>
    %172 = mini_top.weight "model.20.m.0.cv2.conv.bias" : tensor<128xf32>
    %173 = mini_top.conv_silu %170, %171, %172 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x128x40x40xf32>, tensor<128x128x3x3xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %174 = mini_top.weight "model.20.cv2.conv.weight" : tensor<128x256x1x1xf32>
    %175 = mini_top.weight "model.20.cv2.conv.bias" : tensor<128xf32>
    %176 = mini_top.conv_silu %164, %174, %175 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<128x256x1x1xf32>, tensor<128xf32> -> tensor<1x128x40x40xf32>
    %177 = "mini_top.concat"(%173, %176) <{axis = 1 : i64}> : (tensor<1x128x40x40xf32>, tensor<1x128x40x40xf32>) -> tensor<1x256x40x40xf32>
    %178 = mini_top.weight "model.20.cv3.conv.weight" : tensor<256x256x1x1xf32>
    %179 = mini_top.weight "model.20.cv3.conv.bias" : tensor<256xf32>
    %180 = mini_top.conv_silu %177, %178, %179 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<256x256x1x1xf32>, tensor<256xf32> -> tensor<1x256x40x40xf32>
    %181 = mini_top.weight "model.24.m.1.weight" : tensor<255x256x1x1xf32>
    %182 = mini_top.weight "model.24.m.1.bias" : tensor<255xf32>
    %183 = mini_top.conv %180, %181, %182 {dilations = [1, 1], group = 1 : i64, pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x40x40xf32>, tensor<255x256x1x1xf32>, tensor<255xf32> -> tensor<1x255x40x40xf32>
    %184 = mini_top.reshape %183 {shape = [1, 3, 85, 40, 40]} : tensor<1x255x40x40xf32> -> tensor<1x3x85x40x40xf32>
    %185 = mini_top.permute %184 {order = [0, 1, 3, 4, 2]} : tensor<1x3x85x40x40xf32> -> tensor<1x3x40x40x85xf32>
    %186 = mini_top.weight "model.21.conv.weight" : tensor<256x256x3x3xf32>
    %187 = mini_top.weight "model.21.conv.bias" : tensor<256xf32>
    %188 = mini_top.conv_silu %180, %186, %187 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x256x40x40xf32>, tensor<256x256x3x3xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %189 = "mini_top.concat"(%188, %116) <{axis = 1 : i64}> : (tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32>) -> tensor<1x512x20x20xf32>
    %190 = mini_top.weight "model.23.cv1.conv.weight" : tensor<256x512x1x1xf32>
    %191 = mini_top.weight "model.23.cv1.conv.bias" : tensor<256xf32>
    %192 = mini_top.conv_silu %189, %190, %191 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<256x512x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %193 = mini_top.weight "model.23.m.0.cv1.conv.weight" : tensor<256x256x1x1xf32>
    %194 = mini_top.weight "model.23.m.0.cv1.conv.bias" : tensor<256xf32>
    %195 = mini_top.conv_silu %192, %193, %194 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x256x20x20xf32>, tensor<256x256x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %196 = mini_top.weight "model.23.m.0.cv2.conv.weight" : tensor<256x256x3x3xf32>
    %197 = mini_top.weight "model.23.m.0.cv2.conv.bias" : tensor<256xf32>
    %198 = mini_top.conv_silu %195, %196, %197 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x256x20x20xf32>, tensor<256x256x3x3xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %199 = mini_top.weight "model.23.cv2.conv.weight" : tensor<256x512x1x1xf32>
    %200 = mini_top.weight "model.23.cv2.conv.bias" : tensor<256xf32>
    %201 = mini_top.conv_silu %189, %199, %200 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<256x512x1x1xf32>, tensor<256xf32> -> tensor<1x256x20x20xf32>
    %202 = "mini_top.concat"(%198, %201) <{axis = 1 : i64}> : (tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32>) -> tensor<1x512x20x20xf32>
    %203 = mini_top.weight "model.23.cv3.conv.weight" : tensor<512x512x1x1xf32>
    %204 = mini_top.weight "model.23.cv3.conv.bias" : tensor<512xf32>
    %205 = mini_top.conv_silu %202, %203, %204 {dilations = [1, 1], group = 1 : i64, mini_top.gpu_lowering = "conv_silu_cuda_candidate", pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<512x512x1x1xf32>, tensor<512xf32> -> tensor<1x512x20x20xf32>
    %206 = mini_top.weight "model.24.m.2.weight" : tensor<255x512x1x1xf32>
    %207 = mini_top.weight "model.24.m.2.bias" : tensor<255xf32>
    %208 = mini_top.conv %205, %206, %207 {dilations = [1, 1], group = 1 : i64, pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x512x20x20xf32>, tensor<255x512x1x1xf32>, tensor<255xf32> -> tensor<1x255x20x20xf32>
    %209 = mini_top.reshape %208 {shape = [1, 3, 85, 20, 20]} : tensor<1x255x20x20xf32> -> tensor<1x3x85x20x20xf32>
    %210 = mini_top.permute %209 {order = [0, 1, 3, 4, 2]} : tensor<1x3x85x20x20xf32> -> tensor<1x3x20x20x85xf32>
    return %160, %185, %210 : tensor<1x3x80x80x85xf32>, tensor<1x3x40x40x85xf32>, tensor<1x3x20x20x85xf32>
  }
}

