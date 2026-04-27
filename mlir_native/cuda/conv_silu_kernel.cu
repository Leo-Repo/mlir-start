#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void mini_top_conv_silu_nchw_f32(
    const float *input, const float *weight, const float *bias, float *output,
    int n, int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
    int dilation_w) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * oc * oh * ow;
  if (linear >= total)
    return;

  int x = linear % ow;
  int y = (linear / ow) % oh;
  int o = (linear / (ow * oh)) % oc;
  int b = linear / (ow * oh * oc);

  float acc = bias ? bias[o] : 0.0f;
  for (int c = 0; c < ic; ++c) {
    for (int r = 0; r < kh; ++r) {
      for (int s = 0; s < kw; ++s) {
        int in_y = y * stride_h + r * dilation_h - pad_h;
        int in_x = x * stride_w + s * dilation_w - pad_w;
        if (in_y < 0 || in_y >= ih || in_x < 0 || in_x >= iw)
          continue;
        int in_idx = ((b * ic + c) * ih + in_y) * iw + in_x;
        int wt_idx = ((o * ic + c) * kh + r) * kw + s;
        acc += input[in_idx] * weight[wt_idx];
      }
    }
  }

  output[linear] = acc / (1.0f + expf(-acc));
}

extern "C" int mini_top_launch_conv_silu_nchw_f32(
    const float *input, const float *weight, const float *bias, float *output,
    int n, int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
    int dilation_w, cudaStream_t stream) {
  int total = n * oc * oh * ow;
  int block = 256;
  int grid = (total + block - 1) / block;
  mini_top_conv_silu_nchw_f32<<<grid, block, 0, stream>>>(
      input, weight, bias, output, n, ic, ih, iw, oc, oh, ow, kh, kw, stride_h,
      stride_w, pad_h, pad_w, dilation_h, dilation_w);
  return static_cast<int>(cudaGetLastError());
}
