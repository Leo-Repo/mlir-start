# Mini PTQ Summary

## Calibration

- 校准 tensor 数量: `28`
- 最大激活阈值 tensor: `model.24.0`

## Worst Output Images

- `000000000064.jpg`: cosine=0.998938, mse=7.117341e+00, mae=1.954665e-01, max_abs_diff=4.037549e+02
- `000000000071.jpg`: cosine=0.999177, mse=5.506405e+00, mae=2.106951e-01, max_abs_diff=3.926572e+02
- `000000000030.jpg`: cosine=0.999224, mse=5.273715e+00, mae=2.160264e-01, max_abs_diff=4.986168e+02
- `000000000036.jpg`: cosine=0.999247, mse=5.180410e+00, mae=1.796518e-01, max_abs_diff=2.999368e+02
- `000000000034.jpg`: cosine=0.999271, mse=4.910140e+00, mae=1.999492e-01, max_abs_diff=3.617843e+02
- `000000000061.jpg`: cosine=0.999333, mse=4.455350e+00, mae=1.876800e-01, max_abs_diff=3.650552e+02
- `000000000049.jpg`: cosine=0.999343, mse=4.398650e+00, mae=1.793217e-01, max_abs_diff=2.671017e+02
- `000000000025.jpg`: cosine=0.999485, mse=3.455453e+00, mae=1.894277e-01, max_abs_diff=2.385690e+02
- `000000000042.jpg`: cosine=0.999641, mse=2.469405e+00, mae=1.654395e-01, max_abs_diff=2.459966e+02
- `000000000009.jpg`: cosine=0.999689, mse=2.156739e+00, mae=1.583931e-01, max_abs_diff=2.477155e+02

## Worst Layers

- `model.8`: cosine=0.895773, mse=4.558414e-02, mae=1.173158e-01, max_abs_diff=2.784713e+00
- `model.7`: cosine=0.896346, mse=2.783542e-02, mae=8.155641e-02, max_abs_diff=3.159919e+00
- `model.6`: cosine=0.896465, mse=2.875196e-02, mae=8.017488e-02, max_abs_diff=3.831316e+00
- `model.20`: cosine=0.899761, mse=2.755600e-01, mae=2.411118e-01, max_abs_diff=1.304150e+01
- `model.21`: cosine=0.903530, mse=3.619073e-02, mae=9.720075e-02, max_abs_diff=3.802619e+00
- `model.12`: cosine=0.911002, mse=2.949942e-02, mae=8.715031e-02, max_abs_diff=3.831316e+00
- `model.23`: cosine=0.911019, mse=1.334109e-01, mae=1.809209e-01, max_abs_diff=9.052337e+00
- `model.22`: cosine=0.914045, mse=3.348413e-02, mae=9.374049e-02, max_abs_diff=3.802619e+00
- `model.18`: cosine=0.914836, mse=3.864423e-02, mae=1.079754e-01, max_abs_diff=4.423187e+00
- `model.13`: cosine=0.918077, mse=3.129935e-02, mae=9.544074e-02, max_abs_diff=2.578779e+00
- `model.10`: cosine=0.925559, mse=3.000975e-02, mae=9.211815e-02, max_abs_diff=3.105278e+00
- `model.11`: cosine=0.925621, mse=3.000975e-02, mae=9.211815e-02, max_abs_diff=3.105278e+00
- `model.19`: cosine=0.927439, mse=3.665847e-02, mae=1.080521e-01, max_abs_diff=4.423187e+00
- `model.17`: cosine=0.928034, mse=3.958315e-01, mae=2.986793e-01, max_abs_diff=1.062835e+01
- `model.9`: cosine=0.937700, mse=1.615659e-02, mae=6.514382e-02, max_abs_diff=3.112177e+00
