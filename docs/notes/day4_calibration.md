# Day 4 Calibration

## Goal

参考 `tpu-mlir` 的校准流程，自己实现一版最小可运行的 calibration：

- 输入：`data/images`
- 模型：本地 PyTorch `yolov5s.pt`
- 方法：对 YOLOv5 顶层模块输出挂 forward hook，统计激活 `absmax`

## Implementation

使用脚本：

- [`mini_ptq.py`](/home/jay/projs/mlir_start/mini_ptq.py)

命令：

```bash
python mini_ptq.py --mode calibrate --calib-num 100
```

本次实际执行命令：

```bash
python mini_ptq.py --mode all --calib-num 100 --eval-num 10
```

## Output

生成校准表：

- [`yolov5s_mini_cali_table.json`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_cali_table.json)

校准统计结果：

- 校准 tensor 数量：`28`
- 最大激活阈值 tensor：`model.24.0`
- 对应 shape：`[1, 25200, 85]`
- 对应 absmax：`734.6907348632812`

前 10 个最大激活 tensor：

- `model.24.0` -> `734.6907`
- `model.1` -> `89.3055`
- `model.0` -> `50.0672`
- `model.20` -> `35.3468`
- `model.17` -> `33.8894`
- `model.23` -> `29.9739`
- `model.24.1.0` -> `22.2459`
- `model.2` -> `19.1485`
- `model.24.1.2` -> `18.3804`
- `model.24.1.1` -> `17.2605`

## Notes

- 这版 calibration 不是官方 `run_calibration.py` 的完整等价实现。
- 它更像一版“最小 PTQ 校准器”：
  - 直接在 PyTorch 前向中收集激活范围
  - 用于后续 fake quant 和误差分析
