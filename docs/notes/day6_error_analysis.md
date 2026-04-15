# Day 6 Error Analysis

## Goal

对本地 fake quant 结果做误差归因，回答：

- 哪些层最容易被量化破坏
- 哪些图片误差更大
- 下一步该优先优化什么

## Worst Layers

误差最明显的前 10 层：

- `model.8` -> cosine `0.895773`, mse `0.045584`
- `model.7` -> cosine `0.896346`, mse `0.027835`
- `model.6` -> cosine `0.896465`, mse `0.028752`
- `model.20` -> cosine `0.899761`, mse `0.275560`
- `model.21` -> cosine `0.903530`, mse `0.036191`
- `model.12` -> cosine `0.911002`, mse `0.029499`
- `model.23` -> cosine `0.911019`, mse `0.133411`
- `model.22` -> cosine `0.914045`, mse `0.033484`
- `model.18` -> cosine `0.914836`, mse `0.038644`
- `model.13` -> cosine `0.918077`, mse `0.031299`

## Initial Attribution

从结果上看，误差主要集中在：

- 中后段 backbone / neck 交界区域
- 接近检测头的高语义特征层
- 激活值幅度较大的层，例如 `model.20`、`model.23`

这和 calibration 结果是相互呼应的：

- `model.24.0` 激活范围最大
- 检测头前后的层对量化更敏感

## Likely Causes

1. 当前实现只做了最小对称 fake quant

- 没有做更细粒度的策略选择
- 没有使用 qtable / mixed precision
- 没有做更复杂的阈值搜索

2. 权重量化和激活量化都比较粗

- Conv 权重虽然做了 per-channel fake quant
- 激活仍然是 per-tensor 对称量化

3. YOLOv5 的检测头对幅值变化敏感

- 特别是接近输出的特征层
- 小的量化误差会被放大到最终检测输出

## Next Step

下一步最值得尝试的优化方向：

1. 对最差层做局部不量化或保留 FP32
2. 把激活量化从 per-tensor 改成更细粒度策略
3. 扩展 calibration 方法，不只看简单 `absmax`
4. 单独分析三路 detect head 输出 `80x80 / 40x40 / 20x20`

## Artifacts

- 校准表：
  - [`yolov5s_mini_cali_table.json`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_cali_table.json)
- 误差报告：
  - [`yolov5s_mini_ptq_report.json`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_ptq_report.json)
- 摘要：
  - [`yolov5s_mini_ptq_summary.md`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_ptq_summary.md)
