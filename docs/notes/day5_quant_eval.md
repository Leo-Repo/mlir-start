# Day 5 Quantization And Eval

## Goal

参考 `tpu-mlir` 的 INT8 流程，自己做一版最小量化实验：

- 权重：对 `Conv2d` 做对称 int8 fake quant
- 激活：按 calibration table 做对称 int8 fake quant
- 验证：比较 FP32 与 fake-quant 输出误差

## Implementation

脚本：

- [`mini_ptq.py`](/home/jay/projs/mlir_start/mini_ptq.py)

命令：

```bash
python mini_ptq.py --mode all --calib-num 100 --eval-num 10
```

输出文件：

- 报告 JSON：
  - [`yolov5s_mini_ptq_report.json`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_ptq_report.json)
- 摘要 Markdown：
  - [`yolov5s_mini_ptq_summary.md`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_ptq_summary.md)

## Worst Images

前 10 个误差最大的评估图片：

- `000000000064.jpg` -> cosine `0.998938`, mse `7.117341`
- `000000000071.jpg` -> cosine `0.999177`, mse `5.506405`
- `000000000030.jpg` -> cosine `0.999224`, mse `5.273715`
- `000000000036.jpg` -> cosine `0.999247`, mse `5.180410`
- `000000000034.jpg` -> cosine `0.999271`, mse `4.910140`
- `000000000061.jpg` -> cosine `0.999333`, mse `4.455350`
- `000000000049.jpg` -> cosine `0.999343`, mse `4.398650`
- `000000000025.jpg` -> cosine `0.999485`, mse `3.455453`
- `000000000042.jpg` -> cosine `0.999641`, mse `2.469405`
- `000000000009.jpg` -> cosine `0.999689`, mse `2.156739`

## Quick Reading

- 最差图片的输出余弦相似度仍然大于 `0.998`
- 说明这版最小 fake quant 流程虽然粗糙，但整体输出保持了较高相似度
- 误差主要体现在幅值偏移，而不是整体方向完全失真

## Notes

- 当前量化是本地 fake quant，不是生成真正可部署的 INT8 bmodel
- 它的价值在于：
  - 建立 calibration -> quant -> eval 的最小闭环
  - 为 Day 6 的误差归因提供层级数据
