# 第一周任务清单（高执行版）

## 周目标

1. 跑通 `yolov5s.onnx -> Top MLIR`
2. 跑通 calibration 与 INT8 部署
3. 建立第一版 IR 对照笔记与误差记录

## Day 1：准备与盘点

1. 检查模型资产：确认 `models/yolov5s.onnx` 是否可用
2. 盘点工具链：记录 tpu-mlir 环境、python 依赖、GPU/驱动信息
3. 建立实验目录：`experiments/tpu_mlir/01_onnx_to_mlir/`、`experiments/tpu_mlir/02_quant/`、`docs/notes/`

产物：`docs/notes/day1_env.md`

## Day 2：ONNX -> Top MLIR 打通

1. 用 `model_transform.py` 完成导入
2. 保留关键输入参数（shape/mean/scale/pixel_format）
3. 产出 `.mlir` 与一次 baseline 推理结果

产物：`docs/notes/day2_transform.md`

## Day 3：IR 阅读与映射

1. 从 `.mlir` 中抽取关键算子链（Conv、Concat、Upsample、Detect 相关）
2. 建立 ONNX 节点到 MLIR op 映射表
3. 标注你不理解的 5~10 个 op/属性，作为后续突破点

产物：`docs/notes/day3_ir_mapping.md`

## Day 4：Calibration

1. 准备校准样本（首版 100 张）
2. 执行 `run_calibration.py`
3. 记录 calibration 配置与生成表

产物：`docs/notes/day4_calibration.md`

## Day 5：Quantization 与部署产物

1. 执行 `model_deploy.py --quantize INT8`
2. 对比 FP32 与 INT8 输出（先做 10 张固定样本）
3. 记录异常样本与误差趋势

产物：`docs/notes/day5_quant_eval.md`

## Day 6：误差归因与复盘

1. 汇总误差最大的样本
2. 粗分是前处理、算子、后处理还是量化参数问题
3. 列出下一周优化计划

产物：`docs/notes/day6_error_analysis.md`

## Day 7：阶段验收

1. 对照验收标准自检
2. 输出“本周结论 + 下周计划”
3. 明确 GPU lowering 先做哪个算子小样

产物：`docs/notes/week1_summary.md`

## 本周验收标准

1. 有可复现的 ONNX->MLIR 命令与产物
2. 有可复现的 calibration/quantization 命令与产物
3. 有一份 IR 映射笔记
4. 有一份误差分析笔记
