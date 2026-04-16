# 第一周任务清单（高执行版）

## 周目标

1. 跑通 `yolov5s.onnx -> Top MLIR`
2. 建立第一版 IR 对照笔记与误差记录
3. 从 Python 原型切到 MLIR 原生骨架，开始第一步 lowering

## Day 1：准备与盘点

1. 检查模型资产：确认 `models/yolov5s.onnx` 是否可用
2. 盘点工具链：记录 tpu-mlir 环境、python 依赖、GPU/驱动信息
3. 建立实验目录：`experiments/tpu_mlir/01_onnx_to_mlir/`、`experiments/tpu_mlir/02_quant/`、`docs/notes/`

产物：`docs/notes/day1_env.md`

## Day 2：ONNX -> 原始 Top MLIR

1. 用 `legacy_python_frontend/model_transform.py` 完成 `yolov5s.onnx -> 原始 Top MLIR`
2. 固化关键输入参数（shape/mean/scale/pixel_format/output_names）
3. 产出 `.mlir`、权重 `.npz` 与转换说明文档
4. 明确“当前产物是原始 Top MLIR，还未经过 canonicalize”

产物：`docs/notes/day2_transform.md`

## Day 3：原始 IR 阅读 + canonicalize 前后对比

1. 从原始 `.mlir` 中抽取关键算子链（Conv、Concat、Upsample、Detect head 输出相关）
2. 建立 ONNX 节点到 Top MLIR op 映射表
3. 对原始 IR 执行或观察 `canonicalize` 后结果，记录前后变化
4. 使用本地 [`top_canonicalize.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_canonicalize.py) 对 `yolov5s.mlir` 生成 `yolov5s_canonical.mlir`，重点观察 `MaxPool`、`Interp`、`Concat`、`Reshape`、`Permute` 的变化
5. 标注你不理解的 5~10 个 op/属性或 fold 现象，作为后续突破点

产物：`docs/notes/day3_ir_mapping.md`

## Day 4：Calibration 认知与边界澄清

1. 准备校准样本（首版 100 张）
2. 梳理 calibration 应面向 `canonical Top MLIR` 的 blob 对比，而不是检测后处理结果
3. 记录 calibration 配置与生成表

产物：`docs/notes/day4_calibration.md`

## Day 5：Quantization 与部署产物认知

1. 明确官方链路与本地最小 PTQ 原型的区别
2. 对比 FP32 与 INT8 的 blob / tensor 视角
3. 记录异常样本与误差趋势

产物：`docs/notes/day5_quant_eval.md`

## Day 6：误差归因与复盘

1. 汇总误差最大的样本
2. 粗分是前处理、算子、后处理还是量化参数问题
3. 列出下一周优化计划

产物：`docs/notes/day6_error_analysis.md`

## Day 7：MLIR 原生切换

1. 梳理项目结构，明确 Python 原型线与 MLIR 原生线的边界
2. 冻结 Python runtime 扩展，只保留参考实现角色
3. 在 `mlir_native/` 中完成第一个原生 rewrite 和第一个 lowering pass
4. 产出可执行的 `mini-top-opt`

产物：`docs/project_structure.md`

## 阶段验收

1. 对照验收标准自检
2. 输出“本周结论 + 下周计划”
3. 明确 GPU lowering 先做哪个算子小样

产物：`docs/notes/week1_summary.md`

## 本周验收标准

1. 有可复现的 ONNX->MLIR 命令与产物
2. 有 calibration/quantization 边界说明与现有实验产物
3. 有一份 IR 映射笔记
4. 有一份误差分析笔记
5. 有一个可编译、可运行的 MLIR 原生子工程
