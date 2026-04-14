# Project Log

## 2026-04-09

### Progress

- 明确第一周主线任务是跑通 `yolov5s.onnx -> Top MLIR`，并确认项目内相关说明主要集中在 [`week1.md`](/home/jay/projs/mlir_start/week1.md)、[`README.md`](/home/jay/projs/mlir_start/README.md)、[`docs/yolov5_pipeline.md`](/home/jay/projs/mlir_start/docs/yolov5_pipeline.md)。
- 检查仓库现状，确认已经存在 [`models/yolov5s.onnx`](/home/jay/projs/mlir_start/models/yolov5s.onnx)，可以直接进入 ONNX 前端导入阶段。
- 明确目标不是调用 `tpu-mlir` 官方 `model_transform.py`，而是独立实现一个最小可用的 ONNX importer。
- 新建并实现 [`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py) 第一版，完成独立的 `ONNX -> Top MLIR` 文本生成框架。
- 参考 `tpu-mlir` 官方产物格式，确定了 Top MLIR 的总体输出形态、`top.Input` / `top.Weight` / `top.Conv` / `top.Permute` 等文本风格。

### Key Decisions

- 放弃“本地包装脚本调用官方包”的方向，改为独立 importer 路线。
- 第一阶段只覆盖 `yolov5s` 需要的最小算子与 shape 子图，而不追求全量 ONNX 兼容。
- 输出目标对齐 `tpu-mlir` 官方示例中的三个检测头输出：`350 / 498 / 646`。

### Problems

- 初始环境里没有可直接导入的 `onnx` Python 包，导致无法在当前 shell 下立即做图遍历验证。
- 需要明确 `Top MLIR` 的文本结构、loc 风格和权重表达方式，避免生成不符合阅读预期的 IR。

### Resolution

- 通过读取本机已有的 `tpu-mlir` 源码和回归产物，提取官方实现和输出格式作为对照。
- 先实现“结构正确、可读”的 importer，再逐步补齐 shape、loc、常量折叠和子图裁剪。

## 2026-04-10

### Progress

- 在 `llama` conda 环境中真实执行 [`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py)，沿着报错路径逐步补齐 ONNX shape 子图支持。
- 新增并稳定支持：
  - `Shape`
  - `Gather`
  - `Unsqueeze`
  - `Cast`
  - `Range`
  - `Expand`
  - `ConstantOfShape`
- 为 `Sigmoid`、`Mul`、`Add`、`Concat`、`Reshape`、`Transpose`、`Slice` 增加“输入全为常量时的常量折叠”。
- 修复“遍历整张 ONNX 图”带来的无关后处理问题，改为只保留 `350 / 498 / 646` 依赖的最小子图，并做拓扑排序。
- 成功生成：
  - [`experiments/01_onnx_to_mlir/yolov5s.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s.mlir)
  - [`experiments/01_onnx_to_mlir/yolov5s_top_f32_all_origin_weight.npz`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s_top_f32_all_origin_weight.npz)
- 为 MLIR 输出补充了 loc 表，支持将普通 op 与 ONNX 节点名关联。
- 编写设计文档 [`model_transform.md`](/home/jay/projs/mlir_start/model_transform.md)，说明 importer 的架构、原理和支持范围。
- 在 [`week1.md`](/home/jay/projs/mlir_start/week1.md) 中将周计划调整为：
  - Day 2: `ONNX -> 原始 Top MLIR`
  - Day 3: `原始 IR 阅读 + canonicalize 前后对比`

### Key Decisions

- 以官方 `yolov5s` 示例的三个 head 输出为子图截断点，而不是把 detect/postprocess 整段一起导入。
- 将 shape 相关辅助子图视为 importer 阶段的常量计算，而不是保留为真正的运行期图语义。
- 把 `model_transform.py` 的定位明确为“学习型前端 importer”，不是完整工业实现。

### Problems

- 多次遇到 shape 子图相关报错，例如：
  - `Unsupported ONNX op Shape`
  - `Expected constant tensor, got dynamic value: 348`
  - `Unsupported ONNX op Range`
  - `Unsupported ONNX op ConstantOfShape`
  - `Unsupported ONNX op Equal`
- 子图顺序最初不是稳定的拓扑序，出现了“值在定义前被引用”的问题。
- 初版 loc 风格不够清晰，函数参数位置和 `top.Input` 复用了同一个 loc。

### Resolution

- 通过逐步查看 ONNX 局部子图上下文，沿报错链精确补齐常量折叠逻辑。
- 将子图选择改为“从目标输出反向 DFS + 后序追加”，保证拓扑顺序稳定。
- 最终把 loc 风格调整为更接近 `tpu-mlir` 官方形式：
  - 顶部 `#loc = loc(unknown)`
  - `func.func` 参数用 `loc(unknown)`
  - 普通 op 用 `#locN = loc("ONNX节点名")`

## 2026-04-12

### Progress

- 针对 IR 阅读阶段发现的 shape 问题，修复了 [`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py) 中的静态 shape 传播。
- 为以下算子新增本地输出 shape 推导：
  - `Conv`
  - `Concat`
  - `Resize` / `top.Interp`
  - `Reshape`
  - `Transpose` / `top.Permute`
- 为 `Sigmoid`、`Mul`、`Add` 增加 shape 透传，避免静态 shape 在中间链路丢失。
- 修复 `build()` 中函数返回类型提前写死的问题，使 `func.func @main` 返回类型与最终输出 `ValueRef` 保持一致。
- 重新生成 [`yolov5s.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s.mlir)，现在前几层和三个 head 输出都已经是静态 shape。
- 开始进入“原始 IR 阅读”实战，抽出了 `350 / 498 / 646` 三条输出链。
- 新增本地 canonicalize 脚本 [`top_canonicalize.py`](/home/jay/projs/mlir_start/top_canonicalize.py)，用于在不依赖 `tpuc-opt` 的情况下，对 `yolov5s` 相关 Top MLIR 做一层轻量规范化。
- 为 [`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py) 增加 `--canonicalize` 开关，可在导入结束后同步生成 canonicalize 后版本：
  - [`experiments/01_onnx_to_mlir/yolov5s_canonical.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s_canonical.mlir)
- 在 `llama` conda 环境中重新执行了整条链路，确认原始 IR、权重 `.npz` 和 canonicalize 后 IR 都能稳定生成。

### Output Chains Identified

- `350 = Conv_196 -> Reshape_213 -> Transpose_214`
- `498 = Conv_308 -> Reshape_325 -> Transpose_326`
- `646 = Conv_420 -> Reshape_437 -> Transpose_438`

这三条链对应三路检测头输出，形状分别是：

- `1x3x80x80x85`
- `1x3x40x40x85`
- `1x3x20x20x85`

### Questions Answered

- 解释了为什么官方示例选择 `350 / 498 / 646` 作为输出，而不是直接停在 `Conv_196 / 308 / 420`。
- 解释了为什么 `Reshape + Transpose` 更适合作为“去掉真正后处理后的标准输出”。
- 解释了为什么这里做的是静态 shape 推导，而不是动态推导：
  - 当前输入 shape 固定
  - 目标是静态编译链路
  - importer 阶段更适合编译期 propagation，而不是运行期 shape 计算
- 解释了官方 `tpu-mlir` 中 canonicalize 的位置：它属于 Top 层正式优化流水线的一部分，而本项目当前新增的是面向 `yolov5s` 的本地版 canonicalize。

### Canonicalize Rules Added

[`top_canonicalize.py`](/home/jay/projs/mlir_start/top_canonicalize.py) 目前支持：

- 清洗 `top.Interp` 中的字节串属性：
  - `"b'asymmetric'" -> "asymmetric"`
  - `"b'nearest'" -> "nearest"`
- 为以下 op 重新推导静态 shape：
  - `top.MaxPool`
  - `top.Interp`
  - `top.Concat`
  - `top.Reshape`
  - `top.Permute`
- 删除 no-op `top.Reshape`
- 删除 identity `top.Permute`

本次在 `yolov5s_canonical.mlir` 中已经观察到的直接效果包括：

- SPPF 段三层 `top.MaxPool` 全部变成静态 `tensor<1x256x20x20xf32>`
- `top.Concat(%172, %173, %174, %175)` 的输入和输出 shape 全部静态化
- `top.Interp` 的 `coord_mode` / `mode` 属性从字节串风格清洗为普通字符串

### Problems

- 初版生成的 IR 中存在大量 `tensor<?x64x?x?xf32>` 这类不完整 shape。
- `func.func` 返回类型一度仍然保留 `?`，与最终 `top.Permute` 的静态 shape 不一致。
- 位置风格在一次生成中没有完全同步到磁盘文件，需要直接用当前 `build()` 结果覆盖写回。

### Resolution

- 新增显式 shape helper，并在关键 `emit_*()` 中写回 `value_info`。
- 用最终输出 `ValueRef.type_str` 回填函数返回签名。
- 用当前 `build()` 的文本结果直接覆盖写回磁盘，确保文件内容与最新实现一致。
- 将 canonicalize 单独实现为独立脚本，而不是继续塞回 importer，保留“原始 IR”和“规范化后 IR”两份产物，便于 Day 3 对比阅读。

## Working Agreement

- 本文件用于记录每日进展、问题、决定和处理结果。
- 后续在这个项目里继续推进时，默认同步更新本日志。
