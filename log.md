# Project Log

## 2026-04-09

### Progress

- 明确第一周主线任务是跑通 `yolov5s.onnx -> Top MLIR`，并确认项目内相关说明主要集中在 [`docs/archive/week1_archived.md`](/home/jay/projs/mlir_start/docs/archive/week1_archived.md)、[`README.md`](/home/jay/projs/mlir_start/README.md)、[`docs/yolov5_pipeline.md`](/home/jay/projs/mlir_start/docs/yolov5_pipeline.md)。
- 检查仓库现状，确认已经存在 [`models/yolov5s.onnx`](/home/jay/projs/mlir_start/models/yolov5s.onnx)，可以直接进入 ONNX 前端导入阶段。
- 明确目标不是调用 `tpu-mlir` 官方 `model_transform.py`，而是独立实现一个最小可用的 ONNX importer。
- 新建并实现 [`model_transform.py`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.py) 第一版，完成独立的 `ONNX -> Top MLIR` 文本生成框架。
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

- 在 `llama` conda 环境中真实执行 [`model_transform.py`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.py)，沿着报错路径逐步补齐 ONNX shape 子图支持。
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
- 编写设计文档 [`model_transform.md`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.md)，说明 importer 的架构、原理和支持范围。
- 在 [`docs/archive/week1_archived.md`](/home/jay/projs/mlir_start/docs/archive/week1_archived.md) 中将周计划调整为：
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

- 针对 IR 阅读阶段发现的 shape 问题，修复了 [`model_transform.py`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.py) 中的静态 shape 传播。
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
- 新增本地 canonicalize 脚本 [`top_canonicalize.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_canonicalize.py)，用于在不依赖 `tpuc-opt` 的情况下，对 `yolov5s` 相关 Top MLIR 做一层轻量规范化。
- 为 [`model_transform.py`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.py) 增加 `--canonicalize` 开关，可在导入结束后同步生成 canonicalize 后版本：
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

[`top_canonicalize.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_canonicalize.py) 目前支持：

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

## 2026-04-15

### Progress

- 明确开始规划本地版 YOLOv5 fusion 路线，而不是继续把所有变化都堆到 canonicalize 中。
- 重新检查 [`experiments/01_onnx_to_mlir/yolov5s.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s.mlir) 中最典型的激活模式，确认大量重复出现：
  - `top.Conv`
  - `top.Sigmoid`
  - `top.Mul`
- 明确这组模式对应 YOLOv5 中的 SiLU 结构，是最适合做第一条本地 fusion 规则的切入点。

### Key Decisions

- 不把“算子融合”继续塞进 [`top_canonicalize.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_canonicalize.py)。
- 将本地 fusion 设计为 canonicalize 之后的独立层更合适：
  - importer 负责原始导入
  - canonicalize 负责规范化和静态化
  - fusion 负责结构重写和性能导向优化
- 第一条 fusion 规则优先选择：
  - `top.Conv -> top.Sigmoid -> top.Mul`
  - 其中 `top.Mul` 的两个输入必须正好是 `Conv` 输出和对应 `Sigmoid(Conv)` 输出

### Questions Answered

- 明确了“canonicalize”和“fusion”的职责边界：
  - `canonicalize` 更偏向语义不变的整理、补 shape、去 no-op、清洗属性
  - `fusion` 更偏向图结构重写，把多条 op 合并成一个更高层、更适合后续 lowering 的模式
- 明确本地版 YOLOv5 fusion 最适合放在“canonicalize 之后、lowering 之前”的独立 pass 层。

### Next Step

- 设计一个单独的本地 fusion 脚本，例如：
  - [`top_fuse.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_fuse.py)
- 第一阶段先只支持 SiLU pattern 融合，并保留足够清晰的 loc / 注释 / 映射关系，方便继续做 IR 阅读。

### Fusion Prototype Landed

- 已新增第一版 [`top_fuse.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_fuse.py)。
- 当前规则只做一条保守融合：
  - `top.Conv -> top.Sigmoid -> top.Mul`
  - 最终改写为更接近官方 `tpu-mlir` 的形式：`top.Conv -> top.SiLU`
- 融合条件：
  - `Sigmoid` 必须直接消费 `Conv` 结果
  - `Mul` 必须正好消费 `Conv` 输出和对应的 `Sigmoid(Conv)` 输出
  - `Sigmoid` 输出只能被该 `Mul` 使用
- 当前在 [`experiments/01_onnx_to_mlir/yolov5s_fused.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s_fused.mlir) 中已成功融合 `57` 处 SiLU pattern。
- 对照本机 `tpu-mlir` 实现确认：
  - [`TorchConverter.py`](/home/jay/projs/tpu-mlir/python/transform/TorchConverter.py)
  - [`regression_out/yolov5s.mlir`](/home/jay/projs/tpu-mlir/regression/regression_out/yolov5s_bm1684x_num_core_1/yolov5s.mlir)
  官方风格确实使用 `top.SiLU`。

### Current Limitation

- 当前产物已经比最初的 `top.FusedConvSilu` 更接近官方可继续 lowering 的表达方式，但仍然只是本地文本级重写 pass。
- 还没有进一步接入官方 `tpuc-opt` 流水线验证真正的后续 lowering 兼容性。

### Fusion Analysis Utilities

- 为 [`top_fuse.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_fuse.py) 增加了：
  - `--summary-only`
  - `--dump-patterns`
- 现在可以先只做 pattern 统计，而不写回融合后的 MLIR。
- 当前统计结果：
  - `Conv + Sigmoid + Mul -> Conv + SiLU` 共检测到 `57` 处

### Minimal PTQ Landed

- 因为本机 `tpu-mlir` 运行时存在 Python `3.10` / `3.11` ABI 不匹配，官方 `run_calibration.py` / `model_deploy.py` 在当前 `llama` 环境下无法直接运行。
- 因此改为参考官方工程分层，自行实现最小版 PTQ 流程：
  - [`mini_ptq.py`](/home/jay/projs/mlir_start/legacy_python_frontend/mini_ptq.py)
- 该脚本基于本机已有的 [`/home/jay/projs/yolov5`](/home/jay/projs/yolov5) 和 `yolov5s.pt`，实现了：
  - calibration
  - fake quantization
  - layer-wise error attribution

### PTQ Outputs

- 校准表：
  - [`yolov5s_mini_cali_table.json`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_cali_table.json)
- 量化误差报告：
  - [`yolov5s_mini_ptq_report.json`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_ptq_report.json)
- 摘要：
  - [`yolov5s_mini_ptq_summary.md`](/home/jay/projs/mlir_start/experiments/02_quant/yolov5s_mini_ptq_summary.md)

### PTQ Result Snapshot

- 校准图片数：`100`
- 评估图片数：`10`
- 校准 tensor 数量：`28`
- 最大激活阈值 tensor：`model.24.0`
- 最差图片：
  - `000000000064.jpg`
  - cosine `0.998938`
  - mse `7.117341`
- 最差层：
  - `model.8`
  - cosine `0.895773`
  - mse `0.045584`

### PTQ Flow Clarification

- 重新对齐了 `tpu-mlir` 官方链路的关键边界：
  - Calibration / Quantization 的主对象应该是 [`yolov5s_canonical.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s_canonical.mlir) 这类 Top MLIR，而不是 PyTorch 模型本身。
- 当前 [`mini_ptq.py`](/home/jay/projs/mlir_start/legacy_python_frontend/mini_ptq.py) 更准确的定位应视为：
  - 一个“参考官方工程思路的最小近似实现”
  - 方便先理解 calibration、fake quant 和误差归因三个环节
  - 但它还不是严格意义上的 “MLIR 运行 + blob 对比” 主线实现
- 对齐官方流程时，当前阶段更合适的比较对象是三个检测头原始输出 blob：
  - `350`
  - `498`
  - `646`
- 这一步通常不需要引入检测后处理；重点是比较 raw output tensor，而不是 decode/NMS 之后的框结果。
- 后续若继续最小实现，应优先补“canonical Top MLIR 执行 + 输出 blob 对比”这一层，再决定是否继续扩展到更完整的量化流水线。

### Minimal Top Runner Landed

- 已新增本地最小 Top MLIR 执行器：
  - [`top_run.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_run.py)
- 这版执行器不依赖官方 `pymlir` runtime，而是直接解析 [`yolov5s_canonical.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s_canonical.mlir)，并用 `torch.nn.functional` 执行当前 `yolov5s` 需要的少量 `top.*`：
  - `top.Input`
  - `top.Weight`
  - `top.Conv`
  - `top.Sigmoid`
  - `top.Mul`
  - `top.Add`
  - `top.Concat`
  - `top.MaxPool`
  - `top.Interp`
  - `top.Reshape`
  - `top.Permute`
- 默认直接读取官方回归输入：
  - [`yolov5s_in_f32.npz`](/home/jay/projs/tpu-mlir/regression/regression_out/yolov5s_bm1684x_num_core_1/yolov5s_in_f32.npz)
- 默认输出本地运行结果：
  - [`experiments/03_top_run/yolov5s_top_run_outputs.npz`](/home/jay/projs/mlir_start/experiments/03_top_run/yolov5s_top_run_outputs.npz)

### Top Runner Verification

- 已在 `llama` 环境中实际执行：
  - `python legacy_python_frontend/top_run.py`
- 成功跑出三个返回 blob：
  - `350` -> `Transpose_214` -> `(1, 3, 80, 80, 85)`
  - `498` -> `Transpose_326` -> `(1, 3, 40, 40, 85)`
  - `646` -> `Transpose_438` -> `(1, 3, 20, 20, 85)`
- 使用官方 `yolov5s_top_outputs.npz` 中的三个 head `Conv` 输出经 `reshape + permute` 后进行对比，结果高度一致：
  - `350`: cosine `1.000000`, max_abs_diff `0.000031`
  - `498`: cosine `1.000000`, max_abs_diff `0.000019`
  - `646`: cosine `1.000000`, max_abs_diff `0.000018`
- 这说明当前本地 `top_run.py` 已经可以作为后续 calibration / quantization / blob 级误差归因的主执行入口。

### MLIR Native Skeleton Landed

- 根据“项目主线应逐步融入 LLVM/MLIR 体系，而不是继续扩张纯 Python”的新目标，新增了原生子工程：
  - [`mlir_native/README.md`](/home/jay/projs/mlir_start/mlir_native/README.md)
- 这版不是全量复刻 `tpu-mlir`，而是先搭出最小 MLIR 原生骨架：
  - `CMake + find_package(MLIR/LLVM)`
  - TableGen 定义的最小 `mini_top` dialect
  - C++ 实现的 op 包装与 pass
  - 一个可执行的 `mini-top-opt`

### Native Files Added

- 顶层构建：
  - [`mlir_native/CMakeLists.txt`](/home/jay/projs/mlir_start/mlir_native/CMakeLists.txt)
- Dialect / ODS：
  - [`MiniTopDialect.h`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/IR/MiniTopDialect.h)
  - [`MiniTopOps.h`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/IR/MiniTopOps.h)
  - [`MiniTopOps.td`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/IR/MiniTopOps.td)
- Pass：
  - [`Passes.h`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/Transforms/Passes.h)
  - [`Passes.td`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/Transforms/Passes.td)
  - [`FuseSiLU.cpp`](/home/jay/projs/mlir_start/mlir_native/lib/Dialect/MiniTop/FuseSiLU.cpp)
- Tool：
  - [`mini-top-opt.cpp`](/home/jay/projs/mlir_start/mlir_native/tools/mini-top-opt/mini-top-opt.cpp)
- 示例：
  - [`silu_pattern.mlir`](/home/jay/projs/mlir_start/mlir_native/examples/silu_pattern.mlir)

### Native Verification

- 使用本机 LLVM/MLIR 构建目录完成配置与编译：
  - `MLIR_DIR=/home/jay/projs/llvm-project/build/lib/cmake/mlir`
  - `LLVM_DIR=/home/jay/projs/llvm-project/build/lib/cmake/llvm`
- 已成功生成：
  - [`mini-top-opt`](/home/jay/projs/mlir_start/mlir_native/build/bin/mini-top-opt)
- 已实际运行：
  - `mini-top-opt silu_pattern.mlir --mini-top-fuse-silu`
- 结果符合预期：
  - `mini_top.sigmoid + mini_top.mul` 被原生 pass 改写为 `mini_top.silu`
  - 死掉的 `sigmoid` 也会一并删除

### Meaning

- 这一步标志着项目从“Python 原型为主”正式进入“MLIR 原生骨架已建立”的阶段。
- 后续如果继续贴近 `tpu-mlir`，更合理的路线应是：
  - 逐步把关键 pattern / op 迁入 `mlir_native`
  - 再决定是否接入更完整的 interpreter / lowering / runtime

### Project Structure Realigned

- 明确将仓库视为两个子项目：
  - 根目录 Python 原型线
  - [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native) MLIR 原生线
- 新增结构说明文档：
  - [`docs/project_structure.md`](/home/jay/projs/mlir_start/docs/project_structure.md)
- 进一步明确策略：
  - 停止继续扩张 Python runtime
  - Python 脚本只保留“参考实现 / 对照 oracle / bug 修复”角色
  - 新的编译器主线工作优先进入 `mlir_native`

### First Lowering Landed

- 在 [`mlir_native`](/home/jay/projs/mlir_start/mlir_native) 中新增第一条 lowering pass：
  - [`LowerActivations.cpp`](/home/jay/projs/mlir_start/mlir_native/lib/Dialect/MiniTop/LowerActivations.cpp)
- 当前 lowering 范围：
  - `mini_top.sigmoid -> arith + math`
  - `mini_top.mul -> arith.mulf`
  - `mini_top.silu -> arith + math`
- 这一步的定位不是全量 `mini_top -> linalg`，而是先建立一个真实的“从自定义 dialect 往更低层标准 dialect 下沉”的边界。

### Lowering Verification

- 已成功重新编译：
  - [`mini-top-opt`](/home/jay/projs/mlir_start/mlir_native/build/bin/mini-top-opt)
- 后续推荐直接使用如下命令观察两步 native pipeline：
  - `mini-top-opt silu_pattern.mlir --mini-top-fuse-silu --mini-top-lower-activations`
- 已实际运行成功，当前示例输出结果已经体现第一步 lowering：
  - `mini_top.conv` 仍然保留在 `mini_top`
  - `mini_top.silu` 已下沉为：
    - `arith.constant`
    - `arith.mulf`
    - `math.exp`
    - `arith.addf`
    - `arith.divf`
- 这说明项目已经具备：
  - 原生 rewrite
  - 原生 lowering
  - 原生 `mlir-opt` 风格工具

### More YOLOv5 Patterns Migrated To Native

- 继续把 Python 原型中的 YOLOv5 相关 pattern 迁入 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)，这次新增的是 layout 规范化相关规则。
- 在 [`MiniTopOps.td`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/IR/MiniTopOps.td) 中新增：
  - `mini_top.reshape`
  - `mini_top.permute`
- 新增原生 pass：
  - [`CanonicalizeLayout.cpp`](/home/jay/projs/mlir_start/mlir_native/lib/Dialect/MiniTop/CanonicalizeLayout.cpp)
  - pass 名称：`--mini-top-canonicalize-layout`
- 该 pass 当前迁入了两条来自 Python `top_canonicalize.py` 的稳定规则：
  - 删除 no-op `reshape`
  - 删除 identity `permute`

### Layout Canonicalize Verification

- 新增验证样例：
  - [`layout_patterns.mlir`](/home/jay/projs/mlir_start/mlir_native/examples/layout_patterns.mlir)
- 已实际运行：
  - `mini-top-opt layout_patterns.mlir --mini-top-canonicalize-layout`
- 当前输出会直接化简为：
  - `return %arg0`
- 这说明本地原生链路已经不只包含 SiLU rewrite，也开始承接 YOLOv5 中常见的“布局整理类 canonicalize”。

### Native Integration Notes

- 迁入 `reshape/permute` 时，遇到了 MLIR 新版 ODS 自动生成的 bytecode/property 接口编译问题。
- 最终通过为 [`MiniTopOps.h`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/IR/MiniTopOps.h) 补充 bytecode 相关头文件，成功让新 op 的 TableGen 代码在当前 LLVM/MLIR 版本下正常编译。
- 这次问题也进一步说明：
  - Python 原型适合快速试规则
  - 一旦进入原生链路，就必须真正理解 ODS、生成代码和 MLIR 版本接口的关系

## Working Agreement

- 本文件用于记录每日进展、问题、决定和处理结果。
- 后续在这个项目里继续推进时，默认同步更新本日志。

## 2026-04-16

### Python Archive Landed

- 按当前项目分层约定，将已停止继续扩张的 Python 原型脚本统一归档到：
  - [`legacy_python_frontend/`](/home/jay/projs/mlir_start/legacy_python_frontend)
- 本次移动的脚本包括：
  - [`model_transform.py`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.py)
  - [`top_canonicalize.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_canonicalize.py)
  - [`top_fuse.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_fuse.py)
  - [`top_run.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_run.py)
  - [`mini_ptq.py`](/home/jay/projs/mlir_start/legacy_python_frontend/mini_ptq.py)
- 新增归档说明：
  - [`legacy_python_frontend/README.md`](/home/jay/projs/mlir_start/legacy_python_frontend/README.md)

### Documentation Updated

- 已同步更新以下说明文档中的脚本入口与结构描述：
  - [`README.md`](/home/jay/projs/mlir_start/README.md)
  - [`docs/project_structure.md`](/home/jay/projs/mlir_start/docs/project_structure.md)
  - [`docs/archive/week1_archived.md`](/home/jay/projs/mlir_start/docs/archive/week1_archived.md)
  - [`legacy_python_frontend/model_transform.md`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.md)
  - [`docs/notes/day4_calibration.md`](/home/jay/projs/mlir_start/docs/notes/day4_calibration.md)
  - [`docs/notes/day5_quant_eval.md`](/home/jay/projs/mlir_start/docs/notes/day5_quant_eval.md)

### Current Convention

- 仓库顶层不再承载 Python 原型脚本入口。
- Python 线现在是“归档的参考实现 / 行为 oracle”，统一从 `legacy_python_frontend/` 进入。
- 新的编译器主线工作继续优先进入 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)。

### Direction Clarified

- 进一步明确：停止继续优化这条线，不是因为它“用了 Python”，而是因为它主要采用“解析模型后拼接 `.mlir` 文本”的方式。
- 当前更值得投入的方向，是在 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native) 中实现真正贴近 MLIR 体系的 importer、rewrite、pass 和 lowering。

### Archive Renamed And Root Docs Cleaned

- 将原来的 Python 归档线正式命名为：
  - [`legacy_python_frontend/`](/home/jay/projs/mlir_start/legacy_python_frontend)
- 这个名字用来强调两点：
  - 它是历史阶段的 Python 前端原型
  - 它不再作为当前项目主线继续扩展
- 项目根目录的 Markdown 也同步收敛，只保留：
  - [`README.md`](/home/jay/projs/mlir_start/README.md)
  - [`log.md`](/home/jay/projs/mlir_start/log.md)
- 原来的路线图与周计划文档已移入：
  - [`docs/archive/roadmap_archived.md`](/home/jay/projs/mlir_start/docs/archive/roadmap_archived.md)
  - [`docs/archive/week1_archived.md`](/home/jay/projs/mlir_start/docs/archive/week1_archived.md)

## 2026-04-16

### First IR-Object Importer Landed

- 在 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native) 主线上新增了第一版基于 MLIR IR 对象的 importer：
  - [`onnx_to_mini_top.py`](/home/jay/projs/mlir_start/mlir_native/python/onnx_to_mini_top.py)
- 这版 importer 的目标不是再拼接 `.mlir` 文本，而是：
  - 解析 ONNX
  - 使用 `mlir.ir` + `Operation.create(...)`
  - 直接在内存中构造 `mini_top` IR
- 当前支持的 `yolov5s` 子集包括：
  - `Conv`
  - `Sigmoid`
  - `Mul`
  - `Add`
  - `Concat`
  - `MaxPool`
  - `Resize`
  - `Reshape`
  - `Transpose`
  - `Shape/Gather/Unsqueeze/Cast` 常量折叠

### Native Dialect Expanded For Import

- 为了让 importer 落到真正的 `mini_top` 方言，而不是继续输出未定义的文本 op，本次在 [`MiniTopOps.td`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/IR/MiniTopOps.td) 中补充了：
  - `mini_top.weight`
  - `mini_top.add`
  - `mini_top.concat`
  - `mini_top.maxpool`
  - `mini_top.interp`
- 原生子工程已经重新编译通过：
  - [`mini-top-opt`](/home/jay/projs/mlir_start/mlir_native/build/bin/mini-top-opt)

### Environment Note

- 当前默认 `python3` 和 `llama` 环境里都还没有可直接使用的 `mlir.ir` Python bindings。
- 因此这版 importer 目前已经：
  - 完成源码实现
  - 通过 Python 语法检查
  - 与 native dialect 对齐
- 但还需要补上 LLVM/MLIR Python 绑定环境后，才能实际执行 `yolov5s.onnx -> mini_top MLIR` 的端到端验证。
