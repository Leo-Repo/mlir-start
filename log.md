# Project Log

## Working Agreement

- 本文件用于记录每日进展、问题、决定和处理结果。
- 后续在这个项目里继续推进时，默认同步更新本日志。

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

### Key Decisions

- 以官方 `yolov5s` 示例的三个 head 输出为子图截断点，而不是把 detect/postprocess 整段一起导入。
- 将 shape 相关辅助子图视为 importer 阶段的常量计算，而不是保留为真正的运行期图语义。
- 把 `model_transform.py` 的定位明确为“学习型前端 importer”，不是完整工业实现。

### Problems And Resolution

- 多次遇到 shape 子图相关报错，例如 `Unsupported ONNX op Shape`、`Expected constant tensor, got dynamic value: 348`、`Unsupported ONNX op Range`、`Unsupported ONNX op ConstantOfShape`。
- 子图顺序最初不是稳定的拓扑序，出现了“值在定义前被引用”的问题。
- 通过逐步查看 ONNX 局部子图上下文，沿报错链精确补齐常量折叠逻辑，并将子图选择改为“从目标输出反向 DFS + 后序追加”，保证拓扑顺序稳定。
- loc 风格最终调整为更接近 `tpu-mlir` 官方形式：
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

### Output Chains Identified

- `350 = Conv_196 -> Reshape_213 -> Transpose_214`
- `498 = Conv_308 -> Reshape_325 -> Transpose_326`
- `646 = Conv_420 -> Reshape_437 -> Transpose_438`

### Notes

- 解释了为什么官方示例选择 `350 / 498 / 646` 作为输出，而不是直接停在 `Conv_196 / 308 / 420`。
- 解释了为什么 `Reshape + Transpose` 更适合作为“去掉真正后处理后的标准输出”。
- 解释了为什么这里做的是静态 shape 推导，而不是动态推导。
- `top_canonicalize.py` 当前支持属性清洗、`MaxPool/Interp/Concat/Reshape/Permute` 的静态 shape 重推导，以及删除 no-op `Reshape` 和 identity `Permute`。

## 2026-04-15

### Fusion Planning And Prototype

- 明确开始规划本地版 YOLOv5 fusion 路线，而不是继续把所有变化都堆到 canonicalize 中。
- 重新检查 [`experiments/01_onnx_to_mlir/yolov5s.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s.mlir) 中最典型的激活模式，确认大量重复出现：
  - `top.Conv`
  - `top.Sigmoid`
  - `top.Mul`
- 明确这组模式对应 YOLOv5 中的 SiLU 结构，是最适合做第一条本地 fusion 规则的切入点。
- 已新增第一版 [`top_fuse.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_fuse.py)。
- 当前规则只做一条保守融合：
  - `top.Conv -> top.Sigmoid -> top.Mul`
  - 最终改写为更接近官方 `tpu-mlir` 的形式：`top.Conv -> top.SiLU`
- 当前在 [`experiments/01_onnx_to_mlir/yolov5s_fused.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s_fused.mlir) 中已成功融合 `57` 处 SiLU pattern。
- 为 [`top_fuse.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_fuse.py) 增加了：
  - `--summary-only`
  - `--dump-patterns`

## 2026-04-16

### Minimal PTQ And Top Runner

- 因为本机 `tpu-mlir` 运行时存在 Python `3.10` / `3.11` ABI 不匹配，官方 `run_calibration.py` / `model_deploy.py` 在当前 `llama` 环境下无法直接运行。
- 因此改为参考官方工程分层，自行实现最小版 PTQ 流程：
  - [`mini_ptq.py`](/home/jay/projs/mlir_start/legacy_python_frontend/mini_ptq.py)
- 该脚本基于本机已有的 [`/home/jay/projs/yolov5`](/home/jay/projs/yolov5) 和 `yolov5s.pt`，实现了 calibration、fake quantization 和 layer-wise error attribution。
- 已新增本地最小 Top MLIR 执行器：
  - [`top_run.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_run.py)
- 这版执行器不依赖官方 `pymlir` runtime，而是直接解析 [`yolov5s_canonical.mlir`](/home/jay/projs/mlir_start/experiments/01_onnx_to_mlir/yolov5s_canonical.mlir)，并用 `torch.nn.functional` 执行当前 `yolov5s` 需要的少量 `top.*`。
- 使用官方 `yolov5s_top_outputs.npz` 中的三个 head `Conv` 输出经 `reshape + permute` 后进行对比，结果高度一致：
  - `350`: cosine `1.000000`, max_abs_diff `0.000031`
  - `498`: cosine `1.000000`, max_abs_diff `0.000019`
  - `646`: cosine `1.000000`, max_abs_diff `0.000018`

### MLIR Native Skeleton And Native Passes

- 根据“项目主线应逐步融入 LLVM/MLIR 体系，而不是继续扩张纯 Python”的新目标，新增了原生子工程：
  - [`mlir_native/README.md`](/home/jay/projs/mlir_start/mlir_native/README.md)
- 这版先搭出了最小 MLIR 原生骨架：
  - `CMake + find_package(MLIR/LLVM)`
  - TableGen 定义的最小 `mini_top` dialect
  - C++ 实现的 op 包装与 pass
  - 一个可执行的 `mini-top-opt`
- 使用本机 LLVM/MLIR 构建目录完成配置与编译，并成功生成：
  - [`mini-top-opt`](/home/jay/projs/mlir_start/mlir_native/build/bin/mini-top-opt)
- 已实际运行：
  - `mini-top-opt silu_pattern.mlir --mini-top-fuse-silu`
- 结果符合预期：
  - `mini_top.sigmoid + mini_top.mul` 被原生 pass 改写为 `mini_top.silu`
- 在 [`mlir_native`](/home/jay/projs/mlir_start/mlir_native) 中新增第一条 lowering pass：
  - [`LowerActivations.cpp`](/home/jay/projs/mlir_start/mlir_native/lib/Dialect/MiniTop/LowerActivations.cpp)
- 新增原生 layout canonicalize pass：
  - [`CanonicalizeLayout.cpp`](/home/jay/projs/mlir_start/mlir_native/lib/Dialect/MiniTop/CanonicalizeLayout.cpp)

### Project Structure Realigned And Archive

- 明确将仓库视为两个子项目：
  - 根目录 Python 原型线
  - [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native) MLIR 原生线
- 新增结构说明文档：
  - [`docs/project_structure.md`](/home/jay/projs/mlir_start/docs/project_structure.md)
- 进一步明确策略：
  - 停止继续扩张 Python runtime
  - Python 脚本只保留“参考实现 / 对照 oracle / bug 修复”角色
  - 新的编译器主线工作优先进入 `mlir_native`
- 按当前项目分层约定，将已停止继续扩张的 Python 原型脚本统一归档到：
  - [`legacy_python_frontend/`](/home/jay/projs/mlir_start/legacy_python_frontend)
- 新增归档说明：
  - [`legacy_python_frontend/README.md`](/home/jay/projs/mlir_start/legacy_python_frontend/README.md)
- 项目根目录的 Markdown 也同步收敛，只保留：
  - [`README.md`](/home/jay/projs/mlir_start/README.md)
  - [`log.md`](/home/jay/projs/mlir_start/log.md)
- 原来的路线图与周计划文档已移入：
  - [`docs/archive/roadmap_archived.md`](/home/jay/projs/mlir_start/docs/archive/roadmap_archived.md)
  - [`docs/archive/week1_archived.md`](/home/jay/projs/mlir_start/docs/archive/week1_archived.md)

### First IR-Object Importer Landed

- 在 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native) 主线上新增了第一版基于 MLIR IR 对象的 importer：
  - [`onnx_to_mini_top.py`](/home/jay/projs/mlir_start/mlir_native/python/onnx_to_mini_top.py)
- 这版 importer 的目标不是再拼接 `.mlir` 文本，而是：
  - 解析 ONNX
  - 使用 `mlir.ir` + `Operation.create(...)`
  - 直接在内存中构造 `mini_top` IR
- 为了让 importer 落到真正的 `mini_top` 方言，本次在 [`MiniTopOps.td`](/home/jay/projs/mlir_start/mlir_native/include/mlir_start/Dialect/MiniTop/IR/MiniTopOps.td) 中补充了：
  - `mini_top.weight`
  - `mini_top.add`
  - `mini_top.concat`
  - `mini_top.maxpool`
  - `mini_top.interp`
- 这版 importer 当前已经：
  - 完成源码实现
  - 通过 Python 语法检查
  - 与 native dialect 对齐
- 但当时还需要补上 LLVM/MLIR Python 绑定环境后，才能实际执行 `yolov5s.onnx -> mini_top MLIR` 的端到端验证。

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


## 2026-04-19

### Native Importer Bring-Up

- 验证了 LLVM `build-py` 产出的 MLIR Python bindings 已经可用，但在 `llama` conda 环境下直接运行 importer 时失败。
- 进一步定位到失败并不是 `mlir.ir` 缺包，而是底层运行时库冲突：
  - [`libMLIRPythonCAPI.so`](/home/jay/projs/llvm-project/build-py/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/libMLIRPythonCAPI.so.22.0git)
  - 依赖的 `GLIBCXX_3.4.30`
  - 不存在于 [`/home/jay/miniconda3/envs/llama/lib/libstdc++.so.6`](/home/jay/miniconda3/envs/llama/lib/libstdc++.so.6)
- 使用以下方式临时绕过运行时冲突，成功在 `llama` 中加载 MLIR bindings：
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
  - `PYTHONPATH=/home/jay/projs/llvm-project/build-py/tools/mlir/python_packages/mlir_core:$PYTHONPATH`
- 在 native importer [`mlir_native/python/onnx_to_mini_top.py`](/home/jay/projs/mlir_start/mlir_native/python/onnx_to_mini_top.py) 中修复了两个关键问题：
  - `ensure_operand()` 现在会在需要时把可常量折叠的 shape 值物化成 `mini_top.weight`
  - 新增 `--input-shape`，默认将 `yolov5s` 输入静态化为 `1,3,640,640`
- 修复后首次成功跑通基于 MLIR IR 对象的 native importer，生成：
  - [`experiments/04_mini_top_import/yolov5s_mini_top.mlir`](/home/jay/projs/mlir_start/experiments/04_mini_top_import/yolov5s_mini_top.mlir)
  - [`experiments/04_mini_top_import/yolov5s_mini_top_weights.npz`](/home/jay/projs/mlir_start/experiments/04_mini_top_import/yolov5s_mini_top_weights.npz)

### Problems And Resolution

- importer 初始报错表面上像是 “requires MLIR Python bindings”，但真实原因是 conda 环境内 `libstdc++` 版本偏旧，导致 `libMLIRPythonCAPI.so` 在导入时装载失败。
- 在真正进入图转换后，又遇到：
  - `KeyError: 'Unknown operand: 339'`
- 进一步定位发现 `339` 属于 shape 常量子图，根因是输入 `images` 仍然保留动态维度，导致中间 `Conv` 输出 shape 变成 `[-1, 255, 0, 0]`，无法继续做常量折叠。
- 用 `runpy`/单步导入方式抓到了 importer 内部的真实底层异常，而不是停留在泛化错误提示上。
- 确认 `339` 来源于 `Unsqueeze_207`，再向前追踪到 `Shape_197 -> Gather_199` 链。
- 将 graph input `images` 的 shape 覆盖为 `1x3x640x640` 后，`350 / 498 / 646` 这三路所依赖的 shape 子图全部恢复为可常量化状态，native importer 最终跑通。

## 2026-04-20

### Environment Wrappers And Error Reporting

- 为 native importer [`mlir_native/python/onnx_to_mini_top.py`](/home/jay/projs/mlir_start/mlir_native/python/onnx_to_mini_top.py) 补充了更精确的 MLIR 环境报错信息。
- 当底层异常涉及 `GLIBCXX_*` 或 `libstdc++.so.6` 时，错误提示会明确指出是 `libstdc++` 运行时冲突，而不是笼统提示“缺少 MLIR Python bindings”。
- 新增通用环境脚本：
  - [`mlir_native/scripts/with_mlir_python_env.sh`](/home/jay/projs/mlir_start/mlir_native/scripts/with_mlir_python_env.sh)
- 新增 importer 启动脚本：
  - [`mlir_native/scripts/run_onnx_to_mini_top.sh`](/home/jay/projs/mlir_start/mlir_native/scripts/run_onnx_to_mini_top.sh)
- 在 [`mlir_native/README.md`](/home/jay/projs/mlir_start/mlir_native/README.md) 中补充了：
  - 如何通过 wrapper 固化环境
  - 调试器如何继承同样的环境
  - 如何避免 debug 时再次出现绑定导入失败

### Follow-up

- `llama` 环境后续已经补齐 `GLIBCXX_3.4.30`，因此撤掉了之前为临时兼容加入的 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 方案。
- 现在 [`mlir_native/scripts/with_mlir_python_env.sh`](/home/jay/projs/mlir_start/mlir_native/scripts/with_mlir_python_env.sh) 只负责固定 `PYTHONPATH`。
- importer 的错误提示也同步调整为：如果再次遇到 `GLIBCXX_*` / `libstdc++.so.6`，优先提示修复当前环境自身的 C++ runtime，而不是建议继续走 preload 兼容。
- 调试器现在只需要继承同样的 Python 环境与 `PYTHONPATH`，不再额外依赖 `LD_PRELOAD`。

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
- 每次更新日志文件时按照时间顺序组织结构，本节始终位于最后。