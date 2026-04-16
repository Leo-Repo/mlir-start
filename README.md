# MLIR Start: AI推理编译实践项目

本项目以实践为主，面向有边缘推理与部署经验的工程师，目标是快速理解 MLIR 在 AI 推理中的作用，并沉淀为可复用的神经网络引擎设计方法。

当前项目已经**全面转向 `mlir_native/` 主线**。早期的 Python importer / canonicalize / fusion / runner / PTQ 脚手架不再继续扩功能，只保留为归档参考与行为对照。

## 当前主线目标

当前阶段继续围绕这四件事推进：

1. `yolov5s.onnx -> MLIR`
2. `calibration / quantization`
3. `lowering to GPU`
4. `run on GPU`

## 项目定位

这个项目现在明确分成两条线，但只有一条继续演进：

1. **`mlir_native/` 原生主线**  
   通过 `.td + C++ + CMake + pass/lowering` 真正进入 LLVM/MLIR 体系。
2. **`legacy_python_frontend/` 归档参考线**  
   保留早期 Python 文本型 importer 与实验脚手架，只作为对照 oracle 和历史记录。

当前原则：

- 不再继续优化 `legacy_python_frontend/` 的能力边界
- 新的编译器工作优先进入 `mlir_native/`
- 归档线只接受必要的 bug 修复、验证和文档维护

## 文档导航

- 项目结构说明: `docs/project_structure.md`
- 归档计划文档: `docs/archive/`
- YOLOv5 初期实操流程: `docs/yolov5_pipeline.md`
- 引擎设计模板（后续填写）: `docs/engine_design_template.md`

## 建议目录结构

```text
mlir_start/
  legacy_python_frontend/     # 已归档的 Python 文本前端与实验脚手架
  mlir_native/                # 当前编译器主线
  models/                     # 模型与校准样本
  docs/                       # 规划与实验文档
    archive/                  # 已归档的阶段计划与周任务
  experiments/
    tpu_mlir/                 # ONNX->MLIR->量化部署流水线
    upstream_mlir/            # pass/conversion/lowering 小实验
    yolov5_gpu/               # YOLOv5 子图/关键算子的GPU链路
```

## 为什么停止继续优化 Python 文本前端

这里要区分两种“Python 前端”：

1. `tpu-mlir` 的前端  
   虽然很多逻辑写在 Python 里，但它通过 `mlir.ir` 和 dialect bindings 直接构造 MLIR IR 对象。
2. 我们早期的 Python 原型  
   主要是解析 ONNX、维护 SSA、再拼接 `.mlir` 文本，本质上是文本级前端原型。

这条早期路线依然有学习价值，但继续投入的收益已经明显下降，因为它绕开了我们真正想理解的东西：

- TableGen
- dialect / op 定义
- PatternRewrite
- pass pipeline
- lowering 到更低层 dialect
- 与 `mlir-opt` 风格工具的原生集成

所以当前策略是：

- 保留 [`legacy_python_frontend/README.md`](/home/jay/projs/mlir_start/legacy_python_frontend/README.md) 及其中脚本作为参考
- 把稳定规则逐步迁进 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)
- 后续重点投入原生 importer、canonicalize、fusion、lowering

## 当前进度

已经完成的关键阶段：

1. 早期参考线  
   跑通了 `yolov5s.onnx -> Top MLIR`，并实现了本地的 canonicalize、fusion、Top runner、最小 PTQ 实验。
2. 原生主线  
   已建立最小 `mini_top` dialect、`mini-top-opt`、第一条原生 rewrite，以及第一步 lowering。
3. 最近迁移  
   已把部分 YOLOv5 相关规则从 Python 迁到 native，包括：
   - `Conv + Sigmoid + Mul -> SiLU`
   - 去掉 no-op `reshape`
   - 去掉 identity `permute`

## 接下来做什么

下一阶段的主线应集中在 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)：

1. 继续扩展 `mini_top` dialect 和 importer 子集
2. 增加更多 YOLOv5 相关 rewrite / canonicalize
3. 推进 `mini_top -> 更低层 dialect` 的 lowering
4. 逐步建立真正贴近 `tpu-mlir` 分层的 native pipeline

## 里程碑（简版）

1. **M1: ONNX 到 Top MLIR 打通**  
   产出可读 `.mlir`，完成 ONNX 节点到 MLIR op 映射笔记。
2. **M2: Calibration/Quantization 打通**  
   产出校准表与量化产物，完成精度对比与误差归因记录。
3. **M3: 原生 lowering 小样打通**  
   先在 `mlir_native/` 中完成 `mini_top -> 更低层 dialect` 的第一批 lowering。
4. **M4: YOLOv5 子图到 GPU 运行**  
   完成关键子图 GPU 运行与性能/精度记录。
5. **M5: 引擎设计文档定稿**  
   形成“多层IR + pass pipeline + runtime 接口”设计草案。

## 归档材料

以下内容已归档，不再作为根目录主入口维护：

- Python 文本前端与实验脚手架：
  - [`legacy_python_frontend/`](/home/jay/projs/mlir_start/legacy_python_frontend)
- 第一周计划与路线图：
  - [`docs/archive/week1_archived.md`](/home/jay/projs/mlir_start/docs/archive/week1_archived.md)
  - [`docs/archive/roadmap_archived.md`](/home/jay/projs/mlir_start/docs/archive/roadmap_archived.md)

## 参考资料

- [sophgo/tpu-mlir](https://github.com/sophgo/tpu-mlir)
- [OpenMLIR/mlir-tutorial](https://github.com/OpenMLIR/mlir-tutorial)
- [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [MLIR mlir-opt Tutorial](https://mlir.llvm.org/docs/Tutorials/MlirOpt/)
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [知乎文章（你提供）](https://zhuanlan.zhihu.com/p/1924384457349132481)
