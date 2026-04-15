# MLIR Start: AI推理编译实践项目

本项目以实践为主，面向有边缘推理与部署经验的工程师，目标是快速理解 MLIR 在 AI 推理中的作用，并沉淀为可复用的神经网络引擎设计方法。

当前主目标已经从“先把 Python 铺满”切换成“两线并行，但以 MLIR 原生线为主”。

当前阶段目标：

1. `yolov5s.onnx -> MLIR`
2. `calibration / quantization`
3. `lowering to GPU`
4. `run on GPU`

## 项目定位

这个项目当前按两条实现线推进：

1. **Python 原型线**  
   用较快速度验证 importer / canonicalize / fusion / runner 的语义。
2. **MLIR 原生线**  
   通过 `.td + C++ + CMake + pass/lowering` 真正进入 LLVM/MLIR 体系。

当前原则：

- Python 线冻结为参考实现，停止继续扩张 runtime 功能
- 新的编译器主线工作优先进入 `mlir_native/`

## 文档导航

- 项目结构说明: `docs/project_structure.md`
- 路线图与阶段目标: `docs/roadmap.md`
- 第一周任务清单: `docs/week1.md`
- YOLOv5 初期实操流程: `docs/yolov5_pipeline.md`
- 引擎设计模板（后续填写）: `docs/engine_design_template.md`

## 建议目录结构

```text
mlir_start/
  models/                     # 模型与校准样本
  docs/                       # 规划与实验文档
  experiments/
    tpu_mlir/                 # ONNX->MLIR->量化部署流水线
    upstream_mlir/            # pass/conversion/lowering 小实验
    yolov5_gpu/               # YOLOv5 子图/关键算子的GPU链路
  scripts/                    # 一键化脚本（导出、校准、运行、对比）
```

## 当前代码主线

1. 根目录 Python 脚本  
   参考实现，用于理解模型转换、局部推理与结果对照。
2. `mlir_native/`  
   当前主线子工程，用于承载原生 dialect、pass、lowering 与工具。

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

## 参考资料

- [sophgo/tpu-mlir](https://github.com/sophgo/tpu-mlir)
- [OpenMLIR/mlir-tutorial](https://github.com/OpenMLIR/mlir-tutorial)
- [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [MLIR mlir-opt Tutorial](https://mlir.llvm.org/docs/Tutorials/MlirOpt/)
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [知乎文章（你提供）](https://zhuanlan.zhihu.com/p/1924384457349132481)
