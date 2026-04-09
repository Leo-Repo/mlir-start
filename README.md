# MLIR Start: AI推理编译实践项目

本项目以实践为主，面向有边缘推理与部署经验的工程师，目标是快速理解 MLIR 在 AI 推理中的作用，并沉淀为可复用的神经网络引擎设计方法。

当前初期主目标：

1. `yolov5s.onnx -> MLIR`
2. `calibration / quantization`
3. `lowering to GPU`
4. `run on GPU`

## 项目定位

这个项目按两条主线并行推进：

1. **部署编译流水线（tpu-mlir）**  
   学会模型导入、图级IR、校准、量化、部署验证。
2. **编译器机制与GPU lowering（upstream MLIR）**  
   学会 dialect / pass / conversion / bufferization / GPU dialect 的分层机制。

这样做的目的不是只跑通一个 demo，而是形成“可设计神经网络引擎”的能力。

## 文档导航

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

## 里程碑（简版）

1. **M1: ONNX 到 Top MLIR 打通**  
   产出可读 `.mlir`，完成 ONNX 节点到 MLIR op 映射笔记。
2. **M2: Calibration/Quantization 打通**  
   产出校准表与量化产物，完成精度对比与误差归因记录。
3. **M3: GPU lowering 小样打通**  
   从 matmul/conv 小样走通 `linalg -> bufferization -> gpu`。
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
