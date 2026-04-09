# MLIR实践路线图（面向AI推理）

## 0. 目标与边界

### 项目目标

在 4~8 周内完成以下能力闭环：

1. 理解 MLIR 在 AI 推理编译中的核心价值与分层方式
2. 跑通 `yolov5s.onnx -> MLIR -> calibration -> quantization`
3. 跑通典型算子/子图 `lowering to GPU -> run on GPU`
4. 输出一版神经网络引擎编译架构草案

### 范围边界

- 初期不追求“YOLOv5 全网一次性 GPU lowering 全打通”
- 初期优先“可解释性 + 可验证性 + 复用性”
- 先小样（matmul/conv/子图）后整网

## 1. 阶段拆分

## 阶段A：编译地图与环境准备（1~2天）

### 目标

建立“AI 推理编译分层地图”，搭好实验环境与目录结构。

### 必做任务

1. 完成项目目录初始化（`experiments/`、`docs/`、`scripts/`）
2. 阅读并整理三类资料：
   - tpu-mlir 部署链路
   - upstream MLIR 教程（Toy、mlir-opt）
   - GPU dialect 文档
3. 输出 1 页“分层地图”：前端导入层、图级优化/量化层、tensor/linalg 层、bufferization/memref 层、gpu/target codegen 层、runtime 接口层

### 验收标准

- 能口头解释每一层“负责什么、不负责什么”

## 阶段B：YOLOv5 ONNX -> Top MLIR（2~4天）

### 目标

跑通模型导入并读懂关键 IR。

### 必做任务

1. 准备 `yolov5s.onnx`
2. 执行 tpu-mlir 的 `model_transform.py`
3. 生成并阅读 `.mlir`
4. 建立 ONNX 节点到 MLIR op 的映射清单

### 验收标准

- 能解释 70% 以上主干算子的映射关系
- 能说明 preprocessing 参数如何进入编译流程

## 阶段C：Calibration 与 Quantization（2~4天）

### 目标

理解“量化是编译流程的一部分”。

### 必做任务

1. 构建校准样本集（先 100 张可用集）
2. 执行 `run_calibration.py` 生成校准表
3. 执行 `model_deploy.py` 生成量化产物
4. 做 FP32/量化结果对比，记录误差

### 验收标准

- 能解释 calibration table 在 pipeline 中的位置
- 有至少 1 份误差归因记录（算子级或模块级）

## 阶段D：MLIR机制补全（4~7天）

### 目标

建立可修改编译器的基础能力（pass/rewrite/conversion）。

### 必做任务

1. 用 `mlir-opt` 观察 pass 作用前后 IR 变化
2. 完成至少一个最小 rewrite/pass 实验
3. 完成一个小型 dialect conversion 实验
4. 写一份“canonicalization vs conversion”笔记

### 验收标准

- 能独立跟踪一条 pass pipeline
- 能解释 legality/conversion target 的基本思想

## 阶段E：GPU lowering 小样（4~7天）

### 目标

走通最小 GPU 编译链路，不直接整网。

### 必做任务

1. 从 `matmul` 或 `conv+bias+act` 开始
2. 观察 `linalg -> bufferization -> gpu` 关键步骤
3. 产出可运行 GPU demo（可先单算子）
4. 记录 kernel launch 与数据搬运边界

### 验收标准

- 有可重复运行的脚本
- 能画出 lowering 关键节点图

## 阶段F：YOLOv5 子图与引擎设计沉淀（5~10天）

### 目标

把实践沉淀成神经网络引擎设计文档。

### 必做任务

1. 选择 YOLOv5 关键子图（先 backbone block）
2. 完成子图 GPU 运行与精度/性能记录
3. 输出引擎设计草案（见 `docs/engine_design_template.md`）

### 验收标准

- 给出“多层IR + pass pipeline + runtime接口”初版方案
- 明确可复用 pass 与 target-specific pass 的分界
