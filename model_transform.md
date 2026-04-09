# `model_transform.py` 设计与实现说明

## 1. 文档目的

本文档说明本项目中的 [`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py) 的设计架构、实现原理、当前支持范围与后续扩展方向。

这个脚本的目标不是调用 `tpu-mlir` 官方 Python 包，而是独立实现一个最小可用的 `ONNX -> Top MLIR` 前端 importer，用于：

- 跑通 `yolov5s.onnx -> Top MLIR`
- 学习 ONNX 前端如何映射到 Top dialect
- 支撑后续的 IR 阅读、节点映射、算子扩展与实验记录

## 2. 脚本定位

[`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py) 是一个学习型、可运行的 ONNX 前端原型。它完成如下工作：

- 读取 ONNX 模型
- 收集输入、输出、`initializer` 与中间 `value_info`
- 仅保留目标输出依赖的最小子图
- 将 ONNX 节点映射为 `top.*` MLIR 文本
- 将权重保存为 `.npz`
- 为生成的 MLIR 补充 `loc` 表，保留 ONNX 节点名

它不负责：

- 调用 `tpu-mlir` 官方 importer
- 执行 Top 后续优化 pass
- 量化、部署、推理验证
- 完整支持所有 ONNX 算子

## 3. 总体架构

脚本可以分为四层。

### 3.1 入口与参数层

主要函数：

- `parse_args()`
- `main()`

职责：

- 读取命令行参数
- 设定默认模型路径、输出路径、输出节点、输入 shape、预处理参数
- 创建输出目录
- 驱动 importer 执行并写出产物

这一层负责“怎么运行脚本”，不负责图转换细节。

### 3.2 MLIR 文本构造层

主要类：

- `MlirBuilder`

职责：

- 分配 SSA 名称，例如 `%1`、`%2`
- 发射 `top.Input`、`top.Weight`、`top.Conv` 等 MLIR 文本
- 维护 `weight_map`，最终保存为 `.npz`
- 维护 `#locN = loc("...")` 形式的 loc 表

这一层不理解 ONNX 图的业务语义，它只负责把已经决定好的 op、属性和结果类型写成合法的 Top MLIR 文本。

### 3.3 ONNX 图分析与控制层

主要类：

- `OnnxToTopImporter`

职责：

- 加载 ONNX 模型并尝试做 shape inference
- 读取 `graph.input`、`graph.output`、`graph.initializer`、`graph.value_info`
- 从目标输出反向追踪依赖，只保留必需子图
- 对子图做拓扑排序
- 将节点分发到对应的 `emit_*()` 函数

这一层是真正的 importer 控制中心。

### 3.4 算子映射层

主要函数示例：

- `emit_conv()`
- `emit_sigmoid()`
- `emit_binary()`
- `emit_concat()`
- `emit_maxpool()`
- `emit_resize()`
- `emit_reshape()`
- `emit_transpose()`
- `emit_slice()`

职责：

- 定义 ONNX 节点如何映射为 `top.*` op
- 组装属性
- 处理权重、输入和结果类型
- 在需要时触发常量折叠

这一层决定“某个 ONNX 算子最终长成什么 Top MLIR”。

## 4. 核心实现原理

### 4.1 只转换目标输出依赖的子图

YOLOv5 导出的 ONNX 通常包含较长的后处理和 shape 构图链。  
如果直接遍历全图，会引入大量与当前 Top MLIR 目标无关的节点。

因此脚本采用“从目标输出反向追踪依赖”的方式。

主要逻辑在：

- `output_names()`
- `selected_nodes()`

做法是：

1. 读取用户指定的输出节点，例如 `350,498,646`
2. 从这几个输出反向找到生产者节点
3. 递归继续追踪这些节点的输入来源
4. 最终得到一个最小必需子图
5. 以拓扑顺序输出这些节点

这样做的好处：

- 避开无关后处理子图
- 更接近 `tpu-mlir` 官方 `yolov5s` 示例
- 明显降低 importer 需要支持的算子数量

### 4.2 优先做常量折叠

YOLOv5 的一部分节点并不执行真正的特征计算，而是在构造 shape tensor 或网格常量。  
例如：

- `Shape`
- `Gather`
- `Unsqueeze`
- `Range`
- `Expand`
- `ConstantOfShape`

这些节点如果完整保留为动态图语义，会让 importer 复杂度迅速上升。  
因此脚本优先选择在 Python 里直接求值，再将结果物化为 `top.Weight`。

相关函数包括：

- `constant_array()`
- `is_constant_value()`
- `materialize_const_output()`
- `emit_shape()`
- `emit_gather()`
- `emit_unsqueeze()`
- `emit_cast()`
- `emit_range()`
- `emit_expand()`
- `emit_constant_of_shape()`

这个策略的结果是：

- shape 子图被压缩为常量
- importer 能更快跑通 YOLOv5
- 生成的 MLIR 更适合做结构阅读

### 4.3 将普通数据算子映射为 Top MLIR

对于真正参与特征计算的节点，脚本会将其转成 `top.*` op。

例如：

- `Conv -> top.Conv`
- `Sigmoid -> top.Sigmoid`
- `Mul -> top.Mul`
- `Add -> top.Add`
- `Concat -> top.Concat`
- `MaxPool -> top.MaxPool`
- `Resize -> top.Interp`
- `Reshape -> top.Reshape`
- `Transpose -> top.Permute`
- `Slice -> top.Slice`

在这一步里，脚本会：

- 把输入张量转成 SSA 引用
- 处理权重与 bias
- 拼出 Top op 属性
- 根据 ONNX shape inference 结果写出结果 tensor type

### 4.4 文本发射与图语义分离

本脚本刻意把“图分析/语义”和“文本发射”分离。

具体来说：

- `OnnxToTopImporter` 负责理解 ONNX 图
- `MlirBuilder` 负责输出 MLIR 文本

这么做的好处：

- 代码结构更清晰
- 新增算子时，只需要扩展 `emit_*()`
- 后续如果想改输出风格，不必重写图分析逻辑

## 5. 位置元信息设计

为了让输出的 MLIR 更可读，脚本为大多数 op 和权重生成了 loc 表。

相关逻辑：

- `MlirBuilder.loc_ref()`
- `MlirBuilder.loc_definitions()`
- `OnnxToTopImporter.node_loc_name()`

输出形式类似：

```mlir
%4 = "top.Conv"(%1, %2, %3) ... loc(#loc4)
...
#loc4 = loc("Conv_0")
```

loc 表的来源规则：

- 优先使用 `node.name`
- 如果没有 `node.name`，退回 `node.output[0]`
- `Input` 使用输入名
- `Weight` 使用 initializer 名或常量名

这样做的好处：

- 更容易把 MLIR op 与 ONNX 节点对上
- 有利于 Day 3 的节点映射笔记
- 更接近官方 `tpu-mlir` 的可读风格

## 6. 主要数据结构

### 6.1 `ValueRef`

字段：

- `name`: MLIR 中的 SSA 名，例如 `%23`
- `type_str`: MLIR tensor type 文本
- `shape`: 推断出的 shape
- `dtype`: numpy dtype
- `onnx_name`: 原始 ONNX value 名

用途：

- 表示一个已经进入 MLIR 世界的值
- 用作 builder 和 importer 之间的桥梁

### 6.2 `value_info`

保存 ONNX 中每个 value 的：

- shape
- dtype

来源包括：

- `graph.input`
- `graph.output`
- `graph.value_info`
- `initializer`

用途：

- 为 MLIR 结果类型生成 tensor type
- 支撑 shape 子图折叠

### 6.3 `const_tensors`

保存当前已知的常量张量。

来源包括：

- ONNX `initializer`
- `Constant` 节点
- 被常量折叠计算出来的中间结果

用途：

- 判断某个节点能否在 importer 阶段直接求值
- 为后续 `Shape/Gather/Expand/...` 提供输入

### 6.4 `weight_map`

由 `MlirBuilder` 维护。

用途：

- 收集所有需要输出到 `.npz` 的权重或常量
- 最终通过 `np.savez()` 保存

## 7. 当前支持的算子范围

当前脚本支持两大类节点。

### 7.1 普通数据计算节点

- `Conv`
- `Sigmoid`
- `Mul`
- `Add`
- `Concat`
- `MaxPool`
- `Resize`
- `Reshape`
- `Transpose`
- `Slice`
- `Identity`
- `Constant`

### 7.2 shape / grid / 常量构图节点

- `Shape`
- `Gather`
- `Unsqueeze`
- `Cast`
- `Range`
- `Expand`
- `ConstantOfShape`

其中部分算子在“输入全为常量”时会直接折叠，例如：

- `Sigmoid`
- `Mul`
- `Add`
- `Concat`
- `Reshape`
- `Transpose`
- `Slice`

## 8. 当前设计上的取舍

这个脚本是一个“面向学习和小步扩展”的前端，而不是一个完整工业实现。

主要取舍如下。

### 8.1 优先可读和可跑通

目标是先跑通 YOLOv5 的 Top MLIR 生成，而不是一开始就完全复刻 `tpu-mlir` 全量行为。

### 8.2 优先静态 shape 子图折叠

很多 shape 相关节点没有保留成动态图执行，而是直接变成常量。  
这样会牺牲一部分通用性，但能显著简化实现。

### 8.3 只关心目标输出依赖图

脚本不尝试处理全图，而是只处理对目标输出真正有贡献的节点。  
这非常适合 `yolov5s` 当前阶段的 Top MLIR 学习任务。

### 8.4 文本兼容优先于执行兼容

当前重点是生成结构合理、可读、可比对的 Top MLIR 文本。  
它与官方输出风格接近，但并不承诺 100% 等价。

## 9. 当前产物

执行脚本后，当前会输出：

- `yolov5s.mlir`
- `yolov5s_top_f32_all_origin_weight.npz`

其中：

- `.mlir` 用于阅读图结构和做节点映射
- `.npz` 保存权重与常量张量

## 10. 后续可扩展方向

后续可以继续增强的方向包括：

- 支持更多 ONNX 算子，例如 `Sub`、`Pow`、`Equal`、`Where`
- 更完整地模拟官方 Top dialect 属性风格
- 增加 `ONNX node -> MLIR op` 对照表输出
- 增加调试开关，打印子图裁剪结果
- 增加与官方 `yolov5s_origin.mlir` 的差异对比工具
- 增加 shape 推断失败时的容错策略

## 11. 一句话总结

[`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py) 的本质是：

一个围绕 `yolov5s.onnx` 场景打造的、独立实现的最小 ONNX 前端 importer。

它通过：

- 目标输出裁剪
- 常量折叠
- ONNX 到 `top.*` 的显式映射
- 独立的 MLIR 文本构造

完成了 `ONNX -> Top MLIR` 的第一版可运行闭环，并为后续 IR 阅读、节点映射和算子扩展打下了基础。
