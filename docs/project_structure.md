# 项目结构梳理

## 当前判断

这个仓库现在实际上已经分成两条实现线，也可以理解成两个子项目：

1. Python 原型线  
2. MLIR 原生线

这两条线的目标不再相同，后续也不应该继续混着扩。

## 子项目 1：Python 原型线

位于仓库根目录的一组脚本：

- [`model_transform.py`](/home/jay/projs/mlir_start/model_transform.py)
- [`top_canonicalize.py`](/home/jay/projs/mlir_start/top_canonicalize.py)
- [`top_fuse.py`](/home/jay/projs/mlir_start/top_fuse.py)
- [`top_run.py`](/home/jay/projs/mlir_start/top_run.py)
- [`mini_ptq.py`](/home/jay/projs/mlir_start/mini_ptq.py)

它们的定位是：

- 快速验证 importer / canonicalize / fusion / runner 的思路
- 作为后续原生实现的语义参考
- 作为对照 oracle，帮助检查原生 pass 是否做对

后续策略：

- 停止继续扩展 Python runtime 的功能范围
- 只保留：
  - bug 修复
  - 对照验证
  - 文档补充

## 子项目 2：MLIR 原生线

位于：

- [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)

它的定位是：

- 真正融入 LLVM/MLIR 体系
- 使用 `.td + C++ + CMake`
- 在原生 pass、rewrite、lowering 中学习 `tpu-mlir` 的分层思想

当前已经具备：

- 最小 `mini_top` dialect
- TableGen 定义的 op
- 原生 `mini-top-opt`
- 第一条原生 rewrite：
  - `mini_top.sigmoid + mini_top.mul -> mini_top.silu`
- 第一条原生 lowering：
  - `mini_top.sigmoid / mini_top.mul / mini_top.silu`
  - 下沉到 `arith` + `math`

## 为什么要这样拆

如果继续只在 Python 里实现：

- 可以验证算法
- 但会绕开 MLIR 最关键的工程层

例如：

- TableGen
- Dialect 注册
- PatternRewrite
- Pass 管线
- Lowering 到更低层 dialect

而这些正是理解 `tpu-mlir` 和 MLIR 编译器架构的核心。

## 建议主线

从现在开始，项目主线应切换为：

1. Python 线冻结为参考实现
2. 新功能优先落到 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)
3. 优先扩展：
   - rewrite
   - lowering
   - dialect 设计
4. runtime 相关工作暂时不继续扩张 Python 版本

## 当前目录建议理解方式

```text
mlir_start/
  *.py                    # Python 原型与参考实现
  mlir_native/            # 原生 MLIR/LLVM 子工程
  experiments/            # 产物与实验输出
  docs/                   # 文档与阶段笔记
  models/                 # 模型输入
  data/                   # 校准/测试图片
```

## 下一步重点

后续最值得继续做的，应集中在 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)：

- 增加更多 `mini_top` op
- 补更贴近 YOLOv5 的 pass
- 继续做 `mini_top -> 更低层 dialect` 的 lowering
- 逐步建立真正的 MLIR 原生执行/转换链
