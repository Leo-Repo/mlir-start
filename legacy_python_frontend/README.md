# Legacy Python Frontend

这个目录归档了项目第一阶段的 Python 原型脚本：

- [`model_transform.py`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.py)
- [`top_canonicalize.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_canonicalize.py)
- [`top_fuse.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_fuse.py)
- [`top_run.py`](/home/jay/projs/mlir_start/legacy_python_frontend/top_run.py)
- [`mini_ptq.py`](/home/jay/projs/mlir_start/legacy_python_frontend/mini_ptq.py)
- [`model_transform.md`](/home/jay/projs/mlir_start/legacy_python_frontend/model_transform.md)

它们的定位是：

- 作为 importer / canonicalize / fusion / runner / PTQ 的参考实现
- 作为 `mlir_native/` 原生实现的行为对照和语义 oracle
- 用于保留项目早期实验路径，方便回溯和比对

需要特别说明的是，这条线停止继续优化，并不是因为它“用了 Python”，而是因为它主要采用“解析模型后拼接 MLIR 文本”的方式。项目当前更想理解的是 MLIR 原生对象、dialect、rewrite、pass 和 lowering，所以后续重点已经转向 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)。

当前约定：

- 不再继续扩张 Python runtime 能力
- 只做必要的 bug 修复、验证和文档维护
- 新的编译器主线工作优先进入 [`mlir_native/`](/home/jay/projs/mlir_start/mlir_native)
