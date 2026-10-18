# Score AIV 的 G6 一步：动态源码准备

2026-10-18。基线为正式动态源码 `66882634`。本目录提取长档 G6、balanced、1024 候选块的三个可读 helper，供后续跨请求 Score/Attention 交错使用；尚未接入 schedule、PyPTO lowering 或设备验证。

`source_candidate/score_vector_steps_g6.py` 包含：

- `score_vector_leaf_state_g6`：每个有效叶初始化一次 roots/segments。
- `score_vector_step_g6`：执行原完整一步，保留 scale 页读取、FIFO READY/load/ACK、有效候选裁剪、两步拼段、排序及归并顺序，显式返回两个跨步 Tile。
- `score_vector_leaf_publish_g6`：所有步结束后写回真实 query 的叶根；外层在此之后才能发布该波根完成。

G6 和 1024 只固定已有矩阵/排序策略的数学 tile；没有固定 B、T、本轮 S、历史长度、位置或页表。Tensor 动态轴直接引用原模块。其它策略尚未提取，仍由原路径覆盖；本文件不是独立完整 CSA，也不是用测试点生成的冻结 entry。

`prepare_source.py` 从指定 Git 版本读取原函数，保留原正文和中文注释，按数学常量改名后输出新 helper。三个正文 AST 逆归一化后与原文相同；所有自由名字都有显式参数或模块依赖。`source_audit.json` 记录核对结果。这里仅使用 Python 标准库，没有加载 PyPTO/NPU。

后续调用必须携带并接住 roots FP32[3,1024]、segments FP32[6,1024]，偶数步的半段不能在 Attention 切换时丢失。空 query lane 仍消费并 ACK，叶根 store 与 ACK 含义不同。跨步 UB 地址、异步存活和循环回边仍须联合编译验证；当前 JIT 的 Tile 参数注解问题由独立 overlay 修复，不把本次源级核对称作编译或精度 PASS。

两个状态沿原 `tile.full` 保持完整 valid_shape，真实 query/causal 有效范围仍由 Scalar 与 mask 决定。另一个动态 valid_shape Tile 循环夹具曾在类型检查中失败，本接口没有声称解决该问题，也没有改变正式状态的有效行语义。

重现源码核对：在仓库根运行 `python tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_vector_step_source/prepare_source.py`。
