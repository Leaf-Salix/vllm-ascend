# Indexer Key L1行池扩大：CPU通过，未上卡

2026-10-17，`indexer_key_pool.py`只变换现选B16生成源码中的Indexer AIC循环。
Key行池从128×128 INT8扩大至1024×128，16KiB→128KiB，每个Native Key页
占独立32行；16个N64 Right提取使用0,64,...,960。页地址、原提交顺序、
QK/WS、FP16 FIX/ReLU、store和FFTS保持；没有增加预取深度。

`indexer_key_pool_prepare_v1`和`indexer_key_pool_cpu_v1`通过。
用done_16标记和完整C++块边界抽取stage16，两版均32个matmul，
MTE1→MTE2 wait由19处变20处，MTE2→MTE1 wait均18处，PIPE_ALL均1处。
实际没有减少目标等待，因此不提交设备、未接driver、不扩档。
静态指令计数不等于运行时停顿；不声称精度或性能已经测过。
证据`indexer_key_pool_codegen_audit.json`与两份完整阶段C++片段。
初次按v30前缀抽取只覆盖后半段，已改用完整块并断言32个matmul；
不使用此前15个matmul的不完整计数作结论。
