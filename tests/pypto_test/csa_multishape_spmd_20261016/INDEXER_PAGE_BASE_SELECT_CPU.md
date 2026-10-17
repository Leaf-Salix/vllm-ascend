# Indexer 页视图的 CPU 可行性检查（2026-10-17）

结论：本轮两个入口方向均不满足当前编译接口要求，没有提交设备任务，没有改依赖。
不能将这些结果理解为 CSA 性能或精度失败。

Native 每页4160字节：4096字节Key加64字节scale。当前AIC按物理页地址对128的余数，
选择原始和偏移64字节的两个GM视图。希望消除逐页奇偶分支，先核对以下接口边界。

1. 自定义三维stride：当前PyPTO JIT Tensor参数不接受自定义`TensorView(stride=...)`。
   本地`tests/ut/jit/test_jit_compile_extraction.py`的`TestUnsupportedLayoutSlot`明确检查此拒绝。
   `pl.view`也不提供任意stride/offset描述。已读源码，不再为已有明确限制重做设备探针。
2. 先选择两个已有GM输入之一、再统一`gather_row`：v1使用未读取block index的
   `with pl.spmd(1)`被前端拒绝，是探针写法问题。v2改为`for worker in pl.spmd(1)`并
   在control地址中使用worker；随后明确拒绝不同backing SSA的if返回：
   `IfStmt in-place return_var 'source__rv_v0' yields different backing SSAs across branches`，
   两个backing分别为`even__ssa_v0_view`和`odd__ssa_v0_view`。

完整探针与日志：`indexer_page_base_select_probe_v1.py`、`indexer_page_base_select_probe_v2.py`，
`indexer_page_base_select_probe_cpu_v1/`和`indexer_page_base_select_probe_cpu_v2/`。
已有相同backing的Tensor条件返回修复不能直接外推为GM指针选择支持。
没有尝试将65×64页视图随意reshape为32×128的NZ Tile，因为物理布局等价性未证明。

顺带核对了交接建议中的缓存维护分类：`payload_cache.py`已按Scalar GM读写分类，
保留信号DCCI/DSB和pipeline fence，仅省去无Scalar读写阶段的整DCache维护。
旧`b16_cached_dynamic_precise_v1`已完整PASS，中位838.68μs；03:47也复核过。
本轮只是确认它不是新方案，没有重复实现或重测。
