# nalinaly c346实际编排容量审查

独立审查和本线程生成物复核；最终与初次编排逐字节一致，harness报告修正不改kernel。

8K/B32/T192、G2/Score768/OB128、debug57根参数。单一SIMPLER_SCOPE内70个TensorCreateInfo/33次alloc_tensors均在唯一常驻MIX任务前分配。原始payload226625704B=216.127113MiB；按PACKED_OUTPUT_ALIGN=1024逐tensor对齐后226632704B=216.133789MiB，256MiB余41802752B=39.866211MiB。

alloc0/1/2分别26.924805/60.566406/23.521484MiB；其余105.121094MiB。最大单组60.566406MiB；reshape/view无新增payload，33alloc+1kernel低于2048窗口。

没有明显静态超限阻断，但不保证跨调用回收或设备正确，更不构成128K容量资格。内部scratch不可由外部guard直接检查；Native/cache外部guard、eager/raw及ABA仍严格检查。

## 跨调用回收复核

曾对“跨调用保留heap_top”建模得到第三次环尾空间不足；该前提经固定Simpler57e499源码核验不成立。每次kernel执行先aicpu/aicpu_executor.cpp:664 runtime_reset_for_reuse，runtime/shared/runtime_init.cpp:467重新task_allocator.init，runtime/ring_buffer.h:108–109将heap_top/heap_tail归零。当前root只有一个scope，因此该跨调用碎片推测不阻断warmup/replay；未因推测调整ring配置。此结论不外推同launch多scope或128K。
