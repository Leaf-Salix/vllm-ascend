# DQ 独立索引准备：保留 KV 共享合同的有界设计

唯一建议：**下一步只考虑在 DQ 每个完整八行 chunk 内用 `arange(512) XOR 1` 生成固定交换索引，删除 DQ 对 rope_swap 的那次读取；不新增 GM 表、不跨 dq_work 长期保留 Tile。** 有直接源码和生成物依据，值得之后做一次有界 CPU 核对；本轮仅设计，没有生成候选、编译、上卡或修改正式/线性分组源码。不能预记性能收益。

## 基线与排重

正式基线为 `9d5327e5`，其 `query_projection_stages.py:209–215` 仍在每个完整八行 head 组中读取 `rope_swap`，执行行偏移转换链。线性分组候选为：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/query_dequant_linear_20261018/`

其 `source_candidate/query_projection_stages.py:213–219` 仍保留相同准备；`cpu_v1/kernel/ptoas/dynamic_csa_full.pto:9507–9532` 确认这些操作没有因换分组自动消除。已有 generated_audit 表明线性候选与 9d 的完整/尾数学、Tile 分配及事件顺序一致；本报告不改写它的结论。

已查本会话主日志、静态/动态过程 Markdown 及本私有记录，没有找到 DQ 本地 `TCI+TXORS` 生成此置换的同方案。上轮否决的是修改共享 `rope_swap` 内容，因 KV 完整 32 行依赖 0～63 行内索引；本建议不改变这个缓冲或 KV，因此不是重做已否决方案。

## 唯一改动位置与数学合同

仅替换 DQ 完整块内的索引准备，保持其原作用域和现有 `q_flat_gather_tmp`；head 内 RMS、scale 乘法次序、RoPE 两乘一加、BF16 舍入、有效行和全部同步不动。尾路径完全保留。prepare、KV、binding、root ABI 均不需要改变。

当前完整入口已由 `q_rope_prepare_stage` 保证每行交换列为 `swap(c)=c xor 1`。对 `j=64*r+c`，`0<=r<8, 0<=c<64`，因为 64 是偶数且行偏移不占最低位：

`64*r + (c xor 1) = (64*r+c) xor 1 = j xor 1`。

所以直接生成 0～511 的 INT32 arange，再与 1 异或，可逐整数保持同一 gather 地址。与 T、request、position、head 都无关；完整块任意动态 T、mixed 请求均用相同局部置换。这里依赖完整 CSA 的 prepare 合同，不把 DQ 宣称为接受任意自定义 rope_swap 的通用算子。原共享表仍照常生产并供 KV 使用，现有 PHASE_ROPE 等待不能因此删掉，sin/cos 仍依赖它。

## A3 的真实开销：不是一条 XOR 指令

PyPTO 已有 `python/pypto/language/op/tile_ops.py:2768` 的 `xors(lhs, scalar, tmp)`，显式接受 INT32 Tile 和独立临时 Tile。当前隔离 PTO-ISA 的 `include/pto/npu/a2a3/TBitwiseSOp.hpp:123` 将 TXORS 展开为：

`TEXPANDS → TOR → TEXPANDS → TAND → TNOT → TAND`，内部共 5 个 `pipe_barrier(PIPE_V)`。

因此建议的变化是：原 `TCI + 5 TCVT + 2 TMULS + TADD + 2 KiB TLOAD`，变为 `TCI + TXORS`；而 TXORS 含上述六个向量操作阶段，绝不能说“九条变两条硬件指令”。原 TCVT 等本身也会再展开，最终机器指令数尚未取得。TCI、gather 和全部浮点数学仍保留，确定能删除的只是本次索引 GM 读取及旧行偏移转换链。

TXORS 要求 src、dst、tmp 三块 UB 不同址：512 个 INT32 各 2 KiB，共 6 KiB；TCI 还需要现有的 192 个 FP32 临时元素（768 B）。这只是对象容量计数，不能冒充新 allocator 峰值。原实际索引准备已有 arange/row 中间区 2 KiB、swap/cast 区 2 KiB、最终索引 2 KiB、TCI 临时 768 B，新增方向没有天然的大容量要求；是否仍能复用相同物理区须由真实编译确认。原最终索引地址 36864，长度 2048；gather 临时地址 38912，长度 2048。建议每个 chunk 局部生成，避免把新 2 KiB Tile 变成整个动态 work-loop 的长期活跃值，重蹈先前环缓冲规划膨胀的风险。

候选没有新的跨核数据或同步；删除一笔 TLOAD 会允许编译器相应调整局部 MTE2→V 事件，不能手删 global fence 或据此预记流水改善。5 个内部 barrier、额外整数位运算及别名限制可能抵消删除转换/搬运的收益，应以实际 C++/allocator 和完整 L1 判断。

## 为什么不先加一份 GM 布局

为了保留既有共享合同，单独保存八行展平表是可设计的，但 0～511 置换完全固定，最小版本应是一份只读 `[1,512]` INT32 表（2 KiB），而非再建 `[T,64]` 表。前者仍新增一个 GM 根参数与初始化/图生命周期合同，DQ 每个 chunk 仍要读 2 KiB；若每次由设备 prepare 生成，还多一次 store 和发布责任。初始化一次可避免每步 store/新 phase，但仍需新增 binding/ABI 和只读数据生命周期核验。后者更增加 `256*T` 字节与每八行 2 KiB 写入，重复存储相同置换，没有必要。

本地生成方案直接利用已有接口，不引入新 root、GM 布局或 producer。它也没有把索引长期留在循环外，因此是此处更小、边界更清楚的下一候选。代价只有上述局部生成与 UB 非别名要求；不据此声称一定优于只读 GM 常量表，更不在本轮并行新增第二种候选。

证据：[evidence.json](evidence.json)。未来若获准实施，先只核对同一数学、INT32 置换、TXORS 非别名、UB 分配和局部事件；编译若出现超量或不合理生命周期就停止，不拓展流水/布局参数。


## 补充：TGATHER mask、两列 reshape 与相邻元素交织接口

结论：**在本次实际使用的本地 PyPTO 与 A3 PTO-ISA 中，没有找到可直接替代 512 项交换索引的相邻两元素 shuffle 接口。** 这只限定当前软件实现，并不证明 A3 硬件不可能支持。仍可保留前述局部 `TCI+TXORS` 为待评估方向，但不能因为缺少直接接口就预判它优于现状。本补充只读源码，没有生成候选、编译或设备运行，也没有改变共享 `rope_swap` 合同。

1. **`gather_mask` 是压缩筛选，不是置换。** PyPTO `tile_ops.py:3119` 暴露 `gather_mask`；其 `gather.cpp:263–300` 将 P0101/P1010 的输出列数减半，将四选一模式减为四分之一，P1111 原样保留。A3 `TGather.hpp:106–157` 的行内模式使用 `vreducev2`，从每两个元素选出一个并紧凑存放。两次 mask 可以得到偶数、奇数子序列，但还缺把它们按奇偶互换后重新交织的一步，不能一条 mask 直接得到 `[1,0,3,2,…]`。`gatherb` 选择 32 字节块，FP32 相邻元素对只有 8 字节，不能改变一个 32 字节块内部的元素顺序。

2. **`reshape([256,2])` 不使 `[1,0]` 自动成为每行广播索引。** Tile `gather.cpp:47–95` 规定输出形状取自索引 Tile，因此 `[1,2]` 索引只得到 `[1,2]` 输出。A3 `TGather.hpp:33–91` 每行 `vgather` 都使用同一个 `src0Ptr` 基址，索引是整个源 Tile 的绝对元素位置；把 `[1,0]` 重复 256 行会反复取最前两个元素，不会补上 `2*r`。张量层 `gather(dim=1)` 可以表达逐行语义，但 `op_conversion_registry.cpp:1837–1893` 对 A2/A3 使用逐行循环，逐行建立输入/索引视图并做单行 gather；它不是共享两项索引的一次 256 行硬件重排。这里未实际编译，不能把这一分析写成已测编译失败或定量性能结论。

3. **当前 `TINTERLEAVE/TDEINTERLEAVE` 不适用于这条 A3 路径。** 公共 `pto_instr.hpp:2972–3065` 将这些接口包在 `PTO_NPU_ARCH_A5/A6/KIRIN9030/__CPU_SIM` 条件内，没有 A2A3；本地实现文件有 CPU 和 A5 版本，没有 `npu/a2a3` 版本。PyPTO 的公开语言与 IR op 定义中也没有对应 interleave/deinterleave 操作。不能看到公共头文件中的名字就假定本环境可调用。

4. **mask scatter 不能补成便宜的双调用交织。** A3 `TScatter.hpp:118–154` 的行内 mask 路径先初始化整个输出 UB，再用双层 Scalar 循环逐元素写入，并非一条向量交织指令。因每次调用都初始化输出，不能连续向同一 dst 写偶数、奇数位置并假定第一次的另一半被保留。用不同 dst 再合并还会增加临时空间和额外操作，因此本轮没有据此另提实现候选。

以上均为当前本地版本的实际源码合同。完整精确路径及取证片段见 [api_review.json](api_review.json)；不将 API 名称、源码行数或对象容量折算为实际周期，也不新增任何跨循环常驻 Tile。
