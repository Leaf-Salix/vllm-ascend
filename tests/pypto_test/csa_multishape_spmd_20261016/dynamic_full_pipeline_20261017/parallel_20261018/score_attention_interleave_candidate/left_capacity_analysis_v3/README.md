# 联合交错 v3：实际内存池与别名边界

本次仅审阅 `cpu_long_v3/kernel/passes_dump`，不修改候选，不再编译/NPU。v3 已通过 parser/Inline/SSA，在 AllocateMemoryAddr 报 Left 77,824 > 65,536 字节。这里列 pass37 的实际 allocation，总量不是手工推测的 Tile 活跃包络；完整大 IR 留在原目录，仅保存小片段与路径/大小。

| 空间 | pass37 实际池 | 总量 |
|---|---|---:|
| Left | `mem_left_497=12,288`，`mem_left_502=65,536` | 77,824 B（76 KiB） |
| Mat | `mem_mat_487=49,152`，`489=49,152`，`702=262,144`，`1195=393,216` | 753,664 B（736 KiB） |

**Left 已有复用，不能说 Query 与 Attention 同时占了两个独立池。** Score Query Left 为 `502@0,size49152`；Attention PV Left 同为 `502@0,size16384`，QK 两 K128 Left 为 `502@0/16384`。502 的 64 KiB 上界来自更早的主压缩投影两份 K256 Left（32+32 KiB），不是 Attention 与 Score 的数学必须同时活跃。额外 497 的 12 KiB 用于 Coeff 分支归并：叶初始化写497，保留旧状态的分支也复制到497，循环末再复制到502@49152。Query Mat 同样存在 487→489→487 的分支/回边复制。它们是生成代码中的真实 `tile.move`，不能简单删除或将两池强合。

**Mat702 的 256 KiB 是更优先的正确性问题。** pass35 中 gather 结果与紧接的 `transpose_view` 都是702；pass36 MaterializeSemanticAliases 将 gather 结果改为486，而 view 仍702；pass37 将父486合入1195，view仍702。两者之间没有数学 copy。`tile.gather_row` 注册为 output-reuses-input(0)，`transpose_view` 是纯 metadata alias，因此该分裂不是合法双缓冲或必要回边。相同尾 wave 调用在 while 外，最终父/view 同1195，是已有正对照。修复必须让 view 跟随父窗口，不能用两块物理地址偶然相等掩盖失配。

源码还有一个与 Coeff/Query 分裂一致的遍历缺口：`memory_reuse_pass.cpp` 的 `TopDownRetargeter::VisitForStmts`（1286）和 `VisitIfPhisForAccumulator`（1327）均递归 Seq/For/If/Scope，但没有 While。第一 executor 的 For 位于外层 Scalar while 内，尾 executor 在 while 外；前者失去这轮语义归并，后者正常。这里只记录源码与产物对应关系，尚未执行修复验证，不把它当作全部分配问题已解决。

当前组池规则仍应保留：生命周期按同 base 合并；循环前定义且在循环内使用的值延长到循环末；同一 if 返回槽的互斥 producer 可以共享，但外部仍可观察别名不能被豁免；流水 stage、read/write hazard、no-alias 与固定分配隔离均必须保持。后端还要据真实 alias 建立 M/MTE1/MTE2 及回边依赖，Scalar 顺序本身不代表 DMA 或 MMA 已结束。

只将 Coeff 从 Left 搬到 Mat 不能据此保证容量通过：当前 Mat 池已超512 KiB。即使修正702的错误身份，736−256=480 KiB 也只是对旧池表的差值，重新规划可能改变池及事件，不能预报新编译一定为480 KiB或能fit。先修 must-alias 传播，再依据新产物判断容量。

复核命令：`python3 extract_evidence.py`。脚本仅用标准库读已有三个 pass，生成 `evidence.json`、`IR_EXCERPTS.md`、`LOCAL_ARTIFACTS.json`；无哈希。编译器新补丁及其测试另在独立 PyPTO worktree 准备，不属于本份 v3 阶段证据。
