# Query DQ 预存八行展平 RoPE 索引：只读可行性结论

**“仅改 query_projection_stages.py 的 prepare 与 DQ 两函数”不可行，有明确的 KV 消费者反例。** 索引预计算本身成立，也有真实重复指令可删，但必须先把共享 `rope_swap` 的主 KV 消费合同一并处理。当前没有生成候选、编译或设备测试；没有修改正在验证的其他 DQ 方案。

## 生产、消费与反例

当前正式包 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/` 中：

- `binding.py:172` 为 `rope_swap[T,64]` 分配独立 INT32 缓冲；`entry.py`/`vector_schedule.py` 直接传递并返回其原 GM 别名，没有另一份 DQ 专属副本。
- `query_projection_stages.py:325` 的 `q_rope_prepare_stage` 是生产者，按 `qrp_t0=8*tile` 写入每行的 `swap(c)=c xor 1`；最后不足八行只写有效行。
- `q_dequant_rope_stage:209` 在完整八行路径读取它，再算 `64*r+swap(c)`；DQ 尾路径本来自己构造索引，并不读取这张 GM 索引表。
- **`vector_schedule.py:674` 把同一缓冲传给 `kv_norm_rope_stage`。`kv_stages.py:211–212` 的完整 32 行路径读取 `rope_swap`，调用 `pl.gather(..., dim=-1, index=...)`。这里要求每个索引在 0～63。** KV 尾路径自行构造展平索引，不读 GM 表。

T=32 已是充分反例：拟议 prepare 在第 1 行、第 0 列存 `64*1+1=65`；KV 对第 1 行的 64 列执行行内 gather，却收到 65，而应为 1。已有完整运行产物的 `dynamic_csa_full.pto:8047–8056` 将 KV gather 降为 32 次 **1×64 行 subview + TGATHER**，没有自动剥离八行偏移。越界列不是另一种合法布局，也不能靠保留不足八行尾块的旧格式修复。全部 T>=32 都可能含至少一个该完整 32 行消费者。

当前 prepare 后的 `PHASE_ROPE` 发布，以及 KV/DQ 的现有等待只保证读取到本代数据，不能使错误的索引含义变正确。不得在 KV 与 DQ 之间原位改写表：各 AIV 推进异步，这会再引入读写时序协议。

## DQ 侧的精确证明和可删操作

对任意完整八行块 `tg=8*k`、局部 `r∈[0,7]`、`c∈[0,63]`，令 `j=64*r+c`。现有 DQ 算式为：

`trunc(float(j)/64)*64 + swap(c) = 64*r + (c xor 1)`。

这些整数最多 511，且除数为 2 的幂，现有 FP32 中间数能精确表示。拟议预存值恰是该局部八行 Tile 的展平地址，与全局 tg、request 边界和 head 编号无关。prepare 与 DQ 的起点都为 8 的倍数；两者完整块条件均为 `tg+8<=T`。因此**只看 DQ**，同代预存后 load→reshape 可以保持相同 gather 地址，RoPE 的乘/加、RMS 和 BF16 舍入都无需改。最后不足八行继续旧 prepare 格式及原 DQ 尾数学是自洽的。

在已运行 `current_clock_h131072_b16_v1/build_output/decode_csa_spmd_jqlk5eey` 的 `ptoas/dynamic_csa_full.pto:9184–9205` 和 `kernels/aiv/vector_schedule.cpp:47351–47440`，每个完整八行 **head 组** 的索引准备实际生成：

- 1 次 TCI（512 个位置的 arange，带 192 个 FP32 元素的临时 Tile）；
- 5 次 TCVT、2 次 TMULS、1 次 TADD；
- 相应临时 Tile 和流水依赖。reshape 本身主要是别名描述，不能计为一笔 GM 搬运收益。

DQ 的 `rope_swap` TLOAD 仍要保留，TGATHER 及其临时空间也不消失。预计算把上述索引工作移到 prepare 的有效 worker 上，可被其多次写回复用；不是凭空删除所有成本。当前 head 分组已减少准备次数，剩余次数为 `floor(T/8)*(64/head_tile)`，不能把旧 4-head 次数套到最新策略上。最终 UB 分配、实际 barrier 是否减少、完整 L1 净收益均尚无候选证据。

## 排重与后续判断

已查主过程日志、静态/动态两树 Markdown 和本私有优化记录，没有找到“完整八行在共享 rope_swap 中预存局部展平索引”的同方案。它不同于已经采用的 head 扩组：扩组降低准备频率，此方案改变生产/消费的索引表示。

建议保留为**需补 KV 合同后才成立**的独立候选，当前两函数版本应否决。最小安全范围至少涉及第三个 helper `kv_norm_rope_stage`：让完整 KV 路径仍拿到合法 0～63 行内索引，或者一起改其 gather 布局；两者都需保持原浮点算术与现有同步。这超出“只动一个文件两个函数”的理想范围，本轮没有选择具体实现。不能只在 T24 验证后推广，因为 T24 正好只走 KV 尾路径，会掩盖上述 T32 反例。

完整产物路径、摘录及操作计数见 [evidence.json](evidence.json)。没有根据指令数推算微秒收益，也没有为了本建议增加全局 fence、新 GM 缓冲或 runtime 根参数。
