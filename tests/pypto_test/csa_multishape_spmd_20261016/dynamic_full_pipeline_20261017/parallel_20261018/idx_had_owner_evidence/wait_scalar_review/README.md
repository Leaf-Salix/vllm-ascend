# Indexer RoPE → Hadamard 的 wait Scalar 表达式只读核对

结论：两档同轮的设备 C++ 均保留取模；新 16-owner 路径还保留动态起点的除法、乘法和加法。新取模在当前固定 24 AIC／48 AIV 合同下严格冗余，可以用一个连续 owner helper 消除，但目前证据不能把长档回退归因于这些表达式，也没有性能收益预测。未改正式或冻结源、未编译、未运行设备、未做 hash。

核对原件来自 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/idx_had_owner_h131072_b16_v1` 和 `idx_had_owner_h8192_b32_v1` 的 `source_matrix/result.json`，按角色键读取 `kernel_configs`，没有把不同作业当成对照。此前 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/idx_had_owner_evidence/generated_review` 已完成数学、allocation 和局部事件序列审计，本次仅补 Scalar 等待地址表达式。原件路径、大小见 `INPUT_ARTIFACTS.json`，小片段带原行号见 `wait_blocks.txt`。

| 同轮档位 | reference / candidate 中位数（µs） | 配对窗口更快数 |
| --- | --- | --- |
| 128K / B16 | 875.28 / 884.52 | 0 / 5 |
| 8K / B32 | 998.86 / 991.82 | 4 / 5 |

两档完整精度通过，但结果方向相反；本候选尚未采用。原始各五个样本保留在 `review.json`，这张表不是新实验。

原有 `phase_sync.py:121` 的 `wait_row_owners` 按轮转 owner 计算 `(first_tile + ordinal) % producer_workers`。原 Cube 调用的 first 为 0，生成语义为：

```text
U = ceil(T / 8) * 16
if U > 0:
    for ordinal in [0, min(U, 48)):
        offset_words = 156672 + 128 * (24 + ordinal % 48)
        wait_GE(signal[offset_words], generation)
    fence; cacheinvalid; fence
```

候选 `cube_schedule.py:313` 在有效 AIC 上传入 `first=(worker//8)*16`、count=16，生成语义为：

```text
if worker < T:
    for ordinal in [0, 16):
        owner = ((worker / 8) * 16 + ordinal) % 48
        offset_words = 156672 + 128 * (24 + owner)
        wait_GE(signal[offset_words], generation)
    fence; cacheinvalid; fence
```

其中 156672 是 phase 17 的 word 基址，即 17×72×128；24 是 AIV 物理核基址，128 个 INT32 是已有 512 B 单写者槽隔离跨度。long reference C++ 为 `decode_csa_spmd_kyg6wy7u/kernels/aic/cube_schedule.cpp:10282` 附近，candidate 为 `decode_csa_spmd_0du1mi6c/...:10282` 附近；short 分别为 `decode_csa_spmd_lrmhgmqz/...:9396`、`decode_csa_spmd_3gsduf5a/...:9396` 附近。

实际差异有明确边界。候选 PTO 保留两份 `worker/8` 及 `min((first+16)-first,48)`，但最终 C++ 已将循环上界折成常数 16，并未保留这两份 count 推导。候选 C++ 的地址 RHS 仍写有 `worker/8*16 + ordinal` 再 `%48`；旧 C++ 的地址 RHS 为 `ordinal%48`，因此旧代码的地址表达式较简单，而旧循环上界仍含动态 U/min。两者的 signal wait 都调用同一 `CSA_WAIT_LOCAL_GM`；DCCI、volatile 读、100 tick 失败退避和周期 PIPE_ALL 保留，整批等待后的业务 fence/cacheinvalid/fence 也保留。

这不是最终机器指令计数。保存的 `prebuilt/kernel_0.bin` 是裸 AIC 机器码，未含 ELF magic；对应 build 下未留 .o/.elf/.s/.ll。现有 9.0 和 8.3 llvm-objdump 均报告不是可识别对象（8.3 加 `--disassemble-aicore` 同样如此）。本审计没有为此包装对象或新编译。因此尚未确认 CCEC 是否将除法/取模外提、强度削减或消除；C++ 多几个运算、二进制长短以及少等若干信号，都不能直接推导 µs 成本。两档同样的源级表达式差异也不足以解释方向相反的实测。

最小后续设计只涉及 `phase_sync.py` 新增一个连续 owner helper，以及 `cube_schedule.py` 的此一调用点。调用仍位于原 `worker<T` 条件内，`first_core=24+16*(worker//8)`，固定循环 ordinal=0..15，直接计算 `core=first_core+ordinal` 并调用 `csa_signal_wait`。循环结束后只执行一次原业务 fence/cacheinvalid/fence。不可在循环中调用现有 `wait_producer`，否则会把业务全缓存维护从一次变成 16 次。其它 wait、发布、72 核 begin、数学、FFTS 和 payload 都不变。

删除取模的证明不依赖实测 T：0≤worker≤23，所以 first∈{0,16,32}；0≤ordinal≤15，所以 0≤first+ordinal≤47，取模恒等。加物理偏移后恰为 24..71，不发生环绕。worker<T 保证小 T 尾部仍仅等待实际活跃的 16-owner 区间；T≤8 为 0..15，9≤T≤16 的活跃 AIC 对应 0..15 或 16..31，T≥17 时已有全 48 个 RoPE owner。它与已审阅 16-owner 候选的槽集合及访问顺序完全相同，不改任何 release/acquire 边。

旧 first=0 路径的取模同样可证明冗余，因为 ordinal<min(U,48)≤48；本设计不把其它调用点的轮转取模一并删除。是否值得 CPU/设备对照由主会话决定；这里只提出可证伪的表达式简化设计，没有实现候选。

