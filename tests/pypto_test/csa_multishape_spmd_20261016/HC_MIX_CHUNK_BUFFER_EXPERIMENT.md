# HC mix 分块 UB 缓冲：2026-10-17

候选完整通过，但没有显示出优于现选策略的明确收益，未采用、未扩测其他六档。
128K/B16/S6 的完整无插桩 L1 中位为 **814.74 μs**，同轮原动态 PTO 为 907.88 μs。
完整三窗、Event 时间和校验摘要见 [hc_mix_chunk_buffer_result.json](hc_mix_chunk_buffer_result.json)。
现选行依赖版的首轮结果为 807.16 μs；没有同进程直接 A/B，不能将跨运行差值当作精确回退幅度。

实现为 [hc_mix_chunk_buffer.py](hc_mix_chunk_buffer.py)，实验开关 `--hc-mix-chunk-buffer`。
它保持原来的十二个成员、每成员八行、两个 16×K256 pipeline，以及所有数学、BF16 舍入、
RMS 累加顺序和 norm 权重类型。只在混合阶段保存每个 BF16 块，供随后的 norm 从 UB 复用。
原 mixed GM 写出保留；删除每轮共 786432 B 的 mixed GM 回读。

与此前 `hc_mix_resident.py` 的区别：此前版本改为 48 成员逐整行处理，并用 Scalar 读取各部分和。
本版本保持原八行 Vector 数学体，没有增加 Scalar 归约。也没有叠加 HC 三阶段融合。

## 布局及编译过程

首版 `hc_mix_chunk_buffer_cpu_v1` 将 8×256 块插入 8×4096 宽 UB Tile，PTOAS 拒绝生成的
`pto.tmov`：`expects A2/A3 non-mat tmov to use matching src/dst shapes`。未上卡。

第二版按块优先存放为 128×256 BF16，每个 K256 块占连续八行：

```text
UB[k*8 + row, col] = mixed[row, k*256 + col]
norm 第 k 块读取 UB[k*8 : k*8+8, :]
```

`hc_mix_chunk_buffer_layout_proof.json` 验证 32768 个元素完整覆盖、唯一写入。
`hc_mix_chunk_buffer_prepare_v2/` 与 `hc_mix_chunk_buffer_cpu_v2/` 完整编译通过。
生成 IR 中仅一份 65536 B 驻留 Tile，地址 98336～163872，未越过 188416 B 可用 UB。
原 pipeline 的分块插入与 norm 提取均落入生成代码，证据在 CPU 目录的 `codegen_audit.json`。
没有修改 PyPTO、PTOAS、PTO-ISA 或硬件容量配置。

接入正式实验开关后，`hc_mix_chunk_buffer_prepare_v3` 的完整 entry AST 与已编译 v2 相同，
没有重复执行一次相同的 CPU 完整编译。

## 真机结果和范围

任务 `task_20261017_063343_160455814507` exit=0，原始结果在
`b16_row_deps_hc_mix_chunk_buffer_v1/result.json`。

- 正式 W8A8 权重，原动态 PTO 为参照，不执行 Native forward。
- 输出、Top-K、六份 cache/state 逐 bit 检查通过。
- 图重放、30 代信号和输入/工作区保护区通过。
- 固定 fixture 每次恢复，不代表连续 decode 状态轨迹或整网验证。
- 完整 L1 含等待与阶段间隙，三窗；无阶段探针。没有以局部数学耗时替代目标。

复现仍通过 `task-submit --device auto` 调用 `run_device_compare.sh 16 131072 新目录`，
参数为现选行依赖版的公共参数，再加 `--hc-mix-chunk-buffer`。完整任务命令可由队列记录核对，
实际驱动已保存在结果目录 `benchmark_driver.py`。不覆盖旧结果，不裸跑占卡。
