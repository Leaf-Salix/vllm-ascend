# HC 整行驻留与主 Q 预取候选：2026-10-17

本轮继续按档编译；当前两项只限 B16/S6/T96。对照是原动态 PTO，正式 Layer4 权重，
完整 CSA 的 L1 incore 首尾为性能口径。没有执行 Native，没有修改生产或依赖仓。

## HC mix/norm 整行驻留

`hc_mix_resident.py` 将 mix/norm 从 12 个 AIV、每个 8 行改为 48 个 AIV、每个 2 行。
一行的四路输入和 BF16 mixed 留在 UB，保留原 mixed GM 写出，但删除后续 GM 回读
786432 字节。门控相加顺序为 (y0+y1)+(y2+y3)，norm 权重仍是 BF16 入参。

必须纠正早期实现中的归约契约错误：冻结的 `mix_x_rms_norm` 实际是
`pl.pipeline(16)`，每次 256 列，即 **16 个 K256 部分和**依次 Vector 累加。
初版错误地使用 8 个 K512 部分和，改变了归约顺序。旧 audit 没有检查这一点，
不能作为数学等价的证据。相关源码和失败结果保留，不改写历史结果。

| 版本 | 结果 |
| --- | --- |
| prepare/cpu v1 | runtime Scalar 不能作为 tile.full 的 value，CPU 失败 |
| prepare/cpu v2 | INDEX 到 FP32 的 scalar cast 不支持，CPU 失败 |
| prepare/cpu v3 | CPU 通过；错误的 8×512 RMS，Scalar 累加 |
| prepare/cpu v4 | CPU 通过；8×512 不变，仅恢复 Vector 累加 |
| prepare v5 | 恢复 16×256 和 0..15 Vector 累加，并强化冻结源码契约检查；验证中 |

两个旧候选都在精度门槛处停止，没有采性能：

- `task_20261017_053012_51687711509` / `b16_row_deps_hc_mix_resident_v1`：exit1。
- `task_20261017_053800_64833816891` / `b16_row_deps_hc_mix_resident_v2`：exit1。
- 两者相同：输出 15613 处不同、max_abs=0.015625；Top-K 825 处不同。
  SWA、compressed、state 和 indexer_state 也有差异；Indexer INT8 cache 和 scale 相同。
- Scalar 改 Vector 后结果未变，因此不能归因于 Scalar 加法舍入。
  16×256 修正尚待设备证明，不预先宣称它已解决全部差异。

原 CPU 审计还保留一次脚本错误：最初对冻结 body 而非已选 QK 地址缓存 body 做比对，
错误地把 QK 改动也计入本候选；修正参考后仅 mix 函数变化。

## 主 Q 的普通 N128：未上卡

`qproj_n128.py` 保持 20 个 worker、K256 的四段 INT8 累加，将 N256 改为 N128。
prepare/cpu v1 完整生成通过，但实际四个 Right Tile 全部地址 0；
`qproj_n128_codegen_audit.json` 明确 `two_right_buffers_present=false`。
没有实现预期双缓冲，因此没有为此候选提交 NPU 测量。

## 主 Q 的显式 L0 预取

`qproj_l0_prefetch.py` 保持 N256、20 worker、128 列块，将 K 改为 128，
先加载下一块 Left/Right 再提交当前块 matmul，制造真实的重叠活跃期。
QR 的 96×1024 INT8 仅在 worker 开始时装入 L1。每份 Right 为 32KiB，Left 为 12KiB；
两份分别占 64KiB、24KiB，Acc 为 96KiB。INT8 1024 项点积的最大绝对和
16777216，不会 INT32 溢出；仍需完整输出逐 bit 验证。

prepare v1 已通过；cpu v1 进行中。源级预取不等于实际双缓冲或硬件重叠，
必须核对生成 C++ 的 Left/Right 地址与提交顺序后才决定是否上卡。


05:44补充：HC prepare/cpu v5完整通过，IR确认16×256，代码生成保留V到S屏障；
任务task_20261017_054339_75460017083已提交，设备结果待收取。
主Q CPU v1完整通过，两处Left地址0/12288、Right地址0/32768交替，八段提交顺序通过；
任务task_20261017_054303_7422476042已提交。无设备结果前不预记精度或性能。

## 最终结果：两个新任务均 exit0

| 候选 | 单SPMD三个完整L1窗口/μs | 中位/μs | 同轮原动态中位/μs | 采用 |
| --- | --- | --- | --- | --- |
| HC 16×256整行驻留 | 822.46 / 819.02 / 817.06 | 819.02 | 911.56 | 否 |
| 主Q显式L0双缓冲 | 825.54 / 831.16 / 830.62 | 830.62 | 892.68 | 否 |

两者正式权重固定fixture的输出、TopK、六份cache/state、图重放、30代进度和保护区全部PASS。
HC恢复原16×256归约后消除了前两轮差异；没有修改阈值，也没有把精度失败结果当性能测量。
实际NPU产物确认HC的归约指令与CPU相同；主Q两处Left/Right双缓冲地址及提交顺序再次通过，
见各任务目录中的codegen_audit文件。不使用hash。

没有显示比现选行依赖方案更有利的结果，因此不采用、不追加跨档或动态边界测试。
与现选807.16（首轮）/817.88（同进程复核候选）的比较来自不同运行，
不能量化为严格的退步幅度。七档选择不变，700μs目标未达。

## 避免重复的只读核对

重新审视Indexer M192/N128、WS K192、六query拆成两组三query的设想时，确认
indexer_n128.py已经完整实现并测过：此前883μs，不再重做。
主Q普通N128未形成双缓冲与本次显式K128双缓冲是不同候选，两者都保留证据。
当前HCpost已经按一整行4096列计算，不能误称它还在用16次N256分块。
