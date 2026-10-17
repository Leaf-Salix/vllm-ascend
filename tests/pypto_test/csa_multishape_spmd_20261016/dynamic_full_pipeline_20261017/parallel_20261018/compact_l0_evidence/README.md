# Indexer compact projection 显式 Right 双槽 CPU 候选

基线 `593c5d25`，独立稀疏 worktree `csa-ultra-compact-l0-20261018`。仅修改
`compressor_projection_stages.py` 的 Indexer 投影 helper 及中文说明，不叠加阶段
重排、不改共享依赖或主目录、不编译固定 shape 替身。本轮只做完整 CPU lowering，
没有 NPU 任务、设备精度或性能结果，不计算内容摘要。

## 设计及不变式

原 M16/N32/K512 的八个 K 段按 pair=(0,1)/(2,3)/(4,5)/(6,7) 表达，四次 MMA
仍为 `KV0→gate0→KV1→gate1`。两条 FP32 accumulator 各自保持 K0..7 顺序；
只有 pair0 的两次 K0 MMA 初始化，K1 及后续 pair 均继续累加。转置使用视图，
没有数值转换或归约改写；输入有效行仍 `min(16,T-row0)`，输出仍写原 M16 pad。
T、输入/输出容量和 worker 网格保留动态；没有改动 Q/L/position/page 或阶段协议。

每 pair 的六次 L1 load 对应原 pipeline 展开后的两份 x 和四份权重。两份 Left
各 16 KiB；先形成两份 KV Right，各 32 KiB，二者在 KV0 消费前同时存活。
随后 gate0 可复用已消费的 KV0 槽，KV1 必须跨过 gate0 的搬运与 MMA 存活。
目标是消除此前融合版把四份 Right 全分到地址 0 带来的 K0 gate→K1 KV 覆盖边。
不增加硬件事件、核间通知、缓冲 reserve 或全核屏障；具体核内事件仍由后端生成。

普通 pair range 取代原 stage2 标签，避免重复请求额外的 pipeline 槽。是否真正保住
两槽、是否引入其他资源异常，均以本轮完整生成物为准，不以源码形式认定成功。
若资源失败或降槽，仅允许一次有直接证据的修正，保留首轮源码与日志。

## 首轮命令

使用 long 的 G6/N64/1024/balanced 策略，O-A=128/O-B=96 与现有当前长 B16
诊断一致；这些只是既有 constexpr 控制，不传入静态 T/B/L 或样例 Tensor。

```bash
compact_worktree=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-l0-20261018
compact_driver=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
compact_evidence="$compact_worktree/tests/pypto_test/ultra_compact_l0_20261018"
bash "$compact_driver/reproduce_cpu.sh" "$compact_evidence/cpu_long_b16_v1" \
  --attention-query-metadata off \
  --spmd-source "$compact_worktree/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd" \
  --controls-json "$compact_evidence/controls_long_b16.json" \
  > "$compact_evidence/cpu_long_b16_v1.log" 2>&1
```

`candidate_v1.patch` 保存首次源码差异；`cpu_long_b16_v1.log` 保存完整日志。
后续仅追加实际完成状态、物理分配、事件及资源对照。

## 首轮完成：目标两槽与等待边成立

`cpu_long_b16_v1/result.json` 与便于归档的 `cpu_result.json` 均为 PASS。首轮即
完成完整 symbolic/CPU lowering，没有源码修正轮、容量失败或设备执行；58 个
动态维度、单 MIX 提交、原 ABI 和五份权重 BYPASS 审计通过。编译器日志原样保留。

实际 `cpu_long_b16_v1/kernel/ptoas/dynamic_csa_full.pto`：

| 对象 | 地址 / B | 源码与物理约束 |
| --- | ---: | --- |
| KV0 Right | 0 | :814，BF16[512,32] |
| KV1 Right | 32768 | :817，BF16[512,32]；KV0 消费前已经形成 |
| gate0 / gate1 Right | 0 / 0 | :826 / :836，复用已消费的槽 |
| Left0 / Left1 | 0 / 16384 | :809 / :811，BF16[16,512]，同一动态 valid_row |
| KV / gate Acc | 0 / 2048 | :772 / :773，FP32[16,32]，valid_row=16/valid_col=32 |

MMA 在 :821/823、:830/832、:834、:838，分别为 KV0、gate0、KV1、gate1。
前两组只在 pair==0 初始化，后两组始终累加。输入 valid_row 仍为
`min(16,T-global_row0)`，两份输出仍按完整 M16/N32 写原 pad。

实际 `kernel/kernels/aic/cube_schedule.cpp` 片段保存在
`candidate_cpp_excerpt.txt`：KV1 的 TMOV 在 :8036，早于首 MMA :8042/8044；
gate0 MMA :8068/8070 后，KV1 MMA :8074 直接使用驻留于 32768 的 Right，
不再需要原基线的 gate0→KV1 Right 覆盖等待。gate1 的搬运仍在 :8089 等待
gate0 消费完地址 0，再于 :8092 搬入；这一必要边保留。:8073 有 PIPE_M
barrier，维持相应矩阵流水顺序，本审计没有删除它或称其为收益。四次 MMA
本身数量、形状和提交顺序未改变。候选的预取/初始化及 loop 边界仍可能有代价，
CPU 地址变化不等于性能收益。

## 完整源码及资源审计

`python audit.py` 已通过；原始输出为 `audit.log`，详细结果为
`audit_result.json`。该脚本直接读取实际 baseline/candidate 的 helper AST，
执行标量控制流，分别比较三份列表：GM load 的来源/偏移/物理形状/valid_shape、
MMA 的 operand/形状/初始化条件、store 目标及 pad 范围，并检查每类操作内部的
顺序。它没有比较三类操作合并后的完整指令时序；本候选有意改变搬运与 MMA 的
相对时刻。它也不模拟浮点舍入，不能代替设备逐 bit 精度。

- 156 例通过：T=0..33，以及 64/96/128/144/168/192 周围的尾块，分别使用
  1/16/24 个 worker，覆盖空核、空 T、完整行块和多波次。每例两实现的 load、
  MMA、store 三份列表分别一致，包括各自内部顺序；不表示三类操作交错时序
  一致。所有有效 GM 访问在输入范围内。
- 33 份源码直接文本对照，仅 `compressor_projection_stages.py` 改变；该文件
  内主 Compressor helper 的 AST 和 Indexer helper 的参数注解均未改。
  cube/vector schedule、policy、FFTS、发布获取和 metadata 全部与基线逐字一致。
- 对本轮 CPU PTO 与当前 593 长 B16 设备诊断 PTO，按源 helper/空间/dtype/
  物理形状/地址的多重集逐项比较：**全部分配只有一项变化**，Indexer helper
  的一个 BF16[512,32] Right 由地址 0 改为 32768。其余 helper 的 L0/L1/UB/
  Acc 分配地址及数量均一致。这比只比全函数峰值更严格；不比较 SSA 变量名。

| 物理地址上界 | 593 / B | 候选 / B |
| --- | ---: | ---: |
| AIC Mat | 475136 | 475136 |
| AIC Left / Right | 65536 / 65536 | 65536 / 65536 |
| AIC Acc | 131072 | 131072 |
| AIC Vec | 16384 | 16384 |
| AIV Vec | 163872 | 163872 |

这里是 `addr + 物理行列×dtype字节` 的地址上界，不是动态活跃峰值或硬件占用率；
全函数合法性由完整编译另行检查。本轮没有引入新的全函数容量异常。

## 已冻结的设备候选交付

`source_snapshot/` 保存 33 个只读 Python 文件，与本次完整编译使用的
`cpu_long_b16_v1/source/` 逐字一致；路径、字节大小与直接对照结果保存在
`source_snapshot_manifest.json`。之后不再修改候选源码或这个快照。

设备测试没有新增开关，只以 `--spmd-source` 指向以下绝对目录；其余使用
主驱动当前长 B16 策略。不要叠加 compact 阶段前移或其他候选：

```text
/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-l0-20261018/tests/pypto_test/ultra_compact_l0_20261018/source_snapshot
```

是否提交设备任务由主 agent 决定，本 agent 没有提交。完整 CPU 产物保留本地；
归档只需 README、首轮 patch、controls、CPU 日志/小报告、audit 脚本/结果、
C++ 片段及 source_snapshot/manifest，不需要复制约数十 MB 的 passes dump。

## 主会话追加：设备候选已提交

主会话已经提交独立128K/B16三方五窗任务`task_20261018_014826_4729059881`，
使用上述冻结源码；另一个compact阶段前移候选没有叠加进来。
此条为提交记录，不预记设备精度或性能结果。
