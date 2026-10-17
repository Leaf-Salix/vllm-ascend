# 593 长 B16 的流水深度只读审计

2026-10-18。仅使用已经生成的 CPU/设备编译文件，没有重新编译、执行设备任务、改主源码或依赖，也没有计算内容摘要。完整提取结果与带行号的 C++ 片段在 [result.json](result.json)；[extract.py](extract.py) 只读输入，写本目录 JSON。

结论：提示中的 Left group 20 是主 QB，22–26 是主压缩投影，28–30 是 O-A，均不是 Indexer Score。O-A、O-B、QA/KV、主压缩投影仍有真实的 L0 两槽，原分 task 的相应阶段也有同类降深度提示。不能把这些提示解释成“融合后所有 matmul 都单缓冲”。确有融合版才出现的三个分配差异：HC 的 Right、Indexer 压缩投影的 Right、主 QB 的 Left 从两地址变成一地址，生成 C++ 也有相应 M→MTE1 等待。它们位于 Score 之前，不能据此解释当前 Score 约 230 μs，更不能从提示数量推出完整 L1 收益。

## 输入及审计边界

下列别名仅为缩短路径；JSON 为每个输入保存绝对路径和字节大小。

```text
C=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
D=$C/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
B=$D/parallel_20261018/current_clock_h131072_b16_v1/build_output
F=$B/decode_csa_spmd_jqlk5eey
O=$B/_decode_csa_tp1_layer_0leidxnz
W=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018
DEP=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
```

设备真实物理分配取自 `F/ptoas/dynamic_csa_full.pto` 的 AIC 函数，不把 AIV 同名地址混入。事件次序取自实际设备包装的 `F/kernels/aic/cube_schedule.cpp`，原 task 对照均取本次同一 `B` 目录的 `O/ptoas/*.pto` 与 `O/kernels/aic/*.cpp`。这不是比较不同作业的耗时。

最终 PTO 已不保存 pipeline group ID。本轮没有本次设备编译的逐 pass dump，因此 group 对应由两条现有文本证据交叉定位，物理地址仍完全以本次设备 PTO 为准：

1. `W/tests/pypto_test/ultra_sync_20261018/kernel_long_v1/kernel/passes_dump/32_after_LowerPipelineLoops.py` 保留 group 0–31 对应成员、嵌套和先后次序。它的 O-A 列宽/O-B 行宽与本次设备策略不同，不能拿其物理分配替代本次设备结果。
2. `D/kernel_medium_v1/kernel/passes_dump/32_after_LowerPipelineLoops.py` 的 O-B M96 分支保留 group 32/33 的内外层成员；核对本次设备 M96 循环树及源码位置后定位。该 CPU 的前段调度不同，未拿其前段 group ID 代替 long 的对应。
3. `DEP/src/ir/transforms/lower_pipeline_loops_pass.cpp:198` 先递归内层再展开当前 pipeline；每次 `ReplicateBody` 分配一个递增 group（:284），静态尾段另有 group。结合本次设备各源位置的循环树可核对两份辅助 dump 的映射。此为离线追溯，不声称取到了本次已被剥离的 group 标签。

## group 到数学阶段

下表只覆盖 AIC Left/Right。原始完整提示保留于 JSON；Vec 提示不用于本次判断。

| group | 数学阶段及源码 | 本次相关提示 | 设备物理结果 |
| --- | --- | --- | --- |
| 0 | HC 投影，`hc_stages.py:197–203` | Right 32 KiB ×2→1 | Right 单槽；Left 两槽 |
| 1–8 | QA 的 M32/M64 及各自 M16 尾块，`qkv_stages.py:126/137`；每条为自动 L0 + 外层 K | Right 每组 32 KiB ×2→1 | Right 仍 0/32768；Left 依 M 有四个地址 |
| 9 | Indexer 压缩 KV/gate 投影，`compressor_projection_stages.py:127–134` | Right 32 KiB ×2→1 | Right 单槽；Left 两槽 |
| 10 | Indexer query INT8 投影，`indexer_query_stages.py:101` | Right 32 KiB ×2→1 | Right 仍 0/32768 |
| 11 | Indexer head weight 投影，`indexer_query_stages.py:284` | Right 32 KiB ×2→1 | Right 仍 0/32768 |
| 12–19 | 原 KV 的 M32/M64 及尾块，`kv_stages.py:119/137`，自动 L0 + 外层 K | Right 每组 32 KiB ×2→1 | Right 仍 0/32768 |
| 20 | 主 QB 的后续 K256 pair，`query_projection_stages.py:144–152` | Left 32 KiB ×2→1；Right 64 KiB ×2→1 | Left/Right 均单槽 |
| 21 | 主 QB 剩余单 K256 尾段 | 无 depth2 提示 | 一个 clone，不是丢失一份请求槽 |
| 22–25 | 主压缩投影的 KV/gate、首块/累加自动 L0 分支，`compressor_projection_stages.py:96–100` | Left/Right 各组 32 KiB ×2→1 | 两侧仍 0/32768 |
| 26 | 上述主压缩投影外层 K512 pipeline，`:87` | Left/Right 32 KiB ×2→1 | 与内层复用两槽，没有四个 L0 槽 |
| 27 | Attention QK 自动 L0，`attention_stream_stages.py:156` | Right 32 KiB ×2→1 | Left 0/16384；Right 0/32768 |
| 28 | O-A 首 K256 的自动 L0，`output_projection_a_stages.py:154` | Left/Right 32 KiB ×2→1 | K256 拆两次 K128，两侧均 0/32768 |
| 29/30 | O-A 后续块的自动 L0 / 外层 K256，`:161` | Left/Right 32 KiB ×2→1 | 内层两个 K128 地址仍不同；外层复用 |
| 31 | O-A 第 15 个 K256 的单尾段 | 无 depth2 提示 | 该尾段本身仍有两个 K128 槽 |
| 32/33 | O-B M96 自动 K128 L0 / 外层 K256，`output_projection_b_stages.py:164–172` | Right 32 KiB ×2→1 | Left 四槽；Right 两槽 |

提示中的 `achieved depth=1` 是 MemoryReuse 对该 group 的容量约束状态，不是该数学阶段最终所有 SSA operand 的独立地址总数。其实现见 `DEP/src/ir/transforms/memory_reuse_pass.cpp:3869–3929`。嵌套的内层×外层各两 clone 会请求更多的隔离；例如四份 32 KiB 超过 Right 64 KiB，而因实际活跃区间仍可留下两个不同地址。必须看分配与使用次序。原 task 的 QA/KV、主压缩投影、O-A/O-B 也出现同类提示，并非融合特有。

## 三处真实差异及其依赖

所有地址单位为字节，均为对应 Left/Right 空间内的偏移。地址的绝对交换不算差异；比较的是相邻 K 片实际使用的独立槽。

| 阶段 / operand | 形状及单槽大小 | 原分 task | 当前融合 | 物理 IR 定位 |
| --- | --- | --- | --- | --- |
| HC Right | FP32 `[256,32]`，32768 B | 32768、0 | 0、0 | `O/ptoas/hc_pre_linear.pto:59,70`；`F/ptoas/dynamic_csa_full.pto:375,386` |
| Indexer 压缩 Right | BF16 `[512,32]`，32768 B | 偶 K 的 KV/gate 共用 0；奇 K 共用 32768 | 两 K 的 KV/gate 全部 0 | `O/ptoas/kv_score_proj_0.pto:78,87,98,107`；F `:815,824,835,844` |
| 主 QB Left | INT8 `[128,256]`，32768 B | 后续 K pair 使用 32768、0 | 后续 K pair 全部 0 | `O/ptoas/qproj_matmul.pto:70,75`；F `:1524,1529` |
| 主 QB Right（对照） | INT8 `[256,256]`，65536 B | 全部 0 | 全部 0 | O `:72,77`；F `:1526,1531` |

HC：F C++ `:6123` 与 `:6162` 都把 Right 绑定到常量 `v217=0`。第一次 MMA `:6135/6137` 后发 M→MTE1 event1（:6139），第二次 Right 搬运必须在 :6164 等该 event，随后 :6167 TMOV。因此相邻 K 的 Right 装入不能越过前一次消费。原 HC C++ 第一个 MMA 后只归还 event0（:368），第二组搬运等待独立的 event1（:384），在该轮没有第一 MMA→第二 Right 搬运这一边；第二组地址为另一槽。原 kernel 仍有按各槽循环复用的等待，不能称其“没有同步”。

Indexer 压缩：原 `kv_score_proj_0.cpp` 同一 K 的 KV→gate 本来就有 M→MTE1 等待（:471→486）；这不是此次新增成本。新增的是 K0 gate→K1 KV 的边：融合 F `:8113` 发 event1，`:8138` 等 event1 后 `:8141` 才搬入下一 K 的 KV；原 task 在 `:494/496` gate MMA 后，`:523` 下一 K 的 Right TMOV 不等该次 M，利用另一个地址。不能把四个 Right SSA 值误当成四槽或将两条累加链本有的 KV/gate 覆盖边全计为融合损失。

主 QB：原 C++ `:420` MMA 后，第二个 Left 在 :430 可先搬入另一槽，直到覆盖单槽 Right 时才在 :438 等 M；融合在第二个 Left 搬运前就于 F `:10772` 等 M，`:10774` 才 TMOV。Left 单缓冲确实把这份搬运串在 M 之后，但 Right 单槽两边相同，仍是不可移除的边。不能把恢复 Left 两槽预期成整个 QB 双缓冲。已有 N256/K128 显式 L0 候选也不能凭新提示重新包装为未试方向。

## 其余提示为何不证明融合退化

O-A 本次 M128/N128 的两侧都是 0/32768，与原 `proj_a_mm_0.pto` 相同。F PTO `:3899–3906` 是首个 K256 的两个 K128；`:3939–3958` 是外层 pair 的两次 K256，各自复用这两个地址。F C++ `:18621/18645` 分别绑定两个 Left 地址，`:18633/18657` 绑定两个 Right 地址；`:18665/18669` 按 K 顺序消费两组。下一外层 K 的槽复用等待的是对应旧槽（:18678、:18702），不是所有 L0 块共同地址 0。外层没有四槽，内层两槽保留；原 O-A 同样如此。

O-B M96 的 Left 地址为 0、12288、24576、36864，Right 为 0、32768；与原 `_proj_b_mm_nz_kernel__2.pto` 一致。QA/KV 的 Right 也为 0、32768；主压缩投影两侧为 0、32768。这里只确认这些阶段没有上述“原两槽、融合一槽”的差异，不宣称其现有搬运与计算已经达到理想重叠。

Score 不属于上述 group。它使用 `indexer_score_stages.py` 的显式 Left/Right tile 和手写流式顺序：Query Left 49152 B 位于 0，WS Left 12288 B 位于 49152；Right 的 WS tile 49152 B 位于 0，后续 key 的两个 8192 B 槽位于 49152、57344。首个 key 可以在 WS 尚未使用时位于 0，这不是第三个并发 key 槽。原分 task 是同样的 48 KiB WS + 2×8 KiB key 容量结构，只是 WS 在 16384、key 在 0/8192，首块曾复用 16384。没有发现 Score 双 key 槽被这些 PH-MR-001 提示降成一槽的证据。该结论只谈地址与活跃区间；不推断 A3 FFTS 的时钟归属或 MTE 指令实际完成时刻。

主会话的最新 `phase_summary.json` 提供另一个待解释现象：Score AIC 段约 230 μs，而 AIV 在 FFTS 附近记录了较长停顿。此前“wait_flag_dev 必然只有发射耗时”的说法已撤回。阻塞语义的独立证据见另一 worktree 的 `tests/pypto_test/ultra_physical_plan_20261018/A3_FFTS_WAIT_TIMING_READONLY.md`；本审计没有重新验证计时包裹的机器码，也没有把名为 `ffts_issue_us` 的字段解释成无阻塞发射成本。现有 buffer 证据不足以将 Score 的量级成本归因于本次提示，继续观察 AIC 供数/计算服务更有根据。

## 有界的源码内修复方向与历史排除

若以后只修这类融合降槽，一条最小方向是 **Indexer 压缩投影的 K512 pair 内显式 Right 预取活跃期**，位置为 `compressor_projection_stages.py:127–134`。保留 M16/N32/K512、每链 K0..K7 顺序、原 valid rows、init_cond、输出和核间协议，只将每个 pair 的低层 operand 搬运显式化：先形成 K0/K1 两份 KV Right，在 K0 MMA 消费前让两份同时存活；随后仍按 `KV0→gate0→KV1→gate1` 提交四次 MMA。K0 gate 可复用已消费的 K0 Right，K1 Right 在此前保持存活。两份 32 KiB 正好占满 Right 64 KiB，Left 仍用原两份 16 KiB。这样给 allocator 真实的重叠活跃区间，而非再加一个 `stage=2` 标签。

此方向尚未实现或编译，可能因全函数资源规划失败而不能成形；须先核实生成地址和事件。它只针对上述实际新增的 K0 gate→K1 KV 覆盖边，最多隐藏相应的 L1→L0 搬运，不能声称减少 MAC、GM 字节、Scalar metadata、软件通知或整个 Score 的 230 μs。这里不推荐抢在已有 compact 阶段重排之前测试，也不提出第二个候选。

已查 `C/vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md` 的相关历史，避免重开：

- `:8680–8716`：主 Q 普通 N128 未得到第二 Right 地址，未上卡；N256/K128 显式 L0 双缓冲已验证实际地址和完整精度，但未被采用。本次不再提出它。
- `:14704` 附近：O-A pair 共享 Left 的资源探针出现 Right 容量问题，未接入整链；本次已看到原形 O-A 两槽仍在，不以同一个 hint 再提出 pair/retile。
- `:14718–14738`：HC 紧凑 GM 双槽候选改变供数及 K512 组织，未获得现选同进程净收益；它尚无独立 HC_pre 的显式 L0 交错。本次确认的是当前 K256 HC Right 单槽，未把该事实视作复活 HC 整体双槽/整行驻留的理由。
- `:8746` 与独立 `hc_pre.py:230–245` 确实提供“用真实交错存活期保住两槽”的写法依据，但不能直接照抄其 M/K、核间 FIFO 或归约方式。

源码重排会改变 group 遍历编号，且可能改变全函数分配；新 compact 重排候选完成生成后可直接复核以上几个地址与依赖，不能默认候选只改变阶段端点。此提醒不构成编译/设备任务的额外前置审批。

## 复现

```bash
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018
python tests/pypto_test/ultra_pipeline_depth_audit_20261018/extract.py
```

输出包含 22 个输入文件的路径/大小、218 个当前 AIC Left/Right 分配记录、13 个同次原 task kernel 的分配、完整降深度提示、两份辅助 CPU 的 group 成员和八段实际 C++ 证据。脚本不导入 PyPTO，不加载模型，不调用 task-submit。原始验证、计时和编译文件均保持原样。
