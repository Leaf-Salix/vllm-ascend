# DSV4 Flash CSA：完整执行清单

本文把当前所有待做事项整理成可逐项执行的清单，供后续以目标模式驱动。
每项给出落点、完成判据、依赖和占卡情况；完成判据写成可判真假的形式，
不写"验证一下""确认无误"这类无法判定的措辞。

状态口径：`未开始` / `进行中` / `已完成` / `暂停`（暂停项不得自行恢复）。
截至 2026-09-24，已完成 T1.1～T1.9、T2.1、T2.2、T3.1 与 T5.1～T5.4（共 18 项）；T1.10 低优先级、T2.5 待用户拍板。**T1 的 padding 主线至此全部走通。**

相关文档：[padding 开发计划](DSV4_FLASH_CSA_PADDING_PLAN.md)、
[跨会话交接](DSV4_FLASH_CSA_NEXT_SESSION_HANDOFF.md)、
[验证计划](DSV4_FLASH_CSA_VALIDATION_PLAN.md)、
[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)、
[离线 P/D 方案](DSV4_FLASH_CSA_OFFLINE_PD.md)。

## 0. 每项开工前都适用的约束

这些是用户的长期要求，不随单项任务改变：

- 不自行修改 PyPTO、Simpler、PTOAS、PTO-ISA 的计算或运行时实现；已有的本地环境差异如实保留。
- 所有 NPU 测试走 `task-submit` 队列；不绕过队列、不停止他人任务。
  **轮询一律用 `task-submit --status <id>`，绝不要对排队中的任务用 `--wait`。**
  `--wait` 默认 600 秒超时，超时会把尚未运行的任务直接取消——
  `task_20260924_012911_128588925572` 就是这样被取消的（状态从 pending 变
  not_found，三组验收一个都没跑），白丢一轮排队。等待请用
  `until task-submit --status <id> | grep -qE "completed|failed|cancelled|timeout"; do sleep 30; done`。
- 不执行新的 hash／摘要校验；记录路径与大小，并做必要的数值比较。
- 沟通、commit 说明、新增说明性注释一律用中文；提交带 `Signed-off-by`。
- 不做提交检查、不自动运行格式化、全量测试或提交钩子。
- `tests/` 之外原则上只放足够精简的 PTO 算子与适配代码；测试、诊断、过程记录都放 `tests/` 下。
- 大权重、`.pt`/`.safetensors`、`.bin`/`.so`/`.o`、安装包和重复编译产物留本地不提交；
  不要 `git add .`（`build_output/` 约 114MB 未提交）。
- 模型测试一律使用正式 W8A8，不再使用 48 分片 cann_recipe 参考权重。
- **NZ 当前一定不能开**：`weight_nz_mode=0`、`enable_kv_nz=false`、`VLLM_ASCEND_ENABLE_NZ=0`，
  Native 侧也一样；上线脚本里的 `VLLM_ASCEND_ENABLE_NZ=2` 不要照搬。
- **decode 性能测试一律用 ACL Graph `FULL_DECODE_ONLY`**，不用 eager；
  eager 只用于定位问题，其结论不代表上线表现。
- 不过度测试：失败先定位，只重跑受影响项。

**运行入口**：所有涉及 vllm／torch_npu 的命令都必须先
`source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh`，
它负责激活 `.venv-dsv4-0251rc1` 并设置 CANN 9.0.0、PTOAS、ATB 与
`ASCEND_CUSTOM_OPP_PATH`。系统默认的 `/data/server-toolkits/miniconda3/bin/python`
里没有 vllm，直接用它提交会在 `from vllm import LLM` 处失败
（`task_20260924_004251_10658422228` 即因此白跑一轮）。

## 1. T1　padding 支持（当前主线）

方案见 [padding 开发计划](DSV4_FLASH_CSA_PADDING_PLAN.md)。用户已于 2026-09-23 指派恢复。

| ID | 目标 | 落点 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| T1.1 | CPU 复算四处索引，取得越界证据 | `tests/pypto_test/dsv4_csa_padding_probe.py` | 已完成：真实请求四处全部在界内，补位请求在 compact 行号上恒越界，页表类在陈旧 position 超容量时越界。初版按补齐后 token 数算出 10 行，T1.2 实测为 8 行，公式已更正。证据 `results/release_csa_padding_20260923/padding_probe_v1/{uniform,mixed}/` | — | 否 | **已完成** |
| T1.2 | 设备侧确认 compact metadata 真实形状，并定夺有效性判据 | `offline_pd/observer.py` 的 `offline_begin/end_padding_capture`，`offline_pd/run.py` 的 `padding-capture` 命令 | 已完成，任务 `task_20260924_001111_370250932735`：compact 行数实测 8（初版预测 10，公式已更正）；补位请求 `seq_lens=0`、`start_pos=0`、页表行全零；补位段 positions 实测为上一步残留；据此选定方案 C（`seq_lens == 0`），D 因新请求 `start_pos` 同为 0 而有歧义 | T1.1 | 16 | **已完成** |
| T1.3 | 加入设备端有效性判据并改四处索引 | 同左四个文件 | **代码已完成**（`4b40896`）：判据取 `kv_seq_lens[b] == 0`，四处均只把已有 `cmp_seq_lens`/`kv_seq_lens` 传入子函数，顶层签名不变（52 参数），无新增入参与缓冲；与 Native 的对照结论写入提交说明；CPU 全链 lowering PASS。**数值验收已通过**（`task_20260924_024451_206700130649`）：同一负载分别按档位 `12 30`与 `18 30` 跑，前者 256 次 build 中 248 次带补位、后者 152 次，补位量相差 96 次，而两轮输出**逐 token 完全相同**。另修复图捕获时 dummy run 的 compact 行越界（见 T1.Q2） | T1.2 | 16 | **已完成** |
| T1.4 | 放开三道 host 闸门 | `native_adapter.py`、`service.py`、`service_config.py`、`platform.py` | **代码完成，验收进行中**（`1815fac`、`3dfb547`）。三道闸门已放开；另发现并修复第四个阻塞——ACL Graph 档位未按 `uniform_decode_query_len` 对齐，导致 MoE 退到 ALLTOALL 使 `should_skip_allreduce_across_dp_group` 为假、触发 DP 闸门。判据：小 BS 放进较大合法 bucket、不再静默回退 Native。**已通过**：PTO 在图模式下完整跑通、输出与 Native 逐 token 相同（`accept_t14_pto_v5`）；补位档位全部进入图重放（`accept_t13_padded`：`replay_padded=62`、`allowed=62`、`rejected=0`），不再静默回退 Native | T1.3 | 16 | **已完成** |
| T1.5 | graph 覆盖 G04～G06 | 由 D01～D05 在 16 卡离线 D 上覆盖，见下方对应关系 | **已完成**：五条判据全部有实测证据。G04／G05／G06 与"同图不同补位量重放不串数据"由 D01～D05 覆盖；最后一条"无 replay 期重新编译"由 `d05_recapture` 实测——采集窗口内统计 `torch.npu.NPUGraph` 新建次数，**16 个 rank 合计为 0**，且该轮用 `--stagger` 让两 rank 跨越不同档位序列（重放 13 vs 20），是最容易触发重新捕获的场景。落点之争已无实际意义：单卡链已死，要验的内容在 16 卡上都验到了 | T1.4 | 16 | **已完成** |
| T1.6 | 空 rank 整批 dummy | `offline_pd/run.py` 的 `--rank-decode-tokens`、`observer.py` 的 dummy／slot／compact 探针 | **已完成**，四条判据全部有实测证据。①空转已证实：rank0 比 rank1 多跑 21 次 dummy（26 vs 5），直接计数；②无越界读：图捕获时的 MTE 越界已修；③输出无非有限值且不影响其余 rank：与 Native 逐 token 相同；④**不写任何真实 cache／state**——主 slot 路径六个 cache group 全为 -1（`accept_t16_slots_v2`），compact 路径经复算（`compact_slot_v2`）只有 2 行、其中 1 行为 -1、另 1 行页号为 **0**，而 0 号页是 vLLM 保留的 null block（`block_pool.py` 初始化即取走并标记 `is_null`，永不分配，见 T1.Q1），两 rank 三次采样一致 | T1.4 | 16 | **已完成** |
| T1.7 | D01～D05 的 DP 验证 | `offline_pd/run.py` 的 `--rank-batches`／`--rank-decode-tokens`／`--stagger` | **已完成**。六组按上线口径（batch 32、`HCCL_BUFFSIZE=1800`、DP 闸门已移除）全部通过，`rejected` 无一例外为 0，且**输出与 Native 逐 token 完全相同，16 个 rank 无一例外**。D01 (4,32)／D02 (32,4) 补位量随负载对称反转；D03a (8,24) 与 D03b (16,32) 补位量完全相同（208/104），说明补的是到全局最大值的差额、与自己提交多少无关；D04 用 `--rank-decode-tokens 16 64` 造真实空转，rank0 `dummy_runs=26` vs rank1 的 5，是直接计数而非耗时推断；D05 用 `--stagger` 让两 rank 跨越不同档位序列（补位 96 vs 80、重放 13 vs 20），同一张捕获图在不同补位量下反复重放无串数据、无重新编译 | T1.9 | 16 | **已完成** |
| T1.8 | DP16 完整验证 | 离线 P/D 入口 | **已完成**：本轮 D01～D05 即在 DP16／EP16 下跑的（驱动硬性 `tp=1, dp=16`），见 T1.7。通信选择记录：档位按 `uniform_decode_query_len` 对齐后 `mc2_tokens_capacity` 与 `potential_max_tokens` 相等，A3 选中 MC2 | T1.7 | 16 | **已完成** |
| T1.9 | 拿掉 DP 图模式闸门 | `service_config.py` | **已完成**（`05ba642`）。顺序按用户 2026-09-24 的决定提前：原计划 T1.8 通过后再删，用户明确"目标肯定是支持 DP 补齐场景的 aclgraph，放开后遇到问题解决具体问题"。移除后 `_sync_metadata_across_dp` 真的 all_reduce，DP 补齐随之产生，D01～D05 才验得到真实场景并全部通过 | — | 16 | **已完成** |
| T1.10 | eager + embedding_tp 的 DP 补齐 | **已验证，结论是跑不起来**（非集成缺陷，见下方专节）。判据中「`_forward_embed_tp` 的静态缓冲容量不被超出」一条的答案是**会被超出**：`ValueError: embedding_tp static capacity 192 < num_tokens 256`。**PTO 与 Native 两侧同样失败**，与 CSA 用哪套算子无关。另确认 `embedding_tensor_parallel_size` 在框架层面强制要求 `recompute_scheduler_enable=true`，校验信息写明「跨 DP 的 HCCL 集合通信需要各 rank token 数一致」，这从侧面印证了该项「embedding TP 会引出 DP 补齐」的前提 | T1.9 | 16 | **已验证（阻塞于容量口径）** |

用户已指定：**T1.7 的 DP2 必须先跑完再上 T1.8 的 DP16。**

### embedding_tp 为什么会在 eager 下引出 DP 补齐（T1.10 的背景）

`allow_dp_padding` 的四个条件里有 `or embedding_tp_enable()`，**它与 cudagraph
无关**。原因在建组逻辑（`distributed/parallel_state.py` 的 `_create_or_get_group`）：

```python
rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                    ↑ 切的是 DP 维
```

**embedding TP 组是沿 DP 维切的**，不是沿 TP 维。TP=1／DP=16 且
`embedding_tp_size=4` 时，rank{0,1,2,3} 一组、{4,5,6,7} 一组，组内成员是不同的
DP rank。而 `_forward_embed_tp` 在组内做 **all_gather + reduce_scatter**，
要求组内每个 rank 贡献的 token 数完全一致，否则拼接偏移与切分边界对不上。
所以必须先把这些 DP rank 的 token 数补齐——这就是那个 `or` 的由来。
`oproj_tp_enable` 同理（`_OTP` 用同一个 `_create_or_get_group`）。

连带一条：`_forward_embed_tp` 的静态缓冲按
`capacity = get_potential_max_tokens()` 分配，超了直接 `raise`。
`potential_max_tokens` 正是档位对齐时动过的那个量，**两者是同一个来源**——
若开 embedding TP 而档位配置不当，会直接在这里报错。

当前 `embedding_tensor_parallel_size` 为 0（未开），所以这条路径现在遇不到。

### 生产口径暴露的第一条真实约束：HCCL 缓冲

换成 `--batch 40` 后第一轮直接失败在 MoE 的 MC2 派发算子上，**不是 CSA 的问题**：

```
npu_moe_distribute_dispatch_v2 -> aclnnMoeDistributeDispatchV4，错误码 561002
HCCL_BUFFSIZE_EP is too SMALL, maxBs = 240, h = 4096, epWorldSize = 16,
localMoeExpertNum = 16, k = 6
NEEDED = ((maxBs*8704*16*16) + (maxBs*8192*6)) * 2 = 1043MB, HCCL_BUFFSIZE = 1024MB
```

`maxBs = max_num_seqs * 6`。此前一直用 `--batch 5`（maxBs=30），需求约 130MB，
远在限内，所以从没碰到——**小 batch 把这条真实约束整个绕开了**。

查上线参考（`dsv4_perf_accuracy_20260827/runtime`）：decode 侧
`HCCL_BUFFSIZE=1800`、`--max-num-seqs 32`；prefill 侧 1024。按 32 反推
maxBs=192、需求约 834MB < 1800MB，**上线配置自洽**。驱动已改为 prefill 1024、
decode 1800，与上线一致。

**已定**：用户 2026-09-24 定「以 32 为主验收」，T3.2 的档位表去掉 B=40。

### 验收口径与 DP 补齐的定位（2026-09-24 用户定）

**一、验收一律用生产口径，不用小 batch 图快。**
上线参考脚本是 `--max-num-seqs 32`，用户 2026-09-24 定「以 32 为主验收」；
性能数据另有专门指标（见第 2 节，B=16 / seqlen 8k）。
小 batch 会掩盖问题：`max_num_seqs=5` 时对齐后的档位 `[6,12,18,24,30]` 是稠密的，
每个 batch 精确命中、档位补齐根本不发生——本轮那条"档位对齐消除了补位"的错误
结论就是这么来的。而 `max_num_seqs=32` 对应的档位表是稀疏的，补位照常发生。

**二、DP 该补齐的就补齐，否则性能 GAP 全落在 MoE 的集合通信上。**
这条纠正了"补位是额外开销、能省则省"的直觉：各 rank token 数不齐时 MC2 没法按
统一形状走，代价转嫁到 MoE 的集合通信，反而更贵。**跳过 DP 同步不是优化**，
补齐才是生产该走的路。据此 T1.9 的闸门已提前移除（见上表）。

**三、eager／非 aclgraph 路径也要补用例验证。**
本轮的改动——kernel 的 `seq_lens` 守卫、compact 行号兜底、三道 host 闸门放开、
档位对齐——同样会走到 eager 路径，不能只验图模式。eager 下
`allow_dp_padding` 因 `cudagraph_mode == NONE` 而为假，也不注册捕获档位，
**结构上不产生任何补位**，所以要验的是回归：补位守卫在无补位时是否彻底 no-op、
输出有无变化。已排 `task_20260924_115638_182880813886`：eager 下 PTO、Native
基线、以及 PTO 不均衡负载三组，均用 `--batch 40`。

### 档位为什么必须是 6 的倍数（2026-09-24 定论）

这条解释了 T1.4 遇到的第四个阻塞，也回答了"S 恒为 6 为什么还会有形状问题"。

**padding 不发生在 S 这一维，而在请求数那一维。** 一步 decode 的 token 总数是
`batch × 6`；S=6 来自 `DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS`，从不变动。变的是
batch，而 ACL Graph 要固定形状，所以把 batch 补到最近的档位，补的是**整条假请求**
（实测 `query_start_loc = [0,6,12,18,24]`，第 4 条 `seq_lens=0`）。

**但档位是纯 token 计数。** Native 的布局是 TND——`model_runner_v1.py` 里
"when the layout is TND, the first dimension of hidden_states must equal the last
element of actual_seq_lengths_q"——档位只有 T 一个数字，B 和 S 压扁在一起。
vLLM 默认档位来自通用列表（实测 `[1,2,4,8,16,24]`），与 6 无关；vllm_ascend 里
唯二调整它的地方（950 等距抽样、序列并行按 TP 对齐）也都不按 6 对齐。

档位不是 6 的倍数时，`_pad_query_start_loc_for_fia` 走混合分支，**插入一条
长度为剩余全部 token 的 dummy 请求**：档位 16、2 条真实请求会得到
`query_start_loc = [0, 6, 12, 16]`，最后一条长度 4。此时 `s_dim = 16 // 3 = 5`，
PTO 的所有索引全错——不只是尾部那条。

**处置：修档位，不改 kernel。** pypto-lib 的参考实现把两条路径有意分开——
prefill（`prefill_compressor_ratio4.py`）用 `query_start_loc` 走真 TND，
decode（`decode_compressor_ratio4.py` 等）用 `s_dim = bs // b_dim` 走等长 S，
因为 decode 的 S 恒为 6。把 PTO decode 改成 TND 会与参考实现分叉，收益仅限于
一种上游本可避免的形状。用户 2026-09-24 定：**vllm_ascend 的档位设计不合适，
应贴近 DSpark 的 6 的倍数**。已在 `platform.py` 按序列并行那段的既有写法实现，
并留 `align_decode_capture_sizes` 开关（默认开）以便造反例场景。

### 对齐后补位还在不在：**在**，先前的相反结论已更正

这一节曾写成"档位对齐消除了档位补齐、两条补位来源同时没了"。**那是测试配置
的产物，不成立**，2026-09-24 已更正。

对齐逻辑是把默认档位按 6 向上取整后去重，再补上
`min(max_num_seqs*6, max_num_batched_tokens)`。**得到的档位表是否稠密，
取决于 `max_num_seqs`**：

| `max_num_seqs` | 对齐后档位 | 需补位的 batch |
| --- | --- | --- |
| **5**（本轮测试用） | `[6,12,18,24,30]` | **0/5**，全部精确命中 |
| 8 | `[6,12,18,24,36,42,48]` | 1/8 |
| 16 | `[6,12,18,24,36,42,48,60,66,72,84,90,96]` | 3/16 |
| **40**（生产口径） | `[6,12,18,24,36,42,...,240]` | **9/40** |

生产 batch 下档位表是**稀疏的**（缺 30、54、78…），batch 5→36、9→60、13→84
都要补位，**档位补齐照常发生**。此前 `accept_t14_pto_v5` 测到 `padded_builds=0`，
是因为那轮 `max_num_seqs=5` 恰好落在稠密区间，不能外推。

旁证：`max_num_seqs=40` 算出的档位正是 `[6,12,18,24,36,42,48,...]`，与 probe 里
那份"取自 native_dp_v1 的 DP2 实测档位"完全一致，说明真实 DP2 运行早就是这个形状。

**所以 T1.3 的 kernel 判据与 T1.4 的闸门不是防御性死代码**，生产路径上会真的走到。

仍然成立的一条：DP 闸门放行的前提 `should_skip_allreduce_across_dp_group == True`
会让 DP 同步被跳过，因此**DP 补齐**这一条来源在 T1.9 之前确实不产生；
验收时用 `--capture-sizes` 显式构造档位补齐即可，不必等 T1.9。

顺带修好了 MoE 通信选择：`mc2_tokens_capacity` 取自最大档、
`potential_max_tokens` 取 `max(最大档, max_num_seqs*6)`，原先 24 与 30 不等
使 A3 退到 ALLTOALL，直接触发 PTO 的 DP 闸门；对齐后两者相等，MC2 得以选中。
**这是档位对齐的第二个、与形状无关的理由。**

### T1.2 的取证方式：挂在真实生产路径上

用户 2026-09-23 定：**一切以当前 release 的生产路径为准**，取证走真实离线 D，
不复活旧的单层 fixture。旧 fixture 依赖的 `enable_device_metadata`、
`take_device_metadata_tasks`、`DeviceMetadataExecutor` 在当前 release 中均已删除，
整条链在 import 阶段即失败；按上述口径不予适配，也不作为参考。

挂载点最终选 **`AscendDSAMetadataBuilder.build`**。最初挂 `CSAServiceRuntime.eligible`
是错的，有两个问题：它只在 PTO 后端存在；而且 `can_replay_csa_graph` 一旦发现需要
补位就返回 False，使该步回退 eager 并拿到未补齐的 `BatchDescriptor`，等于把要观察的
padding 自己消掉了。builder 两个后端都会走，不受 CSA 闸门影响。

compact 行数直接读 `decode.num_compressed_tokens`，不额外调用 `compressor_metadata`
算子——那需要与当前 builder 同一层的 impl，取错层会因 `compress_ratio` 不匹配而报错
（`task_20260924_000518_3583841341` 即因此失败），也会扰动本步。

索引复算在 CPU 侧离线做，与 `profile`／`profile-export` 的分工一致。

### 已解决：本机图模式此前无法运行

2026-09-23 两轮 16 卡采集的结论，**影响 T1.2 之后的全部图模式工作**：

| 任务 | 配置 | 结果 |
| --- | --- | --- |
| `task_20260923_233825_305642829457` | `--backend pto --graph-mode eager --batch 4` | exit 0，32 步全部无补位 |
| `task_20260923_234823_326614713564` | `--backend native --graph-mode full_decode_only --batch 5` | exit 1，初始化即失败 |

第一轮证明 **eager 结构上不产生补位**：`allow_dp_padding` 取决于
`cudagraph_mode != CUDAGraphMode.NONE`，eager 下为 False，各 rank 保留自己的
token 数；且 eager 不注册捕获档位。所以补位取证必须在图模式下做。

第二轮暴露图模式本身跑不起来：

```
RuntimeError: Worker failed with error 'aclnnAddRmsNormBias or
aclnnAddRmsNormBiasGetWorkspaceSize not in libopapi.so, or libopapi.so not found.'
```

疑似成因（**未验证，勿当结论**）：`vllm_ascend/utils.py:423` 的
`if not torch.compiler.is_compiling(): bootstrap_custom_op_env()`
在图编译期间跳过 bootstrap，导致 libopapi.so 未加载；eager 下首次调用发生在
编译之外，bootstrap 正常执行。`vllm_ascend/ops/layernorm.py:73` 与 `:100`
在 `enable_custom_op()` 为真时才走 `npu_add_rms_norm_bias`。

交接文档记载本轮 D16 一直是 eager，**本工作区没有任何图模式成功运行的记录**，
与该现象一致。

**已于 2026-09-24 修复。** 实测确认 `aclnnAddRmsNormBias` 在基础 CANN 9.0.0 的
`libopapi.so` 和已构建的 CSA 自定义算子包里都不存在，先前"bootstrap 时序"的猜测被证伪。
真正触发路径是 torch 的 pattern matcher 以 `tracing_mode="real"` 追踪融合 pattern，
等于真的执行一次 `norm_quant_fusion_pass.py:61` 里的 `npu_add_rms_norm_bias`，
于是图编译在建 pattern 阶段就崩；eager 不建 pattern 故从未暴露。

修法是配置开关，不改生产代码：`graph_fusion_pass_manager.py:54` 以
`ascend_compilation_config.get("fuse_norm_quant", True)` 控制该 pass，
测试驱动在图模式下将其置 false。`task_20260924_001111_370250932735` exit 0，
本机首次跑通图模式。

**该项偏离上线口径**：参考脚本所在环境具备该算子、融合为开启状态，本机关闭它
意味着图模式性能不直接等同于线上，T2 的性能对照必须注明这一点。

### T1.5 的判据如何被 D01～D05 覆盖

原计划在单卡上验 G04～G06，但单卡全链 fixture 已死——`dsv4_csa_service_dynamic.py`
导入 `dsv4_csa_native_fixture`，后者调用的 `enable_device_metadata` 与
`take_device_metadata_tasks` 在当前 release 中已删除，整条链 import 即失败，
按既定口径不复活。落点之争最终没有实际意义：要验的内容在 16 卡离线 D 上都验到了。

逐条对应：

| T1.5 判据 | 覆盖它的证据 |
| --- | --- |
| **G04** 六档 BS 之间切换，记录图重选 | D05 用 `--stagger` 让活跃 batch 逐档下降，两 rank 跨越的档位序列不同（`replay_padded` 13 vs 20） |
| **G05** 小 BS 放进较大合法 bucket，检查 padding 与 dummy request | D01～D03：低负载 rank 提交 4～16 条却按 32 条的形状跑，`padded_builds` 208/256 |
| **G06** 请求换位、退出、新增及合法页复用 | D05 的 stagger 使请求在不同步数陆续退出；D04 造出真实空转（rank0 `dummy_runs=26` vs rank1 的 5） |
| 同一张图在不同补位量下重放，metadata buffer 复用不串数据 | D05 两 rank 补位量 96 vs 80、重放次数 13 vs 20，`rejected=0`，且输出与 Native 逐 token 相同 |
| **无 replay 期重新编译** | `d05_recapture`：采集窗口内统计 `torch.npu.NPUGraph` 新建次数，16 个 rank 合计 **0**。窗口开在预热与捕获之后，窗口内不再新建即无重新捕获。测的是行为而非耗时尖峰 |

### T1.8 的执行配方（DP16 六组负载）

驱动已支持按 rank 指定实际提交数（`--rank-batches`，2026-09-24 加入）。
`max_num_seqs` 统一取 `--batch`，所以 `--batch` 要给成各 rank 里的最大值；
不足的 rank 用 `--rank-batches` 逐个指定，其余 rank 按最后一个值补齐。

统一前缀（`B` 为 bank，`R` 为结果根目录）：

```
COMMON="--bank $B --graph-mode full_decode_only --decode-tokens 64 --recompute-scheduler --backend pto"
```

| 用例 | 负载 | 命令追加 |
| --- | --- | --- |
| D01 | `(4,40)` | `--batch 40 --rank-batches 4 40` |
| D02 | `(40,4)` | `--batch 40 --rank-batches 40 4` |
| D03a | `(8,24)` | `--batch 24 --rank-batches 8 24` |
| D03b | `(16,32)` | `--batch 32 --rank-batches 16 32` |
| D04a | `(0,4)` | `--batch 4 --rank-batches 0 4` |
| D04b | `(0,40)` | `--batch 40 --rank-batches 0 40` |
| D05 | 连续切换 | 依次跑上述各组，比对图重选与 metadata buffer 复用 |

D04 的 rank0 提交数为 0：不提交任何请求但仍参与 DP 集合通信，
这既是 D04 的空 rank 路径，也是 T1.6 整批 dummy 的前提。

每组都要先记录 `should_skip_allreduce_across_dp_group` 的实际返回值、
通信方法与图模式，再判定预期 padding 量——清单 T1.7 的判据已有此要求，
DP16 同样适用，不能用 DP2 的结论替代。

### 空 rank 的 dummy 是否写 cache：**探针无效，结论全部撤回**

这一节记录一次失败的测量，保留它是为了避免后人重走。

原本要验 T1.6 的"不写任何 cache／state"。做法是在 dummy 步前后克隆该层的
六个缓存视图并逐元素比较。先后修过两轮缺陷（克隆前补同步、跳过前 18 次 dummy
以避开两个 rank 都有的早期步骤），一度得出结论：四处视图被写，其中
`compress_state` 只有 PTO 写、Native 不写。

**对照实验推翻了整套测量**（`accept_t16_stepcontrol`）。用同一套前后对比逻辑
去量 `execute_model`，钩到的前两次恰好是**空闲步——0 个请求、0 个 token**，
却显示 `cmp_kv` 与 `swa` 各有 114 页、`compress_state` 有 62 页发生变化。
**一个什么都不算的步骤不可能写 114 页**，所以这个前后差异根本不能归因于被测
步骤。

因此：

- 页数、页号、`only_null_block` 判定全部作废；
- **那条"`compress_state` 只有 PTO 写"的二值结论同样作废**——对照组显示
  PTO 的 `compress_state` 在 0-token 步上也会变（62 页／31 页），而没有
  Native 的同类对照数据；
- **T1.6 的"不写任何 cache／state"既未证实也未证伪**。先前写的"已被实测
  证伪"不成立，已删除。

**失效原因已查实（不是推测）。** 读 `decode_cache_layout_v1/rank0.cache_layout.json`
——那是 `offline_cache_layout` 早先采到的真实描述符——CSA 的缓存视图**大面积
共用同一块存储**：

| 重叠对 | 重叠字节 | storage_pointer |
| --- | --- | --- |
| `cmp_kv` ↔ `compress_state` | 676,560,896 | 同为 `20733286793216` |
| `inner_compress_state` ↔ `idx_kv_cache` | 85,891,520 | 同为 `20730199080960` |

`cmp_kv` 与 `compress_state` 是同一块 678MB 分配上的两个视图，偏移分别为
466944 与 233472，各自都覆盖 676MB，几乎完全重叠。

由此三点全部解释通：

1. **任何一处写入都会让多个视图同时"变化"**，因为它们本来就是同一段内存，
   所以"四处视图被写"是假象；
2. **"`compress_state` 只有 PTO 写"是重叠的产物**——写 `cmp_kv` 就会让
   `compress_state` 显示变化；
3. **0-token 步也变**：该分配覆盖所有层的块，而 0 个请求的步骤仍会为 DP 协调
   跑一次内部 dummy 前向，所有层都写，于是全部视图一起显形。

**结论：在这套存储布局下，用"视图是否变化"去归因写入根本不可能成立。**
本节的页数、页号一律不要引用。

### 换成测 slot mapping 之后的结果（有效测量）

改测**输入**而非输出——若 slot mapping 全为 -1，kernel 的 `page >= 0` 守卫
必然挡住，与存储布局无关。中间还绕过一个坑：hook `CSAServiceRuntime.__call__`
在 dummy 步上一次都不触发（实测 `dummy_runs=26` 而 `slot_samples=0`），
因为 6 token 的 dummy 被派发到 12 档做**图重放**，而图重放不跑 Python 前向闸门。
改为在 dummy 之后直接读常驻缓冲——图重放读的就是这些固定地址。

**实测（`accept_t16_slots_v2`）：六个 cache group 的 slot mapping 全部为 -1**
（每组 266 个元素、非负 0 个、max 为 -1）。所以 `model_runner_v1.py` 那句
`slot_mapping.gpu.fill_(-1)` 确实生效，**主 slot 这条写入路径在 dummy 步上
写不进去，已实测确认**。

**仍未覆盖的一条，且捷径已被否掉**：compact slot mapping
（`cmp_slot_mapping`、`idx_slot_mapping`）由 `compressor_metadata` 算子在图内从
`start_pos` 与 `block_table` 现算，不来自被 fill 成 -1 的缓冲。图内产出的张量
Python 侧读不到，于是改读它的**输入** `block_table`，本想论证"全零 → compact
slot 指向 0 号 null block → 无害"。

**实测否掉了这条推理**（`accept_t16_blocktable`）：dummy 步的 `block_table`
**不是全零**，保留着已结束请求的真实页号——六个 group 的非零项分别为
3/15、1/5、7/55、7/55、52/880、28/220，最大页号 115～144。主 slot 仍全为 -1
（那条结论稳），但 compact slot 完全可能算出真实页。

所以这条路径**既没被证明无害、也没被证明有害**。要判真假，只剩两条路：
读图内产出的 compact slot 张量本身，或者比较非重叠的存储区间。

一条未验证的旁证：Native 的 `compressor` 算子同样吃 `state_block_table` 与
`start_pos`，输入一致，行为多半相同——但这是推断，不是测量，不能当结论。

### T1 的待确认问题

动手前需实测，不能凭推算下结论：

| ID | 问题 | 归属 |
| --- | --- | --- |
| ~~T1.Q2~~ | **已答且已修**：`_dummy_run` 在图捕获时把所有 `positions` 填成 127、`seq_lens` 填非零，于是 `(127+1)%4==0` 成立、T1.3 的 `seq_lens>0` 守卫放行，推出的 compact 行号 32 远超该档的 12 行——设备实测报 `MTE instruction DDR address out of range`，PTO 首次图捕获即崩。日志第 81 节"该用例曾通过"的矛盾也因此解开：那次 PTO 根本没执行。已按 compact 表的真实行数（动态维）在三处兜住，`88f59bc` | 已闭环 |
| ~~T1.Q1~~ | **已答**：vLLM 把 `block_id=0` 保留为 null block（`vllm/v1/core/block_pool.py:188`），初始化时从空闲队列取走并标记 `is_null`，永不分配给任何请求。补位页表行读到的是该保留页，不会串到其他请求的数据 | 已闭环 |
| T1.Q3 | 补位 token 的 attention 输出会不会带 NaN/Inf 进 MoE。跳过 DP 同步时不传 `mc2_mask`，补位 token 会真的进入专家路由 | T1.7 |
| T1.Q4 | 放宽闸门后 `num_reqs_actual` 与 `num_decodes` 的实际关系 | T1.4 |
| ~~T1.Q5~~ | **已答**：真实 runner 按 `cdiv(max_model_len, block_size)` 分配页表列（`vllm/v1/worker/gpu_model_runner.py:7039`），比单层 fixture 宽。T1.2 实测样本中 A／D 两处补位请求均在界内，读到的是 0 号页 | 已闭环 |

## 2. T2　性能对照

### T2.1 的结果与归因（2026-09-24）

配置：b=16 / s=6 / TP1 / EP-DP16 / seqlen 8k，3 个稳态 decode step。
b=16 在 KV 并发上限 22.8 之内，所以是干净的满批稳态（96 token/16 请求出现 21 次），
不像 batch 32 那轮被调度器拆成 21+11。

| | Native | PTO | 差异 |
| --- | --- | --- | --- |
| 设备侧总耗时 | 240,013 µs | 347,302 µs | **PTO 慢 45%** |
| kernel 记录数 | 7,520 | 5,378 | PTO 少 28% |

按引擎（µs）：

| 引擎 | Native | PTO | 差异 |
| --- | --- | --- | --- |
| **AI_CPU** | 1,270 | **81,318** | **+80,049** |
| MIX_AIC | 91,468 | 141,966 | +50,498 |
| AI_CORE | 33,407 | 22,929 | −10,477 |
| AI_VECTOR_CORE | 48,768 | 37,601 | −11,166 |
| MIX_AIV | 65,101 | 63,487 | −1,614 |

**PTO 替换掉的 Native 算子确实消失了**：`Compressor_*`（8,104µs）、
`SparseAttnSharedkv_*`（6,685µs）、`VllmQuantLightningIndexer`（3,837µs）
在 PTO 侧均为 0，合计约 18.6ms；矩阵乘类也更快
（`QuantBatchMatmulV3` −10,305µs、`TransposeBatchMatMul` −6,079µs）。

**代价是两个 Native 完全没有的条目**：

| kernel | 次数 | PTO 耗时 |
| --- | --- | --- |
| `simpler_aicpu_kernel_exec_*` | 63 | 79,825 µs |
| `aicore_kernel_mode_0_mix_aic` | 63 | 78,582 µs |

63 = 21 层 × 3 步，**每层每步各一次**。前者跑在 AI_CPU 上，正是 AI_CPU 从 1.3ms
暴增到 81.3ms 的来源——这是 PTO/Simpler 运行时的 kernel 下发路径，不是计算本身。

**两点限定**：窗口含 EP 等待、采集与同步开销，是结构对照而非稳态吞吐结论（那是 T2.3）；
`MoeDistributeDispatchV2` 两侧都有且 PTO 高 6,488µs，但 MoE 不在 PTO 替换范围内，
这部分差异更可能来自各 rank 进入集合通信的时刻不同，不宜直接归因给 PTO。

### 主要性能指标（2026-09-24 用户定）

后续性能数据**以这一组配置为主**：

| 项 | 值 | 说明 |
| --- | --- | --- |
| TP | 1 | 驱动 D 侧硬性 `tp=1`；`service_config.py` 也要求 TP=1 |
| DP | 16 | 16 卡各一个 rank |
| EP | 16 | `enable_expert_parallel=True`，EP world size = TP×DP |
| S | 6 | `DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS`，恒定 |
| **B** | **16** | 每卡 16，GBS = 16×16 = 256 |
| **seqlen** | **8192** | `h8192_bank`，四种输入 |

命令形态：

```
python tests/pypto_test/offline_pd/run.py profile \
  --bank .../h8192_bank --graph-mode full_decode_only \
  --batch 16 --backend {native,pto}
```

档位说明：`max_num_seqs=16` 时对齐后的档位为
`[6,12,18,24,36,42,48,60,66,72,84,90,96]`，16 条请求 = 96 token **精确命中 96 档**，
不产生档位补齐；补位只来自 DP。这与"主要指标"的定位自洽。



| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T2.1 | FULL_DECODE_ONLY 下重跑 Native/PTO 对照 | `results/release_csa_perf_8k_20260924/` | **已完成**，按主要指标（b16/s6/TP1/EP-DP16/8k）在图模式下采完整 PyTorch profiling（CPU+NPU、Level1、带 device kernel、`with_stack=False`）。两侧采样窗口完全对齐（第 8/9/10 步，均 96 token / 16 请求），**输出逐 token 相同**。设备侧总耗时 Native 240,013µs vs PTO 347,302µs（**PTO 慢 45%**），kernel 记录数 7,520 vs 5,378。按引擎：AI_CPU 1,270 → **81,318**（主因）、MIX_AIC 91,468 → 141,966；AI_CORE、AI_VECTOR_CORE、MIX_AIV 三项 PTO 均更低。详见下方归因 | T1.9 | 16 | **已完成** |
| T2.2 | 标注 eager 期结论的适用范围 | 已完成：验证日志新增第 98 节。第 95～97 节三轮全是 eager（当时图模式起不来），`_resolve_compiled` 按调用次数计费是 eager 特有现象，图模式下只在预热与捕获时走一遍。同时标注了两个未决前提：PTO 的 kernel 下发是否可被图捕获尚在 T1.4 验证中；本机关闭 `fuse_norm_quant` 偏离上线口径 | 无（不依赖 T2.1 数据） | 否 | **已完成** |
| T2.3 | 稳态性能测量（原 A1） | **已完成**（`results/release_csa_steady_8k_20260924/`）。两侧各 16 rank、每 rank 62～63 个采样步、`sufficient=True`。中位对照：单步 p50 **55.77ms → 66.69ms（1.20×）**、p95 57.10 → 67.59ms（1.18×）、每 rank **1635.07 → 1382.10 token/s（0.85×）**、峰值显存 50.93 → 49.81GiB。即 **PTO 端到端慢 20%、吞吐为 Native 的 85%**，比按设备侧 kernel 算的差距小，因为端到端含 MoE 与通信等两侧共有部分。口径：窗口内不加额外同步，总和与吞吐可用、单步分位数为近似 | T2.1 | 16 | **已完成** |
| T2.4 | 汇总真实 DSpark 与 EP 执行（原 A5） | **已完成**。走 `llm.get_metrics()` 公开出口。Native 与 PTO 同场景**逐个计数完全相同**：`num_drafts=1310`、`num_draft_tokens=6550`、`num_accepted_tokens=6400` → 自然接受长度 **4.885**、每步实际推进 **5.885 token**、接受率 **97.7%**。这比输出逐 token 相同更强——连 DSpark 每步接受几个草稿都一致。按 T4.4 要求不注入任何假定值 | T2.3 | 16 | **已完成** |
| T2.5 | 决定 PyPTO `_resolve_compiled` 重复遍历 AST 的处置 | 该路径在 PyPTO 内，按约束不自行修改。需用户决定走上游还是本地方案；在此之前只记录，不改 | T2.1 | 否 | **待用户决定** |
| T2.6 | 性能版：确立不占卡的分流水线归因手段 | **已完成**。用户 2026-09-24 指出 pypto-lib 的 `.claude/skills/incore-profiling`。已打通全链：主机侧 `JITFunction.warmup(RunConfig(platform="a2a3"))` 编出设备二进制（不初始化 NPU、不占卡）→ `msprof op simulator`（SoC Ascend910B1、`dav-c220`）→ `pypto.tools.clean_sim_trace`。skill 自带的兜底生成器兜不住计算型维度（`qk_pv`、`proj_a_mm` 报 `cannot safely bound computed PTO dimension`），改用 PTOAS 源码仓的 `test/npu_validation/scripts/generate_testcase.py` 后通过；两个源码仓按用户指示克隆在 `pto-eager/ptoas-src`、`pto-eager/pto-isa-src`。**并发化**：每片自带 `--output-root`，5～6 片同时跑，一轮 12 个 kernel。产物 `/data/pyptouser/qinchuanyu/pto-eager/incore_profiling_20260924/`（不入库） | T2.1 | 否 | **已完成** |
| T2.7 | 关键路径定位（不占卡） | **已完成**。`python -m simpler_setup.tools.critical_path` 直接吃已有的 level-4 泳道。makespan 1.194ms，**compute 92.8%、stall 仅 7.2%**（data-wait 6.9%、core-wait 0.4%），确认瓶颈在 kernel 自身而非调度，推翻了此前"损耗集中在任务下发与同步"的方向。关键路径 25 个任务，`indexer_score_topk_leaf_aic` 265.8µs(22.3%) + `qk_pv_aic` 256.3µs(21.5%) 占 43.7% | T2.6 | 否 | **已完成** |
| T2.8 | 性能版：解搬运受限的 matmul | **两项已验证**。in-core 全量清单显示我们慢的三个 matmul 全是搬运受限（Cube 占比均 <30%），而已追平的 `idx_qr_proj_matmul`(0.84×) Cube 占 31.8% 为全场最高。① `weights_proj`：N 块 16→整行 `IDX_N_HEADS`、K 块 256→`D_TILE=512`、按 K 切 4 份并行、去掉 `k_order` 重排与 BF16/FP16 往返 → **191,984→37,550 cycles（0.20×）**，与泳道 5.16× 相符。② `qr_proj_matmul`：`QR_N_TILE` 32→128、`QR_OK` 1→2、去掉 `QR_NATIVE_*` 的 K 块乱序 → **408,606→227,442（0.56×）**，与泳道 1.82× 相符；生成 IR 原为 `partition_tensor_view<256x32xbf16>`，每行只读 64 字节 | T2.6 | 否 | **已完成（T2.11 设备复核通过）** |
| T2.9 | 性能版：补齐权重读的 L2 BYPASS | **代码已完成，收益待设备验证**。上游在 9 处权重读上都有 `pl.set_cache_policy(w, pl.CachePolicy.BYPASS)`（`qkv_proj_rope.py:301/412/698`、`decode_indexer.py:642/810`、`decode_compressor_ratio4.py:98/99`、`decode_indexer_compressor.py:104/105`、`decode_o_proj.py:595/649`），我们一处都没有。性能版已补 11 处（另含上游只在 TP 路径有、TP1 的 `proj_a_mm`/`proj_b_mm` 两处属超出上游）。生成 IR 确认落地：9 个 kernel、28 条 `pto.tload` 带 `cache_policy = #pto.load_cache_policy<l2_bypass>`。**但 in-core 测不出差异（0.99×～1.02×）——单核模拟器不建模 L2 争用**，这条只能靠设备泳道验 | T2.6 | 16 | **已撤销，见 T2.12** |
| T2.10 | 性能版：`qk_pv` 软件流水 | **代码已完成，收益待设备验证**。按上游重写：`QK_PRE_LAUNCH=2`/`QK_TRANSFER_SLOTS=3` 的 tick 流水（AIC 第 k 拍算第 k 块 QK、第 k-2 块 PV），KV 单次进 L1 供 QK 转置视图与 PV 行切片共用（原先 QK/PV 各从 GM 搬一遍），softmax 整条 `H//2` 一次做完（原 `SOFTMAX_HEAD_TILE=8` 是 `ATTN_K_TILE=512` 时 UB 放不下的遗留），块内取局部 max、跨块重标定挪到归并侧，`merge_norm` 相应把 sink 项补进分母并去掉逆 RoPE 前的 BF16 往返。主机侧完整编译通过。**in-core 测不了**：auto golden 把 `valid_block_mask` 清零导致 0 次迭代的退化 trace | T2.7 | 16 | **已完成（T2.11 实测 1.76× → 1.03×）** |
| T2.11 | 性能版：设备侧验收本轮改动 | **已完成**（`results/perf_variant_20260924/bis_bisG/`）。同配置（b=16/S=6/TP1/EP-DP16/8k、eager 泳道）对上游：窗口跨度 1178.7 → **1062.9µs**（上游 728.0），kernel 合计 45,496.5 → **33,745.2µs**（上游 26,821.8），相对上游 **1.70× → 1.26×**，差距从 +18,675 缩到 +6,924µs（消掉 63%）。逐项：`qk_pv` 246.79 → **143.88** /次（1.76× → **1.03×**，单项 −5,115µs）；`qr_proj_matmul` 36.55 → 27.24（1.82× → 1.35×）且任务数 32 → 16 与上游一致；`weights_proj` 已跌出前 22（追平）；`merge_norm` 1.60× → 1.30×；`qproj_matmul` 2.36× → 2.05×；`proj_a_mm` 1.80× → 1.68× | T2.8、T2.10 | 16 | **已完成** |
| T2.12 | 判定 `pl.set_cache_policy(BYPASS)` 在本集成不可用 | **已完成，结论是不能用**。上游 9 处 BYPASS 出自 `56e879c`，标题为 "stream every DeepSeek V4 weight **NZ-ordered** and uncached"——绕缓存是**与 NZ 分块布局一起**引入的，而本项目 NZ 硬性关闭、权重为 ND 且来自 vLLM 分配器。设备实测两种失败：`decode_indexer.py` 的两处（`wq_b`/`weights_proj`）直接 aicore 异常 **507015**，CANN 故障日志写明 *The DDR address of the MTE instruction is out of range*；其余几处不报本地故障，但 16 个 rank 同时 **507057**（集合通信不一致）。全部撤掉后 exit=0、泳道正常采到。依据集中记在 `layout.py`，要重启用须先开 NZ 并单独验证。另记：上游该提交明确 `wo_a`/`wo_b` 是例外——它们 SDMA 预取进 L2，绕过等于扔掉预热 | T2.9 | 16 | **已完成（否定结论）** |
| T2.13 | 单卡 CSA 回放，替代整模型迭代 | **已完成**。`offline_pd/observer.py` 的 `offline_begin/end_argdump` 落盘一次真实调用的全部 52 个根入参（`torch.save`，numpy 不认 BF16），`run.py` 加 `argdump` 命令，`tests/pypto_test/dsv4_csa_single_card_bench.py` 在**单卡**上回放并直接调 `decode_csa_tp1_attention_test`。校验：同一份代码，16 卡整模型 kernel 合计 33,745.2µs vs 单卡回放 **33,610.2µs**，**差 0.4%**、逐任务排序一致。一轮从约 5 分钟降到约 40 秒、占 1 张卡。口径限制写在脚本头部：只跑 CSA 单算子、不含 MoE 与通信；墙钟 p50 约 94ms 几乎全是 eager 主机侧派发开销（PyPTO `_resolve_compiled` 按调用次数计费），量 kernel 必须看泳道 `kernel-duration`。执行目标用 `pypto.torch.init(device=, platform=)` 定在进程上，JIT 调用本身不接 `RunConfig` | T2.11 | 1 | **已完成** |
| T2.14 | 性能版：`indexer_score_topk_leaf` 的键加载路径 | **已穷尽 DSL 层的全部路径，结论是 PyPTO 表达不了**。该项占剩余差距 **88%**（aiv 2.09×、aic 2.18×）。上游的 `idx_kv_cache` 是 `[blocks,32,1,128]` **独立连续分配**，`reshape` 成 `[blocks*32,128]` 是合法视图，12 次整页 DMA 直接 GM→L1；我们的是 `[18879, 4160]`（键 4096 + scale 64/页），键行每 4096 字节被 scale 打断，该视图不存在。六条路径逐一实测被拒：① `create_l1([12,4096])` 按页搬字节——L1 是 16×32 分形，行数须为 16 的倍数；② 补到 `create_l1([16,4096])` 再 reshape `[512,128]`——**语义上就不成立**，分形装箱下两者字节排布不同，编译器在 subview 报 `boxed layout subview offsets must be multiples of inner shape`；③ `create_l1([128,384], transpose=True)` 逐行 DN2ZN——INT8 的 ZN 分形内形状是 32，逐行列偏移 0..31 不合法；④ kernel 内 `pl.slice`+`pl.reshape` 造三维分页视图——lowering 过、**运行期 AICPU 异常 507018**，用「视图建好但不用」的变体隔离确认崩在视图构造（reshape 要求「缓冲的连续前缀」，我们只是每行的前缀）；⑤ 主机侧造好三维视图当入参——运行期 `Parameter 'idx_key_pages' requires a contiguous strided tensor`，**PyPTO 根入参必须完全连续**；⑥ `pl.paged_gather` 返回形状文档写明是 `[max_indices, size]`，与①②同一堵墙；⑦ 一维/窄行源视图解耦寻址——`gather_row` 要求 `src_offset` 至少 2 个元素（一维源 codegen 报 `tile.gather_row offsets and shapes must have at least 2 elements`），换 `[blocks*65, 64]` 的 64 字节行视图后又报 `'pto.partition_view' op size at dim 1 (128) exceeds static source dim (64)`。**至此构成封闭证明**：`shapes` 同时约束源读与目的写、`src_offset` 不能是扁平字节地址、`shapes[1] ≤ src.shape[1]`，三条合起来要求「目的是 `[32,128]` 的 L1 块 ⟹ 源视图行宽必须是 128 字节 ⟹ 页跨度必须是 128 的倍数」，而我们是 4160（mod 128 = 64）。硬件本身做得到——MTE 的源与目的是两个独立描述符，可以「读 4096 个连续字节、写成 L1 的 32×128 分形块」，我们每页的键本来就连续、连跨度都不需要；做不了的是 PyPTO 这层 API 的校验。**Native 用同一份 cache 做到 60.9µs/次**（`VllmQuantLightningIndexer` 3,836.9µs / 63 次），因为它是手写 CCE、吃 `layout_key="PA_BSND"`、自建带页跨度的 DMA 描述符，不受这套视图规则约束。**唯一出路：把 indexer 的 key 与 scale 拆成两块分配**（`AscendSFAIndexerCacheSpec.page_size_bytes` 现在把两者打包）。Native 的算子本来就分开接收这两个张量，但这改的是 vLLM 的 KV cache 规格、同时影响 Native 路径，**需用户裁定** | T2.13 | 1 | **受阻（待用户裁定）** |
| T2.15 | 性能版：清掉残留的 Native 位对齐构造 | **已完成**。性能版按 `__init__.py` 的定义本就不与 Native 对齐到 bit，但仍残留 21 处只为复刻 Native 舍入的构造，逐一换成上游写法：① `qproj_dequant_rms_nope_rope` 的两个 scale 不再先物化 `pl.full([8,512])` 再两次 expand 乘，改成直接 `row_expand_mul`+`col_expand_mul`，并去掉两处 BF16 往返（主路径与尾部各两处）→ 单块中位 27.30 → **19.43µs（1.42× → 0.97×，已快于上游）**；② `qr_hadamard_quant` 去掉两处 BF16 往返与一处 FP16 标度往返 → 15.38 → **7.53µs（1.65× → 0.87×）**；③ `kv_score_proj` 的 `K_TILE` 256→512、`PROJ_OUT_TILE` 16→32，去掉 Hadamard 的 BF16 往返 → 28.83 → **14.01µs（1.20× → 0.72×）**；④ `decode_o_proj` 采用上游的按 group 独立标度：删掉独占一个任务的 `oproj_token_scale`（−86.9µs）与 CORE_GROUP 的 `quant`（−79.1µs），amax 与量化融进一个按 token 块分的 SPMD，`proj_b_act` 改为各 group 先按自己的标度反量化再相加；⑤ `qr_rms_norm_quant` 换掉整个 `q_proj_qr_rms`：原实现先落 BF16、按 1024→512→…→64 折半规约、再逐行用 INT64 整数比较修正 sqrt 末位并逐行标量除法，改成上游的一遍扫描同时求平方和与 gamma 加权 amax（利用 `amax(normed)=inv_rms·amax(qr·gamma)`）+ `rsqrt(high_precision=True)` → 9.40 → **7.61µs（1.20× → 0.95×）**；⑥ 两个压缩器的 `scatter_softmax_pool` 换成上游的在线 softmax（不再把 8 格按 Native 交错次序拼成 `[8,tile]` 再归一化规约）→ 12.05 → **9.08µs（1.45× → 1.02×）**；RMSNorm 的逐块除法换成先取倒数再乘。合计 kernel 从 33,363 → **31,037µs**，相对上游 1.26× → **1.157×** | T2.13 | 1 | **已完成** |
| T2.16 | 性能版：`qproj_matmul` 去掉尾部重读权重 | **已完成**。`q_proj_q` 原按 `QPROJ_TAIL_M_TILE=16` 向上取整分配 `q_proj_i32`，T=96 时整块循环只覆盖 64 行（`QPROJ_M_TILE=64`），余下 32 行走 16 行一块的尾部循环，而**每个尾部小块都要把整块 `[Q_LORA, QPROJ_MM_N_TILE]` 权重重新读一遍**——整张 `wq_b` 因此被读 3 遍而非 2 遍。改为按 `QPROJ_M_TILE` 向上取整、删掉尾部循环（多算的行读的是 `qr_i8_matmul` 的补位行，INT8 乘加不产生非有限值，结果落在补位行、反量化只读前 `tile_rows` 行）→ 单块中位 64.40 → **57.15µs（1.84× → 1.63×）**。另测两个反例：`QPROJ_M_TILE=96`+`N=256`（让 96 行一次覆盖）反而 65.92µs，`QPROJ_M_TILE=48` 也只到 61.83µs——都不如直接消掉尾部循环 | T2.15 | 1 | **已完成** |
| T2.17 | 逐张权重复核 BYPASS 是否可用 | **已完成，结论仍是全部不可用**。T2.12 是 11 处一起加、一起撤，只定位到手段；本轮按张量单独开：`wq_a`、`wq_b`(qkv)、`wq_b`(indexer)、`weights_proj`、两个压缩器的 `wkv`+`wgate`、`kv_proj` 的 `wkv`，共 7 个探针包分别上板，**7 个全部失败**（`wq_a` 报 507057 集合通信不一致，其余报 507015 aicore 异常）。这否掉了「上游对 ND 张量也用 BYPASS 所以我们也能用」这条推测——上游 `decode_indexer.py:642/810` 的 `wq_b`/`weights_proj` 确实是普通 ND 且带 BYPASS，但在本集成下即便单独开也崩。`qr_proj_matmul` 的函数体与上游逐字节相同、唯一差别就是这条 BYPASS，所以它那 1.54× 属于**不可通过算子内改动消除**的部分 | T2.12 | 7 | **已完成（否定结论）** |
| T2.18 | indexer 键路径：算子内新增重排，推翻 T2.14 的受阻结论 | **已完成**。T2.14 断言「唯一出路是改 vLLM 的 KV cache 规格」，前提是必须原地读 Native 的交错页——**加一次重排就绕开了这个前提**。新增 `indexer_key_repack` 任务：每步先把本步可见的页按逻辑页序拷成紧凑的 `[b_dim*repack_pages*BLOCK_SIZE, IDX_HEAD_DIM]`，scale 拷成 `[b_dim, repack_pages*BLOCK_SIZE]`；打分侧读到连续行后，就能用上游那套 `create_l1` + 整块 `gather_row` 直搬 L1（T2.14 里失败的是**转置**与 **12 行**两种形态，非转置的 32 行整页形态本身合法），并把一个 lane 的 6 页并成一次 DMA。**Native 的 cache、块表、分配、调度全不动**，只在算子内多一个任务和一块每步临时缓冲。实测 `indexer_score_topk_leaf_aiv` 138.67 → **55.42µs（2.13× → 0.85×，已快于上游 65.03）**、`aic` 139.20 → **54.76µs（0.86×）**，repack 自身 18.38µs×48 = 818µs，净收益约 **−4,600µs**。降本两步：键与 scale 在页内本就连续（4096+64 = `INDEXER_MIN_PAGE_BYTES`），合成一次 DMA（27.23 → 20.12µs）；尾部余量从 `SCORE_TILE` 缩到 `SCORE_LANE_ROWS` 一个 lane 的量（→ 18.38µs）。余量的作用是让打分侧最后一个 tile 不必把读起点往回夹——往回夹会让 tile 内的行与候选列号错位（实测 absmax 6.28125 → 6.25） | T2.14 | 1 | **已完成** |
| T2.19 | 按编译器 PH-MR-001 修双缓冲：否定结论 | **已完成，结论是不能按提示改**。主机侧编译产出的 `report/perf_hints.log` 报了 25 处 `PH-MR-001`：`pl.pipeline(stage=2)` 申请了双缓冲但只放得下一份，「stages 1 apart share storage and serialize」，即搬运与计算根本没重叠。点名的正是剩余差距最大的几个 matmul（`proj_a_mm` 6 处、`qr_proj_matmul` 4 处、`proj_b_mm` 4 处、`kv_proj_matmul` 4 处、`qproj_matmul` 2 处、`indexer_score_topk_leaf` 4 处）。L0A/L0B 各 64KiB，所以每阶段须 ≤32KiB。按此把各自的 K 分块减半做了 5 个探针（`QR_K_TILE` 256→128、`KV_K_TILE` 256→128、`A_K_TILE` 256→128、`B_K_TILE` 512→256、`Q_PROJ_TILE` 128→64），**目标 task 全部变差**：qr_proj 33.53→38.96、proj_a 28.08→31.50、qproj 52.20→62.83、proj_b 16.64→17.77、kv_proj 基本持平。原因是 ND 布局下 K 决定每行的连续字节数，K 减半等于把 MTE2 的连续段砍一半、循环次数翻倍，代价大于双缓冲的收益。**在 ND 下连续字节长度比双缓冲更主导**，编译器这条提示在本场景是误导 | T2.16 | 6 | **已完成（否定结论）** |
| T2.20 | 非 indexer 项的收口盘点 | **已完成**。追平或反超上游的：`kv_score_proj` 0.65×、`idx_qr_proj_matmul` 0.81×、`qr_hadamard_quant` 0.92×、`kv_proj_matmul` 0.99×、`proj_b_mm` 1.00×、`idx_qr_dequant_rope` 1.02×、`proj_b_act` 1.06×、`qproj_dequant_rms_nope_rope` 1.08×、`scatter_softmax_pool` 1.11×、`quant` 1.12×。`merge_norm` 单看 1.37× 但**不应孤立看**：性能版把跨块重标定从 `qk_pv` 挪到了归并侧，`qk_pv+merge_norm` 合计 152.48 对上游 155.95 = **0.98×**。仍高于上游且已无算子内手段的只剩三项，成因同一个——上游的 `wq_a`/`wq_b`/`wo_a` 是 NZ 且配 `set_cache_policy(BYPASS)`，我们是 ND 且 BYPASS 不可用：`qr_proj_matmul` 1.71×（函数体与上游**逐字节完全相同**，唯一差别就是那一行 BYPASS）、`qproj_matmul` 1.49×（7 种切块全试过）、`proj_a_mm` 1.45×（6 种切块全试过）。三者合计约 +1,140µs。NZ 按用户 2026-09-25 的要求不碰，BYPASS 经 T2.17 逐张量验证全部上板崩溃 | T2.17、T2.19 | 1 | **已完成（余量需用户裁定）** |
| T2.21 | 三方对照口径：补上 Native 这条基准线 | **已完成，并推翻 T2.20 的「无手段」结论**。T2.20 只拿上游（NZ + BYPASS）做参照，于是把 `qproj_matmul`/`proj_a_mm`/`qr_proj_matmul` 判成「算子内改不动」。但目标写的是「native **或**上游最好水平」，而 **Native 用的正是 ND**（`aclnnQuantMatmulV5_QuantBatchMatmulV3` 的名字里就是 `ND_ND_int8_int8`），是更合适也被明确认可的达标线。从 Native profile 的 `kernel_details.csv` 按 `Input Shapes` 定位到同形状调用：qproj = `"80,1024;1024,32768"` 72.91µs、proj_a = `"80,8,4096;8,4096,1024"`（`aclnnTransposeBatchMatMul`）92.24µs、proj_b = `"80,8192;8192,4096"` 54.26µs、qr_proj = `"80,4096;4096,1024"` 35.29µs、kv_proj = `"80,4096;4096,512"` 30.02µs。折算口径：我方泳道给的是每个 SPMD 块的时长，一个 task 的墙钟约等于 `ceil(块数/核池) × 每块中位`（AIC 24、AIV 48），Native 给的直接就是一次调用的墙钟；Native 那一步 80 token、我方 96，按 token 归一。结果：**四项全部优于 Native**——qproj 0.60×、proj_a 0.76×、proj_b 0.77×、qr_proj 0.79×。对照脚本固化在 scratchpad 的 `native_cmp.py` | T2.20 | 1 | **已完成** |
| T2.22 | 提高 AIC 核利用率：split-K 切细任务 | **已完成，两项大幅改善**。按「块数 / 核池」盘点全部 task，发现两处明显闲置：`qr_proj_matmul` grid = `(Q_LORA//QR_N_TILE) * QR_OK` = 8×2 = **16 块，24 个 AIC 核里 8 个全程闲置（66.7%）**；`kv_proj_matmul` grid = `(HEAD_DIM//KV_N_TILE) * KV_OK * kv_m_groups` = 4×2×1 = **8 块（33.3%）**。两者的 split-K 都是现成机制（`qr_proj_seed`/`kv_proj_seed` 负责置零，`assemble(atomic=Add)` 负责规约），只改常量：`QR_OK` 2→8（16→64 块，每块 K 2048→512，墙钟 **33.5 → 24.4µs**，单块 33.53→8.13 即 0.41× 上游，下游 `qr_rms_norm_quant` 也顺带 8.99→6.41）、`KV_OK` 2→8（8→32 块，墙钟 **25.6 → 15.0µs**，0.59×）。输出逐位不变。否定的分支：`QR_N_TILE` 128→64 会改变 atomic 的累加顺序，实测 mean 从 -0.014165431 变成 -0.014159609，不采纳；`QR_OK`=16、`KV_OK`=4 的墙钟都不如 8；`proj_b_act` 的 N 分块 512→256 虽让块数 24→48，但墙钟没改善（说明它不受核利用率限制）且同样改变累加顺序 | T2.21 | 8 | **已完成** |
| T2.23 | 整模型泳道复核：与最初同口径对比 | **已完成**（`results/perf_variant_20260924/final_swimlane/`）。16 卡整模型 eager 泳道、同配置（b=16/S=6/TP1/EP-DP16/8k、`--swimlane-layer 2`）：**窗口跨度 1178.7 → 827.5µs（0.70×，省 351µs）**，**kernel 合计 45,496.5 → 27,712.5µs（0.61×，省 17.8ms）**。对上游（728.0µs / 26,821.8µs）：窗口 1.137×、kernel 合计 1.033×、共有任务合计 1.050×。注意单卡回放与整模型的共有任务合计有约 2% 的口径差（单卡 0.974～1.029×、整模型 1.050×），因为整模型有 16 个 rank 的访存争用；判断单次改动仍以单卡逐 task 中位数为准，整模型用于阶段性验收 | T2.22 | 16 | **已完成** |
| T2.24 | 精度版：同步数值中性的优化 | **已完成**。此前所有优化只落在性能版。按「会不会改变浮点累加顺序或舍入」分辨后，把三项搬进精度版：① qproj 的 `QPROJ_MM_N_TILE` 512 + `Q_PROJ_TILE` 128 + 去尾部循环（INT8×INT8→INT32，整数累加精确且与分块无关）；② proj_b 的 `B_K_TILE` 512 + `PROJ_B_MM_N_TILE` 128（同为 INT8）；③ `indexer_key_repack`（纯搬运，读到的字节完全相同；精度版的键加载段与性能版改前逐字节一致，直接照搬）。**没搬**：`QR_OK`/`KV_OK` 的 split-K（FP32 用 atomic add 规约，分片数变则累加序变；精度版原注释就写明 split-K 会改变 Native 的 QR 舍入）、以及全部数值写法类改动（去 BF16/FP16 往返、在线 softmax、按 group 独立标度、qk_pv 把跨块重标定挪到归并侧）。验证：单卡回放改前改后**输出逐位相同**（absmax 6.28125、mean −0.014240865595638752）；逐 task `indexer_score_topk_leaf_aiv` 173.67 → 128.67µs、`aic` 173.75 → 129.45µs、`qproj_matmul` 78.33 → 55.48µs，共有合计 44,260 → 40,666µs（**1.76× → 1.62×** 上游）。踩过两个坑：`bench.sh` 的标签前缀判定把基线包名 `precbase` 也当成了精度版本体，两次跑的是同一份代码；以及在任务排队期间改了 `decode_indexer.py`，而 `@pl.jit` 编译时重读源文件 | T2.22 | 2 | **已完成** |
| T2.25 | 核实 PTO 的 NZ 现状 | **已完成，结论：基本能力可用，但不能照搬上游写法**。① pypto 自带的 NZ 系统测试 `tests/st/runtime/ops/test_matmul_nz.py` 在本环境（pypto 0.1.0 / PTOAS 源码 0.66 / CANN 9.0.0）**4 passed**，且比对容差是 `rtol=atol=0`，覆盖整张量、N 方向切片（行分形偏移 `n0//16`）、K 方向切片（C0 列块偏移 `k0//c0`）三种寻址；NZ 要求 PTOAS ≥ 0.61，实际 0.66。② **但上游 `qkv_proj_rope.py` 单体在本环境编译失败**，报的正是 NZ 非负性：`w_col0 = qproj_n_idx * QPROJ_MM_N_TILE` 里 `qproj_n_idx` 是 `pl.range(worker, N, WORKERS)` 的循环变量，而 `IsProvableNonNegative` 对 `IterArg` 不做推导。③ 上游 `decode_csa.py` 整体编译也失败（`store() got an unexpected keyword argument 'pre_quant'`），说明**上游 pypto-lib 的 main 依赖比本环境更新的 pypto**，不能拿「上游能跑」当本环境可用的依据。④ 落点只有 4 张权重：上游自己注释了 indexer 的 `wq_b`、`weights_proj`、压缩器的 `wkv`/`wgate` 都**不能** NZ（它们的 spmd 索引含减法，而差永远不可证）。⑤ 显存：`wq_a`/`wo_a`/`wo_b` 在 `prepare_weights` 里本就 `transpose().contiguous()` 过独立副本，NZ 化零额外开销；`wq_b` 共享 Native 权重，NZ 化需 +32MiB/层 × 21 层 = 0.66GiB | T2.21 | 1 | **已完成** |
| T2.26 | 解锁 NZ：框架侧修 entry ABI 校验 | **已完成**。`pl.NZ` 用在带编排的 `@pl.jit` 根入参上会被 `finish_kernel_artifact` 拒掉（`Lowering changed the kernel entry parameter ABI`）。加了参数级诊断后看清：`#46 'wo_a': bfloat16/In/(8,1024,4096) -> 'wo_a__ssa_v0': bfloat16/In/(8,256,64,16,16)`——`BlockNzTensorViews` 把 GM 视图改写成 rank-5 的 `[C/c0, R/16, 16, c0]`（BF16 的 c0=16），**元素数 33,554,432 完全相同**，是同一块内存、同一个池位、同样的 dtype 与方向，只有 shape 的表述变了。而那条校验的注释写明意图是防 lowering「引入或重排」外部池。据此把外部池的身份判据从「逐维 shape 相同」改成「dtype + 方向 + 元素总数」（含动态维时退回逐维比较），并让报错说清是哪个参数怎么变的。改动在 `pypto/python/pypto/ir/_kernel_compile.py`。**注意 `warmup(RunConfig)` 路径不触发这条校验**（`_kernel_abi is None` 时不调用 `finish_kernel_artifact`），只有 `pypto.torch.init` 的 eager kernel-mode 才会，所以主机侧编译「通过」是假通过 | T2.25 | 1 | **已完成（框架侧）** |
| T2.27 | NZ 与 ND 双分支，由 weight_nz_mode 控制 | **已完成**。按用户要求两条分支都保留、由 vllm-ascend 统一开关择一，不做单向替换。新增 `nz_mode.py` 读 `VLLM_ASCEND_ENABLE_NZ`（语义同 `ascend_config.py`：0 关闭／1 只量化权重／2 BF16 也开；`wo_a` 是 BF16 对应 mode≥2）。读环境变量而非 `AscendConfig.weight_nz_mode`，因为 kernel 的参数布局是**模块加载时**由类型注解定下来的，而 AscendConfig 要等 vllm 初始化完才有。三处由同一开关驱动：kernel 签名的 layout 槽放闭包变量 `BF16_WEIGHT_LAYOUT`（`pl.NZ` 或 `None`，后者等价于不声明 layout）、主机侧 `_maybe_pack_nz`、单卡 bench 的 `nz_args.pack_args`——三者必须一致，否则标注说 NZ 而字节还是 ND，不会报错、只会算错。`_pack_nz` 在 **CPU** 上做重排：在 NPU 上 reshape/permute 会让 torch_npu 把 npu format 推断成 `FRACTAL_NZ(30)`，而 PyPTO 根入参只收 NCHW(0)/ND(2)。实测两模式输出完全一致，`proj_a_mm` 单块 ND 29.73µs → NZ **23.22µs**，墙钟 87.0 → **69.7µs（1.45× → 1.20× 上游、0.79× → 0.63× Native）** | T2.26 | 2 | **已完成** |
| T2.28 | 其余三张权重的 NZ：均不采纳 | **已完成（否定结论）**。① `wq_b`（qproj）：偏移可证性要求把 N 索引从 `pl.range(block_idx, N, WORKERS)` 的循环变量换成块索引直乘（循环变量是 IterArg，非负性推导明确不追它，上游同样写法在本环境也编不过）。改完能跑且数值正确，但单块 55.70 → 17.86µs 而块数 24 → 64，**墙钟只从 55.7 降到 53.6µs**（qproj 在 N=512 下 ND 已有 512B 连续段，瓶颈不在搬运连续性），却要多一份 32MiB/层、21 层约 0.66GiB（它不像另外三张那样本来就有独立副本）。② `wo_b`：为消掉偏移里的减法而改 grid 后，编译报 `proj_a_mm` 的标量被 hoist 到 scope 外；且它本来已是 1.02× 上游。③ `wq_a`（qr_proj）：K 分片放 `pl.parallel` 索引再用作行偏移，报 `offset on shape[-2] must be a multiple of 16, cannot be proven`——可证形式只认常量和 start/step 均为 16 倍数的循环变量（`wo_a` 没踩到是因为它的 parallel 索引用在 batch 维，batch 维不做 16 对齐检查）。改成三维 `[QR_OK, D//QR_OK, Q_LORA]` 把 K 分片挪到 batch 维后能编过，但**数值错**（absmax 4.19 对 6.28），且 ND 模式下同样错、两次运行的 mean 还不同：`pl.parallel` 的多个分支要 atomic add 到同一块输出，这个累加保证在 `manual_scope` 下不成立（o_proj 里每个 parallel 分支写的是各自独立的列区间）。收益（qr_proj 1.24× → ~1.0×，约 +90µs）不值这个正确性风险 | T2.27 | 4 | **已完成（否定结论）** |
| T2.29 | 升级 pypto / simpler 到最新分支 | **已完成**。pypto `5495749` → **`879602d`**（`fix(runtime): adopt device-only HBG kernel contract` #2894），simpler `166852bf` → **`dd32e1cc`**（#2433），pypto 的 `runtime` submodule 同步到 `dd32e1cc`。踩到的三点：① 运行时用的 simpler 是**独立 checkout** `/data/pyptouser/qinchuanyu/pto-eager/simpler`（editable install），不是 pypto 的 submodule，两份都要升；② 只 merge 源码不够，两个库的 C++ 扩展都要重编，revision 校验会逐级报错（先 `Kernel ABI requires Simpler <rev>; native binding is <旧>`，重编 simpler 后再 `The PyPTO torch_npu adapter uses a different Simpler revision`），各跑一次 `pip install -e . --no-build-isolation --no-deps`；③ 本地必须保留两处修改——`torch/shutdown.py` 放宽 torch_npu 版本校验（上游只验证 2.6.0.post2，本环境是 2.10.0.post2），以及 T2.26 的 ABI 修复。`_kernel_abi.py` 里那份对齐 simpler commit 的临时修补由上游自己 pin，不必保留。升级后回归：输出逐位不变，共有任务合计与升级前在噪声内（27,114 vs 27,361） | T2.25 | 3 | **已完成** |
| T2.30 | 量 simpler `58180f78` 对整模型窗口跨度的影响 | **进行中**。`58180f78`（*reduce kernel admission and A2/A3 retirement overhead*）改两处运行时路径：CPU role 准入改为「已发布槽位的连续前缀含齐所有请求的 role 就立即接纳」，以及把所有已确认的 AICore window CLOSE 写排到 per-window 回读之前、drain 后才发布 return gate。后者关系设备侧任务退役，理论上会动任务间隙。**第一次对比不可用于归因**：升级前 827.5µs / 27,712.5µs 对升级后 864.2µs / 28,571.6µs，看似退化 4.4%，但 ① 两次泳道之间还混了 `proj_a_mm` 的 grid 改动（`final_swimlane` 跑在 `72c4e6cc`，之后才有 grid 改动）；② **kernel 合计也涨了 3.1%**，而设备侧计算时间不该受准入/退役影响，说明落在运行波动里（单卡实测合计波动 ±2～4%）。正在做的：同配置重复采样量波动幅度，再用干净对照组（同一份算子代码、只 `git revert 58180f78`，已验证能干净应用）对比。单卡口径的同代码对照已有：墙钟 93.2 → 92.8ms、共有合计 27,114 → 27,361，都在噪声内——但单卡只有 1 个 rank，准入/退役压力小，本来就测不出这条改动 | T2.29 | 16 | **进行中** |
| T2.31 | 支持 vllm-ascend 默认的 `weight_nz_mode=1` | **代码已完成，整模型验证中**。此前闸门硬性要求 `weight_nz_mode=0`，而 **1 才是 vllm-ascend 的默认值**——也就是说 PTO CSA 一直在拒绝框架的默认配置，能跑是因为运行口径显式设了 0。查实的链路：CSA 的权重用 vLLM 标准的 `ReplicatedLinear`/`ColumnParallelLinear`/`RowParallelLinear`（`models/deepseek_v4.py:746-783`），而 vllm-ascend 接管了它们的 `process_weights_after_loading`——BF16 走 `AscendUnquantizedLinearMethod`（`ops/linear.py:98`，`_should_trans_nz` 要求 mode==2 才转），INT8 走 `W8A8DynamicLinearMethod`（`w8a8_dynamic.py:148`，mode>=1 就转）。所以 **mode=1 只转 `wq_b`/`wo_b` 两张**。改动：① 闸门放宽到 `weight_nz_mode in (0, 1)`，`enable_kv_nz` 仍拒绝（它改的是 KV cache 页布局，PTO 的 cache 读取按 Native 的 ND 页布局写死）；② 两个包的 `prepare_weights` 把 format 非 0/2 的权重 `npu_format_cast` 回 ND；③ `offline_pd/run.py` 加 `--weight-nz-mode`。**整模型已验证**：mode=1 下 16 卡起得来，权重加载与 `prepare_weights` 无报错，decode 正常、DSpark 投机接受率 100%（平均接受长度 6.00）。**对 replay 无影响**：`prepare_weights` 挂在 `PyptoCSADeepseekV4ForCausalLM.process_weights_after_loading` 上，每层一次、发生在权重加载后 aclgraph capture 前，不在 decode 路径。**代价**：`wq_b` 在 mode=0 下只做 `.contiguous()`（已连续则返回自身、与 Native 共享），mode=1 下必然产生新副本，**+32MiB/层 × 21 层 = 0.66GiB**，而 Native 那份 NZ 权重在 PTO 路径下不再被用到却仍占显存。所以 mode=1 相对 mode=0 是纯亏（多显存、多一次加载期转换、性能相同），价值只在于让 PTO 能在框架默认配置下直接起来。注意**单卡 bench 验证不到这条路径**——它从 argdump 读张量、不走 `prepare_weights`，入参本来就是 ND，压根不会触发 `npu_format_cast` | T2.27 | 16 | **已完成** |
| T2.32 | mode=1 默认配置下的 Native/PTO 完整 profiling | **已完成**（`results/perf_mode1_20260925/`）。16 卡、b=16、`--weight-nz-mode 1`、`FULL_DECODE_ONLY`、稳态 8~10 步共 3 步。**输出 token 完全一致**（`output_token_ids_equal: True`）。分核类型：PTO 的 `AI_CORE` 23,505.7 对 Native 33,476.7（**0.70×**）、`AI_VECTOR_CORE` 37,752.0 对 48,756.8（**0.77×**）、`MIX_AIV` 54,922.9 对 66,304.1（**0.83×**），`MIX_AIC` 117,735.3 对 91,524.6（1.29×）。PTO 替换掉的 Native 算子全是净赚：`QuantBatchMatmulV3` −10,417、`Compressor` −8,068、`SparseAttnSharedkv` −6,660、`TransposeBatchMatMul` −6,270、`VllmQuantLightningIndexer` −3,771。**`device_totals_us` 的 1.182× 是重复计数**：PTO 侧多出 `simpler_aicpu_kernel_exec_*` 49,977.5µs 与 `aicore_kernel_mode_0_mix_aic` 49,450.5µs，两者都是 63 次（3 步 × 21 层），每次 793.29µs 与 784.93µs——后者是 CSA kernel 在设备上的实际执行时间（与泳道 makespan ~800µs 吻合），前者是 AICPU 侧编排线程在 kernel 执行全程驻留等待，两者**重叠而非叠加**，却都被计进了 device 总量。所以看 PTO 真实开销要用泳道 makespan，不能用 `device_totals_us` | T2.31 | 32 | **已完成** |
| T2.33 | 消掉 `rope_swap`：关键路径上最大的单项停顿 | **代码已落地，整模型验证中**。16 卡整模型泳道的关键路径显示 `rope_swap` compute 仅 **1.7µs** 却 **core-wait 72.1µs（占 makespan 8.95%）**——它算的是 `j^1` 的 lane swap 索引表，与 `t_dim` 无关、纯常量，却用一个 `CORE_GROUP` 任务独占一个核、每步重算并等核。顺带查出一条**假依赖**：`rope_cs` 挂着 `deps=[swap_tid]`，但它自己重算符号表（`cs_lane`/`cs_sign`），从不读那张 GM 表。改法：删掉该任务与其 GM 张量、去掉假依赖、把索引表内联进唯一真正的消费者 `merge_norm`（它本就是 SPMD，几条向量指令，与 `rope_cs` 的做法一致）。算式逐字相同，单卡输出逐位不变。**单卡测不出这个改动**：关键路径报告显示 `rope_swap` 在单卡下根本不在关键路径上，那 72µs 是 16 卡并发的核竞争造成的——这与 T2.31 的教训同类，调度层面的优化必须用整模型验证。**整模型实测效果超出预期**：core-wait 从 13.1% 塌到 **1.3%**（0.106ms → 0.011ms，省 95µs），compute 占比从 72.6% 回到 **85.7%**，makespan 0.806 → 0.791ms。不只是它自己那 72µs——`kv_proj_seed`、`qr_rms_norm_quant` 的 core-wait 也一并消失，它们本就在排队等同一批核 | T2.32 | 16 | **已完成** |
| T2.34 | 关键路径的构成变了：stall 占比从 10% 升到 27% | **已识别，待优化**。mode=1 整模型关键路径：makespan 0.806ms、compute 0.585ms（72.6%）、**stall 0.221ms（27.4%）**，其中 data-wait 14.3%、**core-wait 13.1%**。而最初（base）是 compute 90.0% / stall 10.0%（core-wait 仅 0.5%）——kernel 变快之后，调度开销的占比显著上升。关键路径前 5 个任务 compute 合计只有 24.1µs 却背了 109.7µs 的 stall（占 13.6%），全是启动期的核竞争：`rope_swap` 72.1µs（T2.33 已处理）、`kv_proj_seed` 18.9µs、`qr_rms_norm_quant` 13.8µs，都是小任务在等核。**T2.33 落地后此项已解决**：core-wait 降到 1.3%，关键路径回到 compute 主导（85.7%），stall 散成一堆 1% 以下的 data-wait、无单点大坑。新的靶子回到 incore task 本身，按占 makespan 排序：`qk_pv_aic` 149.4µs（18.9%）、`qr_proj_matmul` 75.6µs（9.6%）、`proj_a_mm`×2 74.9µs（9.5%）、`kv_score_proj_0` 65.4µs（8.3%）、`indexer_score_topk_leaf_aic` 61.5µs（7.8%） | T2.33 | 16 | **已完成** |
| T2.35 | 按功能组对照 Native，重新定位差距 | **已完成，纠正了此前的靶子**。之前一直拿上游（pypto-lib）比，漏了功能级的 Native 对照。用 T2.32 的 profile 把 PTO 关键路径上的任务链按功能对齐 Native 的算子（Native 取每次耗时、PTO 取整模型关键路径上的 compute，同为单层单步）：**indexer**（`repack`+`score`+`publish` 117.8µs）对 `VllmQuantLightningIndexer` 59.9µs = **1.97×**；**attention**（`qk_pv`+`merge_norm` 175.9µs）对 `SparseAttnSharedkv_..._1026` 105.7µs = **1.66×**；**compressor**（`kv_score_proj`+`scatter_softmax_pool`+`indexer_boundary_init` 78.8µs）对 `Compressor`×2 128.1µs = **0.62×**（我们更快）。前两组合计 293.7µs、占 makespan 37%，是真正的大头。注意 `qk_pv` 单看是 0.93× **上游**、却是 1.41× **Native**——只跟上游比会看不见这个差距 | T2.34 | 1 | **已完成** |
| T2.36 | indexer：repack 拆成历史段与尾页段 | **已撤回（否定结论）**。设想：`indexer_key_repack` 整体挂 `deps=[cache_write_tid]`，32.9µs 全压在关键路径上；而本步最多新增 `ceil(S/COMPRESS_RATIO)=2` 个压缩 token、只落在每个请求的最后一页，历史页（约 63/64）在本步不会被写，于是把它拆成不依赖 `cache_write` 的历史段与依赖它的尾页段，指望历史段能与 q/kv 投影那批并行、从关键路径让出 32µs。**整模型实测退化 6%**：makespan 0.791 → **0.839ms**。repack 自身确实降了（32.9 → 23.8 + 尾页 4.3），但 `indexer_score_topk_leaf_aic` 从 61.5 **涨到 100.9µs**。前提判断错了——关键路径上 q/kv 投影那段本来就把 AIV 核占满，并没有空闲核等着接历史段，提前跑只是把争抢提前；而且 `key_compact` 由早晚两段分别写入后，打分侧读它时的 cache 局部性也变差。**教训**：把任务移出关键路径的前提是目标时段确有空闲核，不能只看依赖图。代码已撤回，单卡验证过的数值正确性不受影响（输出逐位不变），只是没有收益 | T2.35 | 16 | **已撤回（否定结论）** |
| T2.37 | 去掉 PTO kernel 前逐层重算的 compressor metadata | **已完成**。用户从整模型 PyTorch profiling 里发现 PTO 算子正前方连着两个 `CompressorMetadata`。核对后：**这不是 PTO 比 Native 多出来的**——两边调用次数完全相同（各 186 次），Native 每层同样花 30.6µs，只是它那两次各自紧跟一个 72µs 的 `Compressor` 把结果吃掉、不显眼，而 PTO 把 compressor 融进了 kernel，这两次就裸露成 kernel 前的串行头（`RmsNorm 12.6 → CompressorMetadata 16.3 → CompressorMetadata 14.0 → [PTO kernel 790]`，后面再无任何 Compressor）。但它确实是**两边共有的真冗余**：`_compute_compressor_metadata` 的入参（`full_compress_cos/sin`、`query_start_loc`、`start_pos`、`block_table`、`block_size`、`compress_ratio`、`num_compressed_tokens`、`num_reqs_actual`）全是 per-step、per-KV-cache-group 的量、完全不含 layer，而 `vllm_ascend/worker/model_runner_v1.py:3000` 让同一个 attn group 的所有层共用同一个 metadata 对象，于是 21 层拿一模一样的入参把同一个算子算了 21 遍。改法是纯新增：按 decode metadata 的对象身份缓存，挂在 `forward_context.additional_kwargs`（每次前向由 `platform.py` 的 `set_additional_forward_context` 新建，不跨 step 残留）。16 卡 aclgraph 实测 **186 次 / 2118.8µs → 66 次 / 324.6µs**，省下的 120 次正是 21 层 × 3 step × 2 组减去每 step 保留的 2 次，合计 **1794.2µs ≈ 598µs/step**；16 个 rank 的 `output_token_ids` 与改动前、与 Native 均一致，aclgraph 重放正确。该收益约占 decode step 的 1.7%、低于同配置 3.4% 的测量波动，单次 steady 测不出来，故只以算子级计数为准。两边 CSA 段一层墙钟：PTO 834.8 → 约 805µs，Native 654.0µs（Native 若同样缓存可降到约 624µs——这份浪费上游也有）。提交 `1dcadd85`，性能版与精度版同步 | T2.35 | 16 | **已完成** |

### T3.2 已出三点的完整分析（B=1／8／16）

| | rank | 目标层 | 各 rank 输出组数 | 每请求 token | 单轮耗时 min／中位／max |
| --- | --- | --- | --- | --- | --- |
| B=1 | 16 | 21 | 4 | 64 | 2.18／7.78／20.80s |
| B=8 | 16 | 21 | 4 | 128 | 3.83／5.63／21.20s |
| B=16 | 16 | 21 | 4 | 128 | 5.69／11.18／25.66s |

**输出组数恒为 4** 与设计一致：bank 每档有四个输入 variant，worker 按 `rank % 4` 取，
所以 16 个 rank 只应产生 4 组不同输出，且同 variant 的 rank 之间逐 token 相同。

**捕获期档位覆盖**（计数 336 = 21 层 × 16 rank，即每层每 rank 一次；672 为两次）：

- B=1：`pto_tokens6` 672；Native 只有 `tokens1`／`tokens6`／`tokens256` 各 336
- B=8：PTO 覆盖 6／12／18／24／36／42／48 共 7 档，各 672
- B=16：PTO 覆盖 6／12／18／24／36／42／48／60／66／72／84／90／96 共 **13 档**，各 672，
  与引擎日志里的 `cudagraph_capture_sizes: [6,12,18,24,36,42,48,60,66,72,84,90,96]` 完全一致，
  **全部是 6 的倍数**，档位对齐生效

Native 侧只在 `tokens1`、`tokens256` 和该 batch 的最大 token 数上各有一次，属非 decode 形状
与预热，不构成 decode 路径的回退。结合 T1.4／T1.7 已确认的 `rejected=0`，可判定
**21 个目标层在所有档位上都走 PTO，没有静默回退**。

B=24／32 因上述 `exit=130` 未出，T3.2 保持"进行中"。

### 与上游 pypto-lib 的性能对照：92% 的差距在精度锁定代码里

用户 2026-09-24 提供了上游参考实现在**同配置**（b=16、S=6、TP1、8k）下的泳道
`results/release_csa_perf_8k_20260924/shangyou-merged_swimlane_20260924_005402.json`，
可与我们的逐任务对照。核数分配两边一致（48 AIV／24 AIC／16／8），负载形状可比。

| | 上游 | 我们（移植后） |
| --- | --- | --- |
| 窗口跨度 | 728.0µs | 1178.7µs |
| kernel 合计 | 26,821.8µs | 45,496.5µs |

总差距 18,674.7µs，其中**精度锁定任务占 17,123.2µs（92%）**，其余仅 1,551.5µs（8%）。

所谓"精度锁定"指该实现是为对齐 Native 的数值行为而刻意写成现在这样，代码里有
明确注释或 `NATIVE_*` 常量为证，改动即改变数值：

| 任务 | 上游 | 我们 | 差 | 锁定依据 |
| --- | --- | --- | --- | --- |
| `indexer_score_topk_leaf_aiv` | 3,077 | 8,304 | 5,227 | `NATIVE_QLI_QK_SCALE=1/1024`、FP16 QK tile + Cube 规约；上游用 Vector `col_sum` |
| `qk_pv_aiv` | 6,731 | 11,892 | 5,160 | 跨 Native 512 候选块保持单个 FP32 PV 累加器；概率用 CAST_ROUND；每 512 候选后舍入 |
| `indexer_score_topk_leaf_aic` | 1,512 | 4,151 | 2,639 | 同上 |
| `qk_pv_aic` | 3,310 | 5,903 | 2,594 | 同上 |
| `qr_proj_matmul` | 322 | 1,144 | 822 | `QR_NATIVE_SHIFT_*` 重排 K 累加序 |
| `weights_proj` | 85.5 | 523 | 438 | 针对 CANN9.0 A3 MatMulV2 遍历序的 `k_order` 重排，FP32 累加 |

**结论：PTO 与 Native 输出逐 token 相同这件事，当前代价是约 1.6 倍的 kernel 时间。**
用户 2026-09-24 定"影响精度的先不动"，因此这些项一律不改；要继续压性能，必须先
由用户决定是否放开某一项的精度锁定（`qk_pv` 单项就值 7,754µs）。

上游的 `indexer_score_topk_buffered` 分支**不适用**：它要求 `b_dim >= 64` 且
history >= 32768，我们是 b=16／8k，两条都不满足。

### 性能移植四批的实测效果

| 批 | 改动 | 是否在关键路径 | 实测 |
| --- | --- | --- | --- |
| 1 | `rope_cs` 拆 `rope_swap` + SPMD、`ROPE_CS_T_TILE` 8→S；`csa_rope_sign` 拆 `csa_row_offsets` + SPMD | 否 | 间接使 `merge_norm` 由 1,404.9 降到 927.4（0.66×） |
| 2 | `oproj_token_scale` → 按 token 块 SPMD | 是（第 4 位） | 窗口 1229.9→1178.7 |
| 3 | `idx_qr_proj_matmul` 复用整条 K 权重块 | 是（第 10 位） | 659.8→462.4（0.70×），**追平上游 1.0×** |
| 4 | `qproj_matmul` 的 `QPROJ_MM_N_TILE` 512→256 | 是 | 待验证 |

第 1 批当时是照函数表的 Exec% 挑的，没先算关键路径，**选点方法有误**；改动本身仍有价值
（修了 `ROPE_CS_T_TILE` 与 S 不匹配），且间接解开了 `merge_norm`。

### 排队任务期间不要改 kernel 源文件

`@pl.jit` 在编译时会**重新读源文件**定位函数定义。若在任务加载模型的过程中改动该文件，
JIT 读到的已是新内容，直接报
`OSError: @pl.jit could not locate function definition '<name>' in its own source file`。

实测：PTO steady 任务 17:50:53 启动、17:53:04 报该错，而 `decode_indexer.py` 的
mtime 是 17:51:53，正落在窗口内。这不是代码缺陷，是操作失误，重跑即可。

**规则：有任务在队列里排着或正在跑时，不要编辑它会加载的 kernel 源文件。**
改动要么等任务落地，要么先把任务取消。

### exit=130 的成因：`task-submit --max-time` 默认只有 300 秒

**已查明。** 队列默认值是

```
MAX_TIME=300    # 任务最大执行时间（秒），0=不限
--max-time N    任务最大执行时间(秒，默认 300，0=不限)
```

不加 `--max-time` 的任务满 300 秒即被 daemon 的 max-time watchdog 杀掉，队列记为
`completed (exit=130)`。DSV4 光加载 75 个权重分片就约 4.5 分钟，PTO 的 JIT 图捕获
再加 70 秒以上，**几乎必然超时**。

证据来自给父进程信号处理器加的诊断：

```
OFFLINE_SIGNAL SIGTERM(15) pid=2738064 pgid=2737696 ppid=1
  ... run.py line 850, in launch / time.sleep(2)
```

收到的是 **SIGTERM(15) 而非 SIGINT**，且 `ppid=1`——外层 bash wrapper 已被杀、python
被 reparent 给 init，正是 watchdog 杀进程组的形态。`130` 只是队列客户端侧的约定退出码，
不代表进程收到了 SIGINT。

该结论解释了此前全部现象：反复出现的 4m51s～5m8s 就是 300 秒；B=1／8／16 因加载加短
decode 刚好卡在线内而成功，B=24／32 图捕获更久而超时；Native steady 成功而 PTO steady
失败，是 PTO 的 JIT 把图捕获从 11s 拉到 72s；Native 开 EPLB 也失败，是 EPLB 子进程拉长
了启动；accuracy PTO 数据完整却 `exit=130`，是 decode 跑完后在收尾阶段撞线。注意
`--list` 显示的时长含排队等待，不等于执行时长，判断是否撞线要看任务日志的首末时间戳。

**此前我在本节给过三个归因，全部错误**：本地前台 python 干扰队列、多批 shell 循环死在
批次边界、PTO 后端才会挂。三次都建立在"exit=130 即 SIGINT"这个错误前提上，且没有先去读
队列的默认值。代价是至少八轮任务白跑。

**处置：跑模型的任务一律显式 `--max-time 3600`。** 纯 CPU 的短任务（lowering、trace
导出）默认值够用。

### 精度版与 Native **不是 bit 一致**（2026-09-24 首次测定）

此前精度版只验过输出 token 相同与 DSpark 接受计数相同，从未验过张量。新增的
`bitcompare` 命令在生产路径上挂 `CSAServiceRuntime.__call__`，同一步里先跑 PTO
再跑对照实现，各自 clone 该层输出后 `torch.equal`。

**基线先行：PTO 自比对 48/48 全部 bit 相同。** 这一步是必需的——`qkv_proj_rope.py`
有四处 `atomic=pl.AtomicType.Add`，其中 `kv_fp32` 两处在结构上有竞争（`KV_OK=2`，
同一 `kv_col0` 由两个 K 分片原子加到同一片内存，FP32 加法不结合）。实测证明当前形状下
它**没有**造成不确定性，因此不需要改 `KV_OK`，那个性能与可复现性的取舍不用做。
注意 `torch_npu.npu.set_deterministic_level(1)` 与 `HCCL_DETERMINISTIC` 管不到
PTO kernel 内部的原子加，自比对是唯一能确认这点的办法。

**结论（b=16、8k、eager、`--deterministic`）：**

| 对照 | bit 相等 | 不同元素 | `max_abs` | 显著 ULP |
| --- | --- | --- | --- | --- |
| PTO vs PTO | **48/48** | — | — | — |
| PTO vs Native | **4/48** | 1.77% | 0.031 | 27 |

开确定性前后数字完全一致（4/48、1.77%、0.031、ULP 27），差异不来自归约顺序随机性。
自比对全通过排除了 PTO 侧不可复现。**所以这是 PTO 与 Native 之间稳定、可复现的实现差异。**

三层深度对照（层 2／22／42）显示**差异不随深度单调增长**：`max_abs` 都在 0.016～0.031、
相对 Native 量级约 0.25～0.5%，不同元素占比 1.77%／6.31%／4.06% 波动而非递增。
即每层内部的固定量级差异，不是逐层累积。

**定位更正**：精度版的准确描述是"**输出 token 与 DSpark 接受行为与 Native 完全一致，
层输出在 BF16 末位有约 0.5% 量级的差异**"，不是"与 Native bit 级一致"。
`NATIVE_*` 那套按构造对齐（读 Native 编译出的 CCE 复刻累加次序）减小了差异但没有消除。
这不推翻任何已有验收——逐 token 相同在 B=1～40、8k／32k／131k、D01～D05、16 rank 上
都是实测的，差异小到不改变 argmax。

**指标教训**：首版 ULP 用 `view(int16)` 直接相减，BF16 位模式按有符号整数解释时跨零会得到
约 32768 的假差值，因此报出过 `max_ulp=32307`。已改单调序（负数映射成 `0x8000 - bits`），
并增加只在同号且绝对值不低于 scale 千分之一的元素上统计的 `max_ulp_significant`。
这与早先在 P bank 上误用饱和相对误差是同一类错误。

### 泳道图必须在 eager 下采

首轮泳道采集（`results/release_csa_perf_8k_20260924/swimlane/`）跟着 decode 的上线口径
用了 `FULL_DECODE_ONLY`，任务队列报 exit=0，但进程内抛了
`RuntimeError: Expected exactly one DFX window, captured 0`——**一个采集窗口都没开成**。

成因是采集窗口挂在 Python 层：`offline_begin_swimlane` 把 `CSAServiceRuntime.__call__`
换成在真实 CSA 调用前后执行 `pypto.torch.begin_dfx()`／`end_dfx()` 的包装。ACL Graph 下
decode 步是图回放，不再执行 Python forward，包装函数一次都进不去。这与本项目里
forward hook 在图回放期间不触发是同一件事。

用户 2026-09-24 明确：**泳道图需要在 eager 模式下采**。这在方法上也成立——芯片泳道记录的是
PTO kernel 内部各流水线（MTE／Vector／Cube／Scalar）的占用，属于 kernel 自身性质，
与它由图回放还是 eager 下发无关，变的只是主机侧下发路径。因此泳道**不跟随** decode
性能测量的 FULL_DECODE_ONLY 口径。`run.py` 已加守卫：`swimlane` 命令要求
`--graph-mode eager`，否则直接报错，与 `padding-capture` 拒绝 eager 相对称。

### 泳道采集结果（`results/release_csa_perf_8k_20260924/swimlane_eager/`）

重采成功：`captured=1`、`run_boundaries=1`、`dropped_run_boundaries=0`、915 个 AICore task、
1849 条设备切片、46 个具名 callable，kernel 取自本轮实际 JIT 产物
`_jit__decode_csa_tp1_attention_5_79_0r8`。口径：eager、b=16、layer 2
（`model.layers.2.self_attn.attn`）、8k bank、一次 96 token 的 CSA 调用。芯片 72 核（48 aiv + 24 aic）。

泳道图在 `swimlane/merged_swimlane.json`，用 https://ui.perfetto.dev/ 打开。该文件 2.7M 且可由
`chip_swimlane_records.json` + `name_map.json` 经 `swimlane-export` 秒级重生成，故只留本地不入库；
入库的是原始 `chip_swimlane_records.json`、`deps.json`、`converter_output.txt` 与 `swimlane_report.json`。

窗口总跨度 1213.58µs，**Exec/Latency 仅 55.97%**（逐任务均值：Exec 49.99µs，dispatch→finish 89.32µs），
即近一半时间不在算。按函数分成两类：

- **流水打满的大核**：`qk_pv_aic` 240.08µs / `qk_pv_aiv` 241.62µs（Exec% 98.2%／98.5%）、
  `indexer_score_topk_leaf_aic` 169.49µs / `_aiv` 169.45µs（97.7%／97.4%）。这四个是主要耗时来源，
  但本身效率没有问题。
- **被下发开销压住的核**：`merge_norm` Exec 29.27µs 而 Latency 275.97µs（**Exec% 10.6%**，
  head OH 244.61µs 中 NoC 传播占 234.25µs）、`indexer_topk_single_leaf_publish` 14.88／184.82µs（8.0%，
  传播 157.79µs）、`qr_rms_norm_quant` 7.96／90.54µs（8.8%，但开销在 Local 即 dcci+ack 73.66µs 而非传播）、
  `quant` 9.78／95.78µs（10.2%，Tail OH 83.09µs）、`oproj_token_scale` 82.50／135.06µs（61.1%，
  Tail OH 50.56µs）。

结论方向与 T2.1 一致：**PTO 的计算核效率没问题，损耗集中在任务下发与同步**，
对应 T2.1 里 AI_CPU 从 1,270µs 涨到 81,318µs 的观测。

**⚠ 这条结论已被 T2.7 推翻（2026-09-24 晚）。** 上面按 Exec/Latency 比值得出的
"损耗在下发与同步"是**口径错误**：DFX 窗口自带的边界同步开销被算进了 Latency，
而这些开销在 eager 下尤其大。用 `simpler_setup.tools.critical_path` 在**同一份**
level-4 泳道上重算：makespan 1.194ms 里 **compute 占 92.8%、stall 只占 7.2%**
（data-wait 6.9%、core-wait 0.4%），且 compute+stall 与 makespan 逐 tick 精确对齐。
即**瓶颈就在 kernel 自身的执行时间**，不在任务下发。后续优化据此改按单核
分流水线（MTE2/MTE1/CUBE/VECTOR）归因，见 T2.6/T2.8。

两条限定必须随数据一起说明：其一，DFX 窗口自带边界同步开销，这些绝对耗时不能当稳态性能；
其二，本轮在 eager 下采集，主机下发路径与图模式不同，head／tail OH 的**绝对值**偏大，
核内 Exec 与各核相对关系仍可用。要把调度开销继续拆细，可拿本轮 `deps.json` 单独跑
`sched_overhead_analysis`。

## 3. T3　场景扩展

| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T3.1 | 扩展离线 P 长场景（原 A3） | **已完成**：五档全部生成并通过 audit，每档四种输入。H4095 710M（1761 处非致命报告项）、H32767 3.4G（零）、H131071 13G（零）、H131072 13G（零，2¹⁷ 边界）、H131073 13G（零）。几何／布局／覆盖／history 校验全过，可交给 D 使用。四档对照坐实"只有 H4095 有跨副本差异"，成因见上方确定性一节 | — | 16 | **已完成** |
| T3.2 | 扩展 D batch（原 A4） | **已完成**：B=1／8／16／24／32 五档全出，每档 16 rank、21 个目标层、输出组恒为 4。PTO 捕获档位随 batch 增长——B=16 十三档、B=24 十九档、B=32 **二十五档（6…192）**，全部是 6 的倍数；Native 只出现在 `tokens1`／该 batch 最大值／`tokens256` 三个非 decode 形状上，不构成 decode 路径回退。**B=40 按用户 2026-09-24 的追加要求另测功能**（超出 `--max-num-seqs 32` 的上线口径，只验功能是否正常） | T1.9 | 16 | **已完成** |

### T3.1 现状：H4095 bank 已生成，但 audit FAIL（成因已查清，待用户定处置）

`h4095_bank` 四个 case 都已产出，audit 报 `computed-prefix data mismatch`，
错误全在 tp1／tp2／tp3（tp0 是基准）。2026-09-24 用
`dsv4_csa_bank_replica_diff.py` 测出差异分布，结论是**BF16 末位舍入噪声**：

| 张量 | 存储 | 差异元素 | ULP 中位 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| `layers.2.swa_cache` | BF16 | 1528/81920 | 1 | 6 | 60 |
| `layers.2.attn` | BF16 | 83/524288 | 1 | 3 | 13 |
| `layers.16.compressor.state_cache` | FP32 | 14336/20480 | 12677 | 106854 | 1057230 |
| `layers.2.indexer.compressor.state_cache` | FP32 | 3584/5120 | 8752 | 57606 | 589815 |

FP32 两项的 ULP 数看着大，但 FP32 的 12677 ULP ≈ 相对 1.5e-3，而 BF16 的
eps = 2⁻⁸ ≈ 3.9e-3——state_cache 是 FP32 存储、BF16 精度计算，中位差异只有
0.4 个 BF16 ULP。`max_abs_diff` 实测 0.024～0.045。差异自层 1 起每层都有，
swa／attn 集中在最后一页的第 24～30 行（最后 7 个位置），state 类则是约 70%
元素各差一点点——都符合"TP/EP 归约顺序不确定"的特征。H255 因序列短未显形。

**更正一条早先的错误判断**：先前记录的"最大相对误差 6.17／28.8，不是纯舍入"
不成立。那个指标用 `|a-b|/max(|a|,|b|)`，在近零值上会饱和到 2.0 附近，
度量本身有问题，不能据此断定非舍入。

**已处置（2026-09-24 用户批准）**：audit 原本要求四个副本逐 bit 相同，但 Native
在 TP+EP 下从不保证这一点，而 D 侧只读 tp0（`connector.py` 里 tp 固定为 0，且
DSA 的 KV 是 MLA 压缩潜变量、跨 TP 复制而非切分，每个副本都是完整一份）。
现在几何／布局／覆盖／history 这些 D 真正依赖的检查保持致命，跨副本数据比较
降为报告项并附 ULP 中位／最大值与 `max_abs_diff`。H4095 audit 已 PASS，
报告 1761 处非致命差异，bank 可交给 D 使用。

**H32767 的结果推翻了"随长度累积"的解释。** 2026-09-24 用当前代码生成
`h32767_bank`（3.4G，四个 case），audit **PASS 且跨副本报告项为零**——四个
TP 副本逐 bit 相同。H32767 比 H4095 长 8 倍，若差异来自舍入噪声随层数／长度
累积，它应该更严重才对，结果反而没有。

**重建结果：差异可复现，不是一次性异常。** 用当前代码重建的
`h4095_rebuild` 仍有 1773 个张量跨副本不同（旧 bank 是 1761）。所以先前
"更可能是那一轮生成时的特定情况"这个猜测被推翻。

现有事实：

| bank | 跨副本差异张量数 |
| --- | --- |
| `h4095_bank`（旧会话产出） | 1761 |
| `h4095_rebuild`（当前代码） | 1773 |
| `h32767_bank` | **0** |

H4095 稳定有差异、H32767 稳定没有，与长度无关，是 H4095 这个配置的特性。
**成因已定性（`task_20260924_025528_23757828067`）：归约顺序不确定。**
用 `--deterministic`（`torch_npu.npu.set_deterministic_level(1)` +
`HCCL_DETERMINISTIC=true`，按用户要求保留 AIV 展开模式）重建的 `h4095_det`，
跨副本差异从 1773 降到 **0**。日志确认 `OFFLINE_DETERMINISTIC level=1` 生效，
且没有出现 libhccl.so 里那条 `Deterministic do not support aiv` 告警。

| 配置 | 跨副本差异张量数 |
| --- | --- |
| H4095 默认 | 1761（旧）／1773（重建） |
| **H4095 + 确定性开关** | **0** |
| H32767 默认 | 0 |
| H131071 默认 | 0 |
| H131072 默认 | 0 |
| H131073 默认 | 0 |

**尚未解释的剩余问题**：**只有 H4095 这一档有差异**，H32767 与 H131071 在不开
确定性时本来就是 0。所以这既不是长度效应也不是普遍现象，而是 H4095 特有的，
成因无解释。已知的只有"开确定性开关能消除它"这一条，不要当成已经理解了。

**性能代价：只有一次粗测，不足以下结论。** 比较 `h4095_rebuild`（不开）与
`h4095_det`（开）两轮 prefill 的 `elapsed_including_io_seconds`：

| | n | 最小 | 中位 | 最大 | 合计 |
| --- | --- | --- | --- | --- | --- |
| 不开 | 4 | 7.05 | 8.28 | 10.39 | 33.05s |
| 开 | 4 | 7.08 | 11.19 | 11.41 | 37.49s |

合计约 +13%。**但每档只有 4 个 case、各跑一次、且含缓存落盘 IO**，最小值几乎
相同（7.05 vs 7.08）而最大值差得多，属噪声很大的单次观测。**不能据此建议是否
在生产配置里默认开启**——要下这个判断需要排除 IO、多次重复的稳态测量。

有价值的一点是两轮**输出完全相同**（都是 `[11799]`），确定性开关不改变结果。

audit 的降级判断不受影响且仍然成立：D 侧只读 tp0，Native 默认口径下不保证
四副本逐 bit 相同，所以不该拿它当门禁。现在多了一条：**确实需要逐 bit 可复现
时，`--deterministic` 是可用手段。**

H255 直接复用 `smoke_bank`，不必为矩阵重新生成。
`full_bank/plan.json` 已有 24 个场景的输入计划但**没有对应长场景缓存**，
更适合按一个新历史长度建一个新 bank 开始。

## 4. T4　P5 最终验收缺口

来自交接文档第 5 节。不要把当前短场景成功扩展为"P5 全部通过"。

| ID | 项目 | 仍缺 | 依赖 | 状态 |
| --- | --- | --- | --- | --- |
| T4.1 | F01 全模型接入 | **已完成**，三项判据都有实测：**其他上下文长度**——D 侧从只有 8k 扩到 **8k／32k／131k**，其中 H131071 的 `max_leaves=4` **首次执行多 leaf 归并路径**（8k 与 32k 的 `max_leaves` 都是 1，该路径此前从未被跑到），三档均 16/16 rank 输出与 Native 逐 token 相同；**其他 BS**——B=1／8／16／24／32／40 六档，均 16 rank 逐 token 相同，21 个目标层全命中，捕获档位随 batch 增长至 31 档且全为 6 的倍数；**必要层级数值验收**——层 2／22／42 的 `bitcompare`，结论是**不 bit 一致但差异稳定**（详见上方专节），`max_abs` 0.016～0.031、相对量级约 0.25～0.5%，不随深度累积 | T3.1、T3.2 | 16 | **已完成** |
| T4.2 | F02 实际 BS/GBS 与 graph | **已完成**，四项判据都有实测。**其他负载**与**图验证**见下方六档命中表（B=1～40，21 个目标层全命中，PTO 档位 1→31 且全为 6 的倍数，Native 恒定只占 `tokens1`／该 batch 最大值／`tokens256` 三个非 decode 形状、命中数恒为 1008=21 层×16 rank×3 档，不构成 decode 路径回退）；**图容量**见下方图显存表（PTO 常驻多占约 3.5GiB、捕获耗时随档位数放大）；**实际 DP 协调**见 T1.7 的 D01～D05（`rejected` 全 0、16 rank 逐 token 相同）。另有 T1.5 的 0 次重新捕获 | T1.9 | 16 | **已完成** |
| T4.3 | F04 EP/EPLB | **已停止（用户 2026-09-24 定）**，恢复需重新指派。停止原因是环境层面的算子缺失，不是集成代码问题：EPLB 本身能起来（`Dynamic EPLB is True`、`Policy: SwiftBalanceEplb (type=2)`、子进程拉起、warm-up 完成耗时 11s），但在重排后的第一次 MoE 前向报 `RuntimeError: aclnnGroupedMatmulSwigluQuantWeightNzTensorList ... not in libopapi.so`，调用栈为 `no_shared_forward_impl` → `_quant_method.fused_experts`。算子名中的 `WeightNz` 表明 **EPLB 让 MoE 选择了 NZ 布局的融合 grouped matmul**，而本机 CANN 9.0.0 的 `libopapi.so` 没有该算子。这与我们「NZ 一定不能开」的配置不矛盾——`weight_nz_mode=0`、`VLLM_ASCEND_ENABLE_NZ=0` 都已设，是 EPLB 路径自行选了 NZ 版本。同类限制此前还有 `fuse_norm_quant` 因缺 `aclnnAddRmsNormBias` 而关闭 | T1.8 | 16 | **已停止** |
| T4.4 | F05 真实 DSpark | **已完成**。四组 PTO/Native 同场景对照，每组三个原始计数完全相同，覆盖 4 个 batch 档（16／24／32／40）与 3 个上下文长度（8k／32k／131k）。接受长度稳定在 **4.79～4.89**、接受率 **95.8%～97.7%**、每步推进 5.79～5.89 token。全部取自 `llm.get_metrics()` 的 Prometheus 快照，**未注入任何假定值**（判据明确要求不强制注入平均 3.8）。详见下方表 | T2.4 | 16 | **已完成** |
| T4.5 | F06 稳定性与性能 | **稳态延迟／吞吐／显存已由 T2.3 给出**（p50 55.77→66.69ms、吞吐 0.85×、峰值显存 50.93→49.81GiB）。**剩余的长时间稳定性与异常／超时统计属用户 2026-09-24 定的「长稳先不管」**，恢复需重新指派 | T2.3 | 16 | **部分完成，其余已暂停** |

F03 的在线传输与网络故障恢复不是本轮前置条件——用户当前选择离线方式。

### T1.10：eager + embedding_tp 的容量口径缺口

实测两侧（PTO 与 Native）同样失败于：

```
ValueError: embedding_tp static capacity 192 < num_tokens 256;
increase max_cudagraph_capture_size or max_num_batched_tokens.
```

数字来源：

| 量 | 值 | 来源 |
| --- | --- | --- |
| `capacity` | 192 | `get_potential_max_tokens()` = `max_num_seqs(32) × QUERY_TOKENS(6)` |
| 实际 `num_tokens` | 256 | prefill 阶段的 `max_num_batched_tokens`（run.py 取 `max(256, batch*6)`） |

即 **`get_potential_max_tokens()` 只按 decode 的 `max_num_seqs × query_len` 计算，
没有覆盖 `max_num_batched_tokens`**，而 embedding 层是 prefill 与 decode 共用的，
`_forward_embed_tp` 的静态缓冲因此在 prefill 上不够用。

注意错误信息给的第二条出路是**反的**：`num_tokens` 正来自 `max_num_batched_tokens`，
调大它只会超得更多。

**不自行绕过。** 最直接的绕法是把 `max_num_batched_tokens` 压到 192，但那会改变 prefill
的分块行为，属于为了让测试通过而改被测配置；改 `get_potential_max_tokens()` 则是动 release
生产代码，且「embedding TP 的容量该按哪个口径算」属于设计决策。两者都需用户裁定。

### T4.2 六档捕获期命中（B=1～40）

| batch | 目标层 | PTO 档位数 | PTO 命中 | 最大档 | 全 6 倍数 | Native 档位 | Native 命中 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 21 | 1 | 672 | 6 | 是 | 1／6／256 | 1008 |
| 8 | 21 | 7 | 4,704 | 48 | 是 | 1／48／256 | 1008 |
| 16 | 21 | 13 | 8,736 | 96 | 是 | 1／96／256 | 1008 |
| 24 | 21 | 19 | 12,768 | 144 | 是 | 1／144／256 | 1008 |
| 32 | 21 | 25 | 16,800 | 192 | 是 | 1／192／256 | 1008 |
| 40 | 21 | 31 | 20,832 | 240 | 是 | 1／240／256 | 1008 |

三点结论：

1. **PTO 档位数与命中数随 batch 线性增长**（档位 1→31，命中 672→20,832），
   最大档恒为 `batch × 6`，**六档全部是 6 的倍数**，档位对齐机制在整个范围内生效。
2. **Native 命中数恒为 1008**，即 21 层 × 16 rank × 3 档，且这三档永远是 `tokens1`、
   该 batch 的最大 token 数、`tokens256`——都是非 decode 形状与预热，**不随 batch 变化，
   也不构成 decode 路径的回退**。
3. B=40 超出 `--max-num-seqs 32` 的上线口径，但**功能完全正常**，与其余五档一致。

### T4.2 图容量实测：PTO 的常驻显存代价

同配置（b=16、S=6、TP1、8k）下从各轮的 rank0 日志提取：

| 配置 | 捕获档位数 | 图捕获耗时 | 图占显存 | 可用 KV cache |
| --- | --- | --- | --- | --- |
| Native b=16 | 13 | 11s | 0.63 GiB | **19.83 GiB** |
| Native b=40 | 31 | 21s | 0.98 GiB | **19.79 GiB** |
| PTO b=1 | 1 | 19s | 0.48 GiB | 16.26 GiB |
| PTO b=16 | 13 | 72s | 0.83 GiB | **16.26 GiB** |
| PTO b=24 | 19 | 100s | 1.13 GiB | 16.26 GiB |
| PTO b=32 | 25 | 127s | 1.54 GiB | 16.25 GiB |

补齐 B=24／32／40 后另见两点：

- **PTO 的图显存随档位数近似线性增长**（13→19→25 档对应 0.83→1.13→1.54 GiB），
  而**捕获耗时从 72s 涨到 127s**——PyPTO 的 JIT 是每档编译一次的。对照 Native 的
  b=40：31 个档位只用 21s、0.98 GiB。即**冷启动代价随档位数放大的是 PTO，不是 Native**。
- 可用 KV cache 的约 3.5 GiB 差距在 b=1 到 b=32 之间**基本不变**（16.26／16.26／16.25 GiB），
  确认它是 PTO 的固定常驻开销，与档位数无关。

三点结论：

1. **可用 KV cache 少 3.57 GiB（−18%）。** 这不是图占的——日志顺序确认
   `Available KV cache memory` 在 `Graph capturing finished` **之前**打印
   （Native 17:49:31 对 17:49:44，PTO 16:47:39 对 16:48:52），即 KV cache 容量
   在图捕获前就已定下。3.57 GiB 是 PTO 侧的**非图常驻显存**（kernel 缓冲、JIT 产物）。
   这直接压低了能承载的上下文长度或并发数，是 F02 图容量一项的实质结论。
2. **图本身 PTO 多占 0.20 GiB**（0.83 对 0.63，+32%）。b=1 只有 1 个档位时仍占
   0.48 GiB，说明存在约 0.48 GiB 的固定开销，其余 12 个档位共增 0.35 GiB。
3. **图捕获耗时 PTO 是 Native 的 6.5 倍**（72s 对 11s），来自 PyPTO 的 JIT 编译。
   这是一次性启动代价，不影响稳态，但会拉长服务冷启动。

其余三项的既有证据：**其他负载**见 T3.2 的 B=1／8／16（21 个目标层全命中、13 个档位
全覆盖且均为 6 的倍数）；**实际 DP 协调**见 T1.7 的 D01～D05 六组（`rejected` 全为 0，
16 个 rank 输出与 Native 逐 token 相同）；**graph 验证**见 T1.5（0 次重新捕获）。

### T4.4 多场景接受统计（全部实测，无注入值）

| 场景 | drafts | draft_tok | accepted | 接受长度 | 推进/步 | 接受率 |
| --- | --- | --- | --- | --- | --- | --- |
| 8k b16 PTO | 20,960 | 104,800 | 102,400 | 4.885 | 5.885 | 97.7% |
| 8k b16 Native | 20,960 | 104,800 | 102,400 | 4.885 | 5.885 | 97.7% |
| 8k b24 PTO | 8,816 | 44,080 | 42,240 | 4.791 | 5.791 | 95.8% |
| 8k b32 PTO | 11,760 | 58,800 | 56,320 | 4.789 | 5.789 | 95.8% |
| 8k b40 PTO | 14,704 | 73,520 | 70,400 | 4.788 | 5.788 | 95.8% |
| 8k b40 Native | 14,704 | 73,520 | 70,400 | 4.788 | 5.788 | 95.8% |
| 32k b16 PTO | 5,872 | 29,360 | 28,160 | 4.796 | 5.796 | 95.9% |
| 32k b16 Native | 5,872 | 29,360 | 28,160 | 4.796 | 5.796 | 95.9% |
| 131k b4 PTO | 1,456 | 7,280 | 7,040 | 4.835 | 5.835 | 96.7% |
| 131k b4 Native | 1,456 | 7,280 | 7,040 | 4.835 | 5.835 | 96.7% |

四组对照里 PTO 与 Native 的三个原始计数**逐个相同**。这比输出逐 token 相同更强——
连 DSpark 每一步接受了几个草稿都一致。

一个反直觉的观察：**接受长度与上下文长度基本无关**（8k 的 4.885 最高、131k 的 4.835 居中），
说明 DSpark 的草稿质量不随历史变长而退化。

这批数据无需另跑任务：`spec_decode` 取数在 `4eb55049` 接上后，其后所有 decode 轮都自带。

### 性能差距的真正位置：**单次 kernel 执行时长，不是任务组织**

对性能版与上游做逐任务、按**非零事件**统计的对照（DFX level 4 会为每个块额外发一个
`kernel-duration=0` 的 orchestrator 事件，按事件数或含零事件的均值统计都会得出错误结论——
我因此一度误判为"我们每次更快"）：

| 任务 | 上游总 | 性能版总 | 上游次数 | 性能版次数 | 上游/次 | 性能版/次 | **单次倍** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qk_pv_aiv` | 6,731 | 11,846 | 48 | 48 | 140.23 | 246.79 | **1.76×** |
| `qk_pv_aic` | 3,310 | 5,876 | 24 | 24 | 137.90 | 244.83 | **1.78×** |
| `indexer_score_topk_leaf_aiv` | 3,077 | 7,015 | 48 | 48 | 64.11 | 146.15 | **2.28×** |
| `indexer_score_topk_leaf_aic` | 1,512 | 3,480 | 24 | 24 | 62.98 | 145.00 | **2.30×** |
| `qproj_matmul` | 875 | 2,064 | 24 | 24 | 36.44 | 85.99 | **2.36×** |
| `merge_norm` | 795 | 1,273 | 48 | 48 | 16.57 | 26.51 | **1.60×** |

**调用次数几乎完全相同**（48 对 48、24 对 24），**差距全部落在单次执行时长上**。这排除了
"多调了 kernel""并行度不足""任务拆分不同"这几种可能——同样的核数、同样的调用次数，
单次就是慢 1.6～2.4 倍。

**这解释了为何前两批移植几乎无效**：我改的是任务组织方式（规约链换 Vector `col_sum`、
分块 512→128、删掉一个任务），而差距在单个 kernel 内部的执行效率上。

最干净的证据是 **`merge_norm`：该函数我们与上游逐行相同、`MERGE_WORKERS` 都是 48、
调用 48 次，单次仍慢 1.60×**。代码一样、调度一样，差异只能来自更底层——数据布局、
访存模式或编译出的指令序列。

**候选方向（未验证）**：我们为绑定 Native 缓存而保留的 `INDEXER_KEY_BYTES` 页布局与
`aic_gather` 读法改变了访存模式，而访存模式直接决定 kernel 内部流水效率。验证需读
lowering 产出的 CCE 或逐处比对 PH001 类性能提示。**该方向需用户确认后再投入。**

### 性能版前两批的设备实测：**收益远低于预估，原推论被否定**

| | 上游 | 精度版 | 性能版（前两批） |
| --- | --- | --- | --- |
| 窗口跨度 | **728.0µs** | 1178.7µs | 1217.1µs |
| kernel 合计 | 26,822µs | 45,497µs | 44,184µs |
| 对上游 | 1.00× | 1.70× | **1.65×** |

**只拿回 1,313µs，而预估是 15,620µs。** 逐项看：

| 任务 | 上游 | 精度版 | 性能版 | 精→性 |
| --- | --- | --- | --- | --- |
| `indexer_score_topk_leaf_aiv` | 3,077 | 8,304 | 7,015 | 0.84× |
| `indexer_score_topk_leaf_aic` | 1,512 | 4,151 | 3,480 | 0.84× |
| `indexer_head_coefficients` | 0 | 104.5 | **0** | 已删除 |
| `qk_pv_aiv` | 6,731 | 11,892 | 11,846 | **1.00×** |
| `qk_pv_aic` | 3,310 | 5,903 | 5,876 | **1.00×** |

第 1 批（indexer 规约换 Vector `col_sum`）生效但只降 16%，换完仍是上游的 **2.3×**；
第 2 批（`ATTN_K_TILE` 512→128）**基本没动**，`qk_pv` 为 1.00×。

**结论：此前"92% 的差距在精度锁定代码里、换掉即可拿回"的推论被实测否定。**
真正差距不在数值写法上，继续移植第 3～6 批（预估合计仅约 1,500µs）不会接近 730µs。
要往下走必须先定位真正瓶颈，候选方向是 kernel 的 tiling／流水安排、裁剪版与上游在
任务依赖图上的结构差异、以及 `INDEXER_KEY_BYTES` 那套读取方式本身的代价。

## 5. T5　仓库卫生

| ID | 目标 | 完成判据 | 状态 |
| --- | --- | --- | --- |
| T5.1 | 决定 `offline_pd/run.py` 未提交改动的去留 | 两个开关都保留：`--graph-mode` 默认 `full_decode_only`，把 decode 默认口径从 eager 改成上线口径；`--recompute-scheduler` 默认关闭，保留它是因为在 T1.9 拿掉 DP 闸门之前，它是让 PTO 在 DP16 走图模式的唯一开关 | **已完成** |
| T5.2 | 提交 T1.1 的探针与结果 | `dsv4_csa_padding_probe.py` 与 `results/release_csa_padding_20260923/` 入库 | **已完成** |
| T5.3 | 更正 padding 计划里的 S0 描述 | 原文写"把四处改成显式报错"不可实现——PTO device 代码抛不出 Python 异常，越界读只会读到无关数据。已改为 CPU 复算索引公式，计划正文同步更正并补入 predict 结果 | **已完成** |
| T5.4 | 决定 `dsv4_perf_accuracy_20260827/` 的去留 | **用户 2026-09-24 裁定：不入库，永久保持本地。** 该目录含内网地址 `172.21.100.73`～`76`（`config.sh`、`docker_run.sh`、`start_decode.sh`、`start_proxy.sh`），而本仓库推送到公开 fork `github.com/nalinaly/vllm-ascend`。已加入 `.gitignore` 防止误 `git add`；用户未选择"脱敏后入库"，所以也不要改写地址后再提交 | **已完成** |

## 6. 保持暂停，不得自行恢复

以下项目用户此前明确叫停，**恢复需要用户重新指派**；
性能工作与 padding 工作本身都不代表解除暂停。

| 项目 | 未完成范围 |
| --- | --- |
| T4.3 EPLB | 用户 2026-09-24 明确停止。卡点是本机 CANN 9.0.0 缺 `aclnnGroupedMatmulSwigluQuantWeightNzTensorList`，EPLB 的 MoE 路径要求 NZ 布局融合算子。两次任务（PTO 与 Native 后端）均在 EPLB warm-up 完成后的首次 MoE 前向失败，说明与 CSA 用哪套算子无关。注意失败进程会挂死不退，需要 `task-submit --kill` 收回卡 |
| 剩余精度差异 | 旧基线正式权重 B40 step46 在冻结容差内仍有 attention BF16 末位差异。旧报告的 35 个 attention BF16 差异不是新 release 已复现的问题 |
| 正式 P0/P2 | 旧基线正式矩阵仅 B4/B40、H131071 通过，旧剩余 16 组；迁移后不能简单宣布只剩 16 组 |
| P3 连续轨迹 | 最新正式完整 100 步验收尚未完成；旧参考 100 步或旧基线 47 步不能替代新 release |
| P3 G07 | Prefix 共享。不在 padding 计划范围内 |
| P3 G08 | release 的 metadata 生产方式与旧 main 不同，需按实际机制验证，不能强行引入旧接口 |
| 非 6 倍数档位的 padding 处理 | 用户 2026-09-24 定：档位就固定在 DSpark+1（即 6）的倍数上，这项作为遗留事项先放着。不要主动去实现让 PTO 吃下 ragged 档位的能力——既不要改 kernel 走 TND，也不要加兼容层。`align_decode_capture_sizes` 开关保留，仅供将来恢复该项时造场景用 |

如果推进过程中遇到必须依赖这些行为的新阻塞：先提供具体失败证据、说明影响，再和用户讨论方案，
不要为了让测试通过自行新增冗余缓冲或改变 Native padding 协议。

## 7. 建议执行顺序

```
T1.2 ✅ → T1.3 ✅代码 → T1.4 ⏳验收中 → T1.5 ⏸待定落点 ┐
                                       └ T1.6 ────────┴→ T1.7(DP2) → T1.8(DP16) → T1.9
                                                                                    ├→ T2.1 → T2.3 → T2.4
                                                                                    └→ T3.2 → T4.2
T2.2 ✅（已完成，不再依赖 T2.1）
T3.1 可与 T1 并行（只用 P 侧，不依赖 padding）：H4095 ✅、H32767 ⏳ 生成中
T5.1～T5.4 ✅ 全部完成
```

关键路径是 T1.4 到 T1.9。T2 和 T4 的多数项都压在 T1.9 之后，
因为在 PTO 拿不到图模式之前，性能数字和 graph 相关验收都没有意义。

**当前唯二需要你拍板的**：T1.5 的落点（见上），以及 T2.5 的 PyPTO
`_resolve_compiled` 处置。其余条目要么在跑、要么依赖关系明确。
