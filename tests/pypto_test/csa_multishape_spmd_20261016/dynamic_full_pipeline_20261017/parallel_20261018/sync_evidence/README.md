# 动态 CSA：Attention 入口传递依赖候选

本工作线从 `593c5d25` 的独立 worktree 开始，只修改本工作区。只读参考为正式仓
`dynamic_full_pipeline_20261017` 的历史 phase clock、生成代码和优化日志。
不提交、不 push，不改共享 PyPTO/Simpler 或测量驱动。已读取本 worktree 的
`AGENTS.md`。候选策略默认关闭；完整精度和性能验证前不宣称收益。

## 候选及偏序依据

`attention_transitive_dependencies` / `ATTENTION_TRANSITIVE_DEPENDENCIES` 仅折叠
Attention 入口四处已由既有阶段顺序蕴含的等待：AIC 的 `PHASE_QUERY`，AIV 的
`PHASE_ORIGINAL_CACHE`、`PHASE_MAIN_OUTPUT`、`PHASE_ROPE`。没有新增 phase、缓冲、
task 或 host metadata 回读。数学 helper、循环和次序均未修改，保留 `pl.dynamic`
及 device Q/L/position/page table。

每个 AIV 在发布 `PHASE_SCORE_VECTOR` 前已无条件发布上述四种业务完成信号。
每个 AIV 在 TopK 前无条件等待全部 48 个 `PHASE_SCORE_VECTOR`，然后才发布 TopK
并执行 plan。因此每个 `PHASE_ATTN_PLAN` 发布者的已获取闭包包含全部旧业务
生产者。AIC 和 AIV 仍等待真实 plan owner，随后仍执行原有业务
`fence/cacheinvalid/fence`，才进入 Attention。单写者 512B 隔离、代际、发布端
流水排空/cache 维护、信号读前 DCCI 与 FFTS READY/ACK 均保留。

无有效请求和尾块只改变数学 helper 的工作量，不跳过这些发布和 Score 集合。
T>0 时至少有一个 plan owner；T=0 时没有 query 行可读，候选不增加访问。
本候选不把普通 phase 同步称为全卡 barrier，也不根据历史就绪时间删依赖。

## 历史定位证据

历史 `diag_h131072_b16_output_groups_v1/phase_clock/phase_clock.json` 中，Attention
AIC 的累计等待调用中位数由 213 增至 273，即 12 个必要 plan owner 加 48 个
QUERY owner；AIV 从 510 增至 562，即 12 个 plan 加 12+16+12 个旧业务 owner。
8K/B32 同目录为 AIC 297→369（24+48），AIV 623→727（24+24+32+24）。
候选对应 B16 每次共 3,072 个重复信号读取、B32 共 4,992 个；这些是协议调用
计数，不能换算为性能收益。历史插桩的区间包含循环、cache 和写时戳开销，
不作为性能验收。已有 `release_sync_v1` 发布端精简无收益并撤回，本候选不
重试或复用那项变更，也不删除任何必要获取。

## 验证方式与当前状态

`check_transitive_dependencies.py` 直接解释本工作区两份 schedule 的 AST。
每个核独立维护已获取 phase 闭包，成功等待才合并生产者发布时的闭包；
在每个 plan 发布及每个 Attention 数学入口断言全部必需生产者已被获取。
这比只检查随机交错恰好已完成更强，但仍不模拟设备 cache、FFTS 或数值。
用例覆盖 T=0、全 dummy、混合 Q、1/7/8/9 尾块、七档 T、满/跨 48 AIV 波次，
两种开关与多个交错 seed。设备精度、图重放和完整 L1 另行执行。

主 agent 提供统一隔离源码加载和同进程三方对照入口；仅通过该入口做完整
CPU 编译与 NPU 验证。CPU 编译不申请卡。NPU 必须 `task-submit --device auto`，
先比较完整输出、TopK/score、cache/state、图重放、保护和代际，全部通过后
才采五个轮换顺序完整 L1 窗口。原分 task、基线 SPMD 和本候选共享一进程。
Native 仍有继承差异，本工作线不放宽阈值，不把原 PTO 等价写成 Native PASS。

新增控制文件：`controls_enabled.json` 用于 CPU constexpr，`policy_enabled.json`
用于完整设备策略覆盖。所有结果、失败与任务号将在下面追加。

## 2026-10-18 首轮验证

- 实际编排 AST 闭包证明：144 例通过，结果为 `dependency_proof.json`。
- `kernel_long_v1/result.json` 与 `kernel_medium_v1/result.json` 均通过完整符号
  编译；未传样例 shape、未运行设备，58 个动态维度，单个 MIX 提交，原有五份
  权重 BYPASS 与内部 key 视图 ABI 核对通过。两份生成 C++ 均保留 42 个 TWAIT
  站点；plan 的等待之后仍有完整 PIPE_ALL/DSB/DCCI/PIPE_ALL/DSB 获取序列。
- Python 语法编译与 `git diff --check` 通过。候选源码于设备提交后冻结。
- 128K/B16：`task_20261018_003350_299816415546`，输出为正式仓
  `D/parallel_20261018/sync_h131072_b16_v1`。
- 8K/B32：`task_20261018_003400_30016459311`，输出为正式仓
  `D/parallel_20261018/sync_h8192_b32_v1`。

两任务均由 `task-submit --device auto --max-time 1200` 提交，指定本 worktree
源码、开启候选 policy JSON、`baseline_source` 参考与 `--wait-backoff`，每档五个
L1 窗口。两档均已完成，原始结果保留在上述目录。

## 完整真机结果：功能通过，未证实收益，不采用

两档的 A→B→A 三代逐 bit 对照、原 task/旧 SPMD/候选三方 graph、末窗完整
输出/TopK/score/cache/state、Native 与工作区保护、72 核代际均通过。每份
SPMD 窗口都核对为 72 核、单个 task；下表为各五窗完整 incore 首尾的中位数。
同进程中除新候选开关外，两份 SPMD 的所有策略项一致。

| 规格 | 原分 task μs | 基线 SPMD μs | 候选 μs | 候选相对基线耗时降低 | 相对原 task 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16/S6 | 882.04 | 912.02 | 915.70 | -0.40% | -3.82% |
| 8K/B32/S6 | 986.30 | 1014.36 | 1011.18 | +0.31% | -2.52% |

去掉数千次重复槽读取没有形成稳定的完整 CSA 收益；两档方向不一致，且幅度
约 0.4% 以内。因此保留候选为默认关闭，不新增按规格启用分界，不扩大七档。
没有一档达到相对原分 task 降低 30%。不能从上述小幅差值进一步声称硬件缓存
压力、寄存器或调度中哪项抵消了轮询节省；本轮没有采候选插桩作归因。

`result_summary.json` 保留两档全部原始样本、任务号、基线/候选策略关系和完整
验证清单；`summarize_results.py` 从原始报告重新计算中位数与比例，并拒绝未
完成的任务或非五窗/非单 task 的 SPMD 记录。

`source_audit.json` 证明关闭候选后的两份调度 AST 等于 `593c5d25`，29 份数学、
信号及辅助源码直接对照未变，并记录全部源码路径和字节大小。四份修改源码与两个控制
JSON、三个小型验证/汇总脚本、证明/汇总 JSON 和本 README 是可审阅交付；
`kernel_long_v1` / `kernel_medium_v1` 共约 156MB 的完整编译产物仅留本地，不应
纳入源码补丁。`ruff check`、Python 语法检查与 `git diff --check` 通过。

主 agent 已确认本轮不合入正式源码、不扩七档。`candidate.patch` 保存四份候选
源码相对 `593c5d25` 的差异，包含关闭分支和默认关闭策略；未提交或 push。
全部源码在提交设备任务后保持不变，后续只补充本地验证/汇总文本。
两份设备任务保存的 33 个 Python 源码快照均与当前候选逐字节一致。
CPU 完整报告另复制为顶层 `cpu_long_result.json`、`cpu_medium_result.json`，便于
只归档小型证据；无需复制对应完整编译目录。`candidate.patch` 反向应用检查通过。

## 复现命令

以下脚本路径是本轮主 agent 提供的统一入口，引用的 `baseline_source` 是
`593c5d25` 动态源码快照，保留 `pl.dynamic`，不是固定 shape 的生成代码。
复现须使用全新输出目录，下面 `*_repro` 名称也只可使用一次。

```bash
sync_worktree=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018
sync_driver=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
sync_source="$sync_worktree/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
sync_evidence="$sync_worktree/tests/pypto_test/ultra_sync_20261018"

python "$sync_evidence/check_transitive_dependencies.py"
python "$sync_evidence/audit_source.py"
bash "$sync_driver/reproduce_cpu.sh" "$sync_evidence/kernel_long_repro" \
  --attention-query-metadata off --spmd-source "$sync_source" \
  --controls-json "$sync_evidence/controls_enabled.json"
bash "$sync_driver/reproduce_cpu.sh" "$sync_evidence/kernel_medium_repro" \
  --strategy medium --attention-query-metadata off --spmd-source "$sync_source" \
  --controls-json "$sync_evidence/controls_enabled.json"

task-submit --device auto --max-time 1200 \
  "bash '$sync_driver/run_device.sh' '$sync_driver/parallel_20261018/sync_h131072_b16_repro' --history 131072 --batch 16 --spmd-source '$sync_source' --policy-json '$sync_evidence/policy_enabled.json' --source-matrix-reference '$sync_driver/parallel_20261018/baseline_source' --wait-backoff --l1-windows 5"
task-submit --device auto --max-time 1200 \
  "bash '$sync_driver/run_device.sh' '$sync_driver/parallel_20261018/sync_h8192_b32_repro' --history 8192 --batch 32 --spmd-source '$sync_source' --policy-json '$sync_evidence/policy_enabled.json' --source-matrix-reference '$sync_driver/parallel_20261018/baseline_source' --wait-backoff --l1-windows 5"

# 仅汇总本轮jobs.json引用的已完成原始任务，不运行设备。
python "$sync_evidence/summarize_results.py"
```
