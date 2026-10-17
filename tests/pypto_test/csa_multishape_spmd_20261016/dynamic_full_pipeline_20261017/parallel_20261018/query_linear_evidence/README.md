# Query 反量化连续 head 区间候选

基于正式 `9d5327e5`（含 DQ 整波分组，不含 QA 小 M16），只改 `query_projection_stages.py` 的工作分配。T1..512 调度模型与数学 AST 检查通过；唯一一次完整符号 CPU lowering 通过，未新增 Tile、事件或 PIPE_ALL。候选和 reference 已冻结，本子任务未修改正式包/依赖、未执行设备；主会话后续收取的设备结论附于文末。下面的 prepare 数是工作量，不能当作性能结果。

## 历史排重

查 MASTER、VALIDATION_LOG、动态 ATTEMPTS 及既有 DQ 归档，找到 2026-09-26 的 [qdequant_contiguous 补丁](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_baseline_20260926/perf_qproj_upstream/rejected/qdequant_contiguous/candidate.patch>)。该方案把数学 Tile 从八个 token × 单 head 改成单 token × 连续 16 heads，固定 `Q_DEQUANT_HEAD_TILE=16`，并重写规约、scale 和 RoPE 路径；task_20260926_172807_15086503832 整层 p50 861.06 μs，未保留，见 [VALIDATION_LOG:3991](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md:3991>)。它是实际负证据，但不是本次保留八行×单 head 数学、仅改变动态工作区间的同一方案。

另检查旧 `q_dequant_24_workers` 和 2026-09-29 `q_dequant.patch`，分别改变 worker 数和 sync_start；当前已采用的 [整波分组](../query_dequant_group_20261018/README.md) 只取 g4/8/16/32/64。未找到同一 floor 区间划分和跨行 chunk 的已有设备试验；此结论仅限已查本地记录。

## 分配、边界及数学

当真实设备 `T%8==0` 且原 g4 任务数 `(T/8)*16>=WORKERS` 时，设 `N=(T/8)*64`，AIV w 领取 `[floor(N*w/48), floor(N*(w+1)/48))`。这里正式 WORKERS=48；源码使用 WORKERS，不冻结 T。区间沿每 64 heads 的行边界拆 chunk，计算 `tg=row_tile*8`、`hg=max(start,row_base)-row_base` 和 `heads_in_row=min(end,row_base+64)-max(start,row_base)`。每个 chunk 只准备一次 scale/cos/sin/swap，再走原两级逐 head pipeline。

入口保证每核区间非空。各区间首尾相接、并集恰为 `[0,N)`，跨行切分不重叠；所以每个有效 token/head 唯一写一次。每个 head 仍依次执行原 INT32→FP32、激活 scale、权重 scale、512 列 RMS 和 RoPE，不跨 head 归约。相邻 head 的写回列片段互斥，改变 owner 不改数学依赖。

T8/T16 保留 g4，以免扩大到 48 个准备任务；任何含尾行的 T 全部保留原 g4 条带，完整块的 rsqrt 与尾块的 recip(sqrt)、有效读写范围均不变。T0 空工作。只保留一个完整数学体及原尾数学体，没有复制第二套计算；修改的唯一数学循环字段是 pipeline 的 trip 变量名。

外层 QB 等待仍覆盖全部 24 AIC，全部 48 AIV 在本核 DQ 完成后发布 QUERY，Attention 仍等全部 48 QUERY owner；未改变 GM alias、发布、获取或代际协议。S1/S5/mixed Q 都使用实际 T，未假定 S6。

## 有限模型结果

[audit_source_and_owners.py](./audit_source_and_owners.py) 直接执行冻结源中的 Scalar 调度 AST，去掉 Tile 数学后枚举 T1..512；同时比较原/新数学 AST，单独检查 T0。全部 token/head 唯一写、最大每核 head 数与 token/head 数不增，总 prepare 及最大每核 prepare 均不增，尾行和 T8/T16 的逐 owner 记录完全一致。非整除时具体 owner 可换工作，不声称每个固定 owner 都不增加。

| T | 总 prepare 原→新 | 每核最大 prepare 原→新 | 每核最大 head 原→新 |
| --- | ---: | ---: | ---: |
| 8 / 16 | 16 / 32 → 不变 | 1 → 1 | 4 → 4 |
| 24 | 48 → 48 | 1 → 1 | 4 → 4 |
| 32 | 64 → 48 | 2 → 1 | 8 → 6 |
| 40 | 80 → 52 | 2 → 2 | 8 → 7 |
| 48 | 48 → 48 | 1 → 1 | 8 → 8 |
| 96 | 48 → 48 | 1 → 1 | 16 → 16 |
| 144 | 144 → 60 | 3 → 2 | 24 → 24 |
| 168 | 336 → 66 | 7 → 2 | 28 → 28 |
| 192 | 48 → 48 | 1 → 1 | 32 → 32 |
| 384 | 48 → 48 | 1 → 1 | 64 → 64 |
| 512 | 1024 → 96 | 22 → 2 | 88 → 86 |

共 57 个 T 的实际 owner/chunk 记录变化，20 个 T 存在长度 1 chunk，例如 T40 的 owner28、row24、head0。完整明细见 [ownership_cases.json](./ownership_cases.json) 和 [source_and_owner_audit.json](./source_and_owner_audit.json)。固定区间会减少部分点的重复准备，但加入端点整除、chunk min/max 和分支；它还改变缓存访问与各 AIV 进入后继阶段的时刻。没有假设同步/Scalar 开销为零。

## 唯一 CPU 与真实生成物

[cpu_command.sh](./cpu_command.sh) 是已执行的唯一完整命令：主只读 `reproduce_cpu.sh`、`--strategy long`、本目录明确 controls 和独立 source。exit 0，58 个符号轴、130 个 ABI 项、single MIX、sync_start，无样例 shape。`cpu_v1/result.json` 原件和 [cpu_result.json](./cpu_result.json) 记录结果。没有重编基线；使用已采用 DQ 的 [既有 CPU](../query_dequant_group_20261018/cpu_v1/result.json)，其 33 源逐文本等于本 reference，controls 完全相同。

[audit_generated.py](./audit_generated.py) 与 [generated_audit.json](./generated_audit.json) 确认：

- 全链 2370 条 Tile 分配及 DQ 155 条分配的顺序、类型、shape、地址相同；DQ UB 地址末端 131072 B，全 AIV UB 163872 B。AIC Mat 475136 B、Left/Right 各 65536 B、Acc 131072 B，均不增。这是 `max(addr+声明字节数)`，不冒充运行时 live 峰值。
- 完整 AIC C++ 去注释和 SSA/label 名后相同。DQ 完整及尾行数学 PTO 去路径/行号/SSA 名后相同，保留常量、Tile 和地址；准备数据与双槽生命周期未扩大。两份 8×512 INT32 载入槽仍为 65536/81920，各 16384 B。
- 全部硬件调用计数相同；GM TWAIT/TNOTIFY、局部 flag、FFTS、DSB/DCCI、屏障的顺序在 SSA 归一后相同，保留 event ID 与 pipe。初次原始参数文本对照出现 DQ 后跨核事件的 SSA 编号顺延；它不代表新增同步，已保留原始不等字段及归一方法。

长度 1 的真实 lowering 已核对：PTO 将主循环限制为 `2*floor(heads_in_row/2)`，再以余数==1执行最后一个 head。实际 C++ 48263 行计算主界，48265–48266 在循环前预置 EVENT6/7，48269–48275 在 n=1 时跳过双头体；48768 后排空对应事件，48779 检查余数，48805 只读取 `hg+0` 的 8×512，没有 `hg+1` 读写。必要片段保存在 [pipeline_one_head_cpp.txt](./pipeline_one_head_cpp.txt)，主/尾数学生成段也已留存。这是源/PTO/C++控制检查，未替代设备长度 1 精度与活性验证。

可供主会话追加一个最小边界 probe：现有 `run_native_query_probe.sh` 的 `--history 8192 --batch 8 --query-tokens 5 --causal-extension-reference --spmd-source <本目录/source_candidate>`。S5/B8 的 T40 完整整除8但不整除24，owner28区间为 `[186,193)`，拆成 row16/head58..63 六头与 row24/head0 一头，确实命中长度1路径；prepare80→52、每核最大head8→7。此处只提供规格，未提交任务；主会话计划的 S6 长 B24/B28 不代替该奇数长度边界。

## 冻结与归档

[source_reference](./source_reference) 是正式 9d5327e5 的 33 源；[source_candidate](./source_candidate) 仅上述一文件变化，差异见 [candidate.patch](./candidate.patch)。两份源从唯一 CPU 启动起冻结。没有叠加 QA，没有再次 CPU、NPU 或共享安装修改。

可以归档本目录除 `cpu_v1/` 外的报告、脚本、JSON、必要生成片段、日志及两份源；原始全量 PTO/C++ 留私有 `cpu_v1/kernel/`。路径大小清单由 `write_archive_manifest.py` 生成，不做内容校验摘要。当前仅支持“候选可编译且数学/资源/同步未见新增变化”；是否采用仍待主会话安排的完整设备验证。

## 主会话后续设备结果：不采用

主会话收取两档完整矩阵均 PASS；同进程五窗完整 L1 如下，不能由 prepare 减量推算实际收益：

| 档位 | 正式DQ整波 / μs | 连续区间 / μs | 相对参照 | 更快配对 |
| --- | ---: | ---: | --- | ---: |
| 128K/B24/S6 | 1174.38 | 1181.24 | 慢0.584% | 1/5 |
| 128K/B28/S6 | 1337.28 | 1328.84 | 快0.631% | 3/5 |

任务分别为 `task_20261018_050316_27283926583`、`task_20261018_050316_27292720517`。本候选不采用、不组合、不以B28建立特化，也不追加性能扩测。主会话已有 S5 单 head probe 在运行，仅用于归档边界，本子任务不再提交或等待新设备任务。源码继续冻结。上面的 CPU/模型结果保留其原本范围，不被改写成性能证明。
