# d56 无 phase 消费者发布审计与独立 AIC 候选

2026-10-18。参考为主 PAR 的 `adopted_source_d56c9eb6`，不叠加同目录上级
正在设备验证的 HC 四 guard。先完成以下只读审计，再按主会话授权，仅在
本目录 `source_candidate/` 删除两个 AIC 阶段发布，完整源码保持动态。

## 历史范围

主日志 `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md:15698–15720` 的
`release_sync_v1` 是改所有阶段的通用 release 实现：PIPE_ALL→DCCI→gm
fence/DSB→TNOTIFY，等待、phase 和数学不变。该轮完整精度通过但未显示
收益，128K/B16 原 task 871.80μs、SPMD 949.32μs；未采用且恢复旧 helper。
不把它与前轮总时差当同进程因果收益。本候选不改通用 `phase_sync.py` 或
`signals.py`，只针对已有完整数据交接的两个无消费者阶段。

## 信号无人等，不等于业务 release 无用

两个 schedule 共找到七个没有 phase wait 的名称，详见
`phase_inventory.json`；组输出按 base+group 对齐，不因循环变量不同误判。

| 阶段与 d56 位置 | 数据路径、业务维护与本轮处理 |
| --- | --- |
| SCORE_CUBE，cube_schedule.py:375 | 整阶段槽无人等。每份 FIX 输出经 SCORE_READY 被双 AIV 等待、读取，再由 MTE2 ACK；每 leaf 在 helper 返回前等 min(steps,2) 个剩余 ACK。唯一 score-transfer 消费已经完成，下一阶段的 plan/query 获取原样保留。可删除这条整阶段发布。 |
| ATTN_CUBE，cube_schedule.py:394 | 整阶段槽无人等。helper 最后一次 PV_READY 后不假设 AIV 已完成；紧随其后的 ATTN_VECTOR wait 等全部实际 query owner 的双 AIV。PV_READY→AIV PV 读取/归并→packed 写回及发布→AIC ATTN_VECTOR 获取，才允许 O_A 读 packed。删除 orphan 不删除这个后继交接。 |
| GATES，vector_schedule.py:518 | 无跨核 phase wait，但 pre 的同核 mix_norm 随即读取；仅凭槽无人等不足以删除这里的业务排空与缓存维护。理论上可只去信号写而保留完整业务 release，本轮不改。 |
| COMB，vector_schedule.py:527 | 无直接 phase wait；最终 HCpost 仍需要 comb。后继 QR 发布和全 48 SCORE_VECTOR 链可提供合并 release 的分析入口，但涉及本核数学/缓存可见性，本轮保持。 |
| IDX_COMMIT，vector_schedule.py:630 | state 写回仍需完成，且前面的全体 pool-reader 等待必须保留。后继 IDX_NORM 发布可作为合并分析入口；norm 不读 state，但本轮不移动或删业务 release。 |
| MAIN_COMMIT，vector_schedule.py:794 | 同理，state 与后续 norm/cache 输出不同；可分析并入 MAIN_OUTPUT，但本轮不动 commit release 或 pool-reader 集合。 |
| OUT，vector_schedule.py:954 | 无核内 phase wait；是最终输出写回边界。不能仅因信号无人等就假定 kernel 返回已替代全部 MTE3/缓存维护，本轮完整保留。 |

因此只有前两个进入本轮候选；其余“无 phase wait”只是信号图事实，不是
删 fence 的许可。AIV 全部源码、所有真实 producer、输入获取、代际和最终
OUT 发布均逐字保留。

## 两处具体 happens-before

Score 的依据为 `indexer_score_stages.py:375–378` 的 FIX READY 与逐 leaf
ACK drain，以及 AIV `:641–649` 的 READY wait→FIFO load→MTE2 ACK。
ACK 允许 AIC 复用两槽，不代表 AIV TopK 数学已完成；后者仍由完整
SCORE_VECTOR→TOPK→ATTN_PLAN 交接负责，候选没有删这些等待。Score 与
Attention 的 FFTS ID 域不同，Score ACK 在 helper 中排尽才切阶段。

Attention 的依据为 `attention_stream_stages.py:196–198` 的最后 PV store/
FIX PV_READY、`:390–391` 的 AIV PV wait，以及后续归并/packed store。
AIC 在 `cube_schedule.py:399` 仍等 `2*min(T,24)` 个 ATTN_VECTOR owner。
每个 q 按 q%24 分组，这一集合包括全部实际 query 的双 lane；plan 的失效
块在两侧一致跳过，但 query 尾部依然完成初始化/归并与 packed 发布。零
query 的组没有 ring 数据需要移交。没有用“helper return”代替真实交接。

两条 phase 信号原本不承担获取；后面的 plan/query/ATTN_VECTOR wait 仍
包含原 fence/cacheinvalid/fence。AIC 的实际 GM 输出是 DMA/FIX 写入的
FIFO/ring；这些 payload 的消费者本来就经 FFTS 读取，不依赖事后 orphan
信号。该结论以原 helper 的 READY/ACK 和合法 workspace 别名合同为前提。

## 有条件消费者与禁止误删项

- RAW 只在 `HC_CONSUMER_REDUCE=false` 发布并消费；row-dependencies
  切换等待 owner 计算，未消灭真实数据依赖。
- OA/O_QUANT 只在 `OUTPUT_GROUP_PIPELINE=false` 的整阶段路径使用；
  true 路径使用 44..51 与 52..59 的每组槽，已有精确 owner guard。
- META 只由动态 planner 发布，两个 schedule 仍等同一真实 planner。
- IDX_HADAMARD 的 wait 在 compact count 为零时为空，COEFF 在 groups
  为零时为空；这不是全路径 orphan。其它 row-owner 等待也会因行数为零
  成为空集合，不能推广为阶段永久无人消费。
- IDX_SCALE、QUERY、SCORE_VECTOR 有明确全 48 角色等待，不能按某次
  数学量为零裁掉发布；输入与其它前驱的传递可见性也不可随之删去。

## 实现、潜在编译变化与验证

`prepare_candidate.py` 从 d56 的 33 源复制，仅删两条调用并补中文说明。
去除同样两条调用后 AST 完全相同，32 份其它源码逐字相同；无新 policy
flag、无新模型、无新 barrier。原槽编号和 phase0 的 72 核推进不变；旧
SCORE_CUBE/ATTN_CUBE 值可留在信号区，已无任何等待读取它们。

删除 orphan 同时删除其 PIPE_ALL/DCCI/DSB，可能改变相邻 helper 的编译器
事件分配、资源复用或流水排空。源级依赖证明不能预言这些变化的性能：应
检查最终 C++ 与物理片上地址，再由完整设备精度和同进程 L1 决定。减少
48 个发布仅为计数，不是性能收益。

唯一一次完整 CPU lowering 已通过，使用主 D/reproduce_cpu.sh，参数
`--attention-query-metadata off --spmd-source 本目录/source_candidate`；
完整命令见 `cpu_command.txt`，输出为 `cpu_v1` 与 `cpu_v1.log`。该候选
没有新增 policy 或 controls 开关，设备侧只需 source override。编译后
33 份 candidate 源与编译保存的 `cpu_v1/source` 逐字相同，现已冻结。

## CPU 与物理生成物核对结果

`cpu_v1/result.json` 为完整动态 PASS：58 个符号维度、56 个要求传入的
root dimension、130 个 ABI tensor 参数，单 MIX、未提供样例 shape。
策略为 G6/N64/SCORE1024、balanced、OA256/OB128、output group=true、
HC row=true、query metadata=false、HC consumer reduce=false。CPU 结果
不包含设备精度或性能结论。

对照使用已有 compact-priority `cpu_v1`，其保存的 33 源与 d56 逐字一致，
controls 与 ABI 也一致，因此没有重新编译基线。`codegen_audit.py` 只
读取两侧已有产物，结果在 `codegen_audit.json/log`：

- 2365 条 `pto.alloc_tile` 按顺序比较完全一致：角色、物理 byte 地址、
  完整 tile type（含 shape/layout）、来源文件名与行列均相同。这里是
  allocation 记录数，包含复用与重复视图，不是 2365 个独立物理缓冲区。
  AIC acc/mat/left/right/vec 分别 66/271/73/141/2 条，AIV vec 1812 条；
  本次没有新增片上空间或地址重分配。
- 两侧全部 FFTS PTO 指令记录（pipe、ID、mode、来源、顺序）仅源码
  路径前缀不同；Score 每 leaf ACK drain 保留。AIC 与 AIV 各自的
  `set_flag`/`wait_flag` 调用序列逐条相同。
- AIV C++ 仅去掉注释后全文相同；AIV 所有发布、获取、数学及维护保留。
- AIC phase34/38 的 word offset 常量 313344/350208 均消失，其它
  TWAIT 未变；删除处相邻代码见 `codegen_excerpt.txt`。候选 C++
  `:17440–17476` 展示 ACK drain 后进入 plan 获取，`:18015–18057`
  展示 PV_READY 后进入 ATTN_VECTOR 获取，没有将 helper 返回当作
  AIV 已消费完成。
- `perf_hints.log` 仅源码前缀不同，其余全文一致；原 38 条 PH-MR-001
  仍存在。本轮没有修复或引入这些降槽提示，也不据提示判断性能。

以下是函数体中静态出现次数，不是动态执行次数或实测耗时：

| AIC C++ 项 | d56 | 本候选 |
| --- | ---: | ---: |
| TNOTIFY | 15 | 13 |
| TWAIT | 13 | 13 |
| wait_flag_dev / ffts_cross_core_sync | 4 / 3 | 4 / 3 |
| PIPE_ALL | 15 | 13 |
| DSB_DDR | 55 | 51 |
| whole DCCI | 27 | 25 |
| set_flag / wait_flag | 691 / 691 | 691 / 691 |

这些证据证明此次 lowering 实现了预期删除，且已检查的片上分配和硬件
事件未改变。没有声称 AIC 删除发布后的完整指令时序相同；没有验证最终
机器码运行。所有数学与其它 wait 的参数、控制流由 source AST 对照约束，
后续仍由主会话的完整精度及同进程 L1 对照判断。本子任务不上卡、不改
正式源、不做 hash。

## 证据文件

- `phase_inventory.json`：两个 schedule 的 publish/wait 源级清单。
- `prepare_candidate.py`、`candidate.patch`、`source_audit.json/log`、
  `source_manifest.json`：两条删除、33 源直接文本与 AST 核对，路径和大小。
- `source_candidate/`：已编译的冻结 33 源，独立从 d56 出发。
- `cpu_command.txt`、`cpu_v1.log`、`cpu_v1/result.json`：唯一完整 CPU 命令与结果。
- `codegen_audit.py/json/log`、`codegen_excerpt.txt`：已有生成物直接对照与邻域。
  JSON 同时列出原始 PTO/C++/report 路径和大小；未复制 baseline 大生成物。
