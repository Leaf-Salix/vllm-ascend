# QA v2：仅原两份 M16 尾块区间的 constexpr 分核

私有候选基于正式9d5327e5的33份执行源，包含已采用DQ，未包含v1 QA；参照保存在 [source_reference_9d](./source_reference_9d)，候选在 [source_candidate](./source_candidate)。两份源码于第一次CPU启动时冻结。只修改policy.py、entry.py、cube_schedule.py、vector_schedule.py、qkv_stages.py五文件，见 [source.diff](./source.diff)。不修改正式源、共享依赖或v1证据，不自行上卡。

v1的T<=48策略不采用。主会话已报告其独立完整精度通过，但长B4改善3.624%（543.60→523.90μs，4/5配对改善），长B8回退0.577%（679.66→683.58μs，1/5改善）。v1全部源码、CPU和审计留在同级 `ultra_qa_small_m16_20261018`；这些结果不等于本v2已验证。

## 策略、动态性与同步

原QA在16<T<32时没有M32完整块，只有两份M16尾块；v2仅在这个连续token区间将它们分给两组AIC，不增加原数学或逻辑权重请求。Host `select_policy`使用已有Tensor shape中的tokens选择 `qa_m16_parallel=16<tokens<32`，不读取device值、不按batch或S6样例白名单，不推断请求等长。

策略字段默认False，`QA_M16_PARALLEL`在entry的两个入口、cube/vector及q_a_stage均显式接线。旧controls省略此字段时仍默认False。False时constexpr消去新数学与新等待分支，只保留原完整q_a_stage/qa_tasks；True时只保留全局M16 helper及对应等待，避免v1新增device分支扰动未命中区间的编译。

True的数学仍源级动态：`blocks=ceil(T/16)`、`logical=block*8+column`、`owner=logical%24`；任务按24跨波次领取，循环上界不写死为两块或T24。每行仍完整K4096，K256×16原序累加；create_tensor、valid_shape、matmul_acc首段初始化和M16×N128写回复用原尾块数学。正式选择区间只有16个任务，row0归核0..7、row16归核8..15，各核一块。

Vector开启时等待 `min(24,8*ceil(T/16))` 个QA物理owner；显式强制True到较大T时也能覆盖循环跨波次全部owner。False的原M32/M64 owner公式不变。Cube原PHASE_NORM获取、所有24核PHASE_QA发布、后继Indexer工作、其它notify/wait与72核begin均保留。qa容量仍pad16，QR按pad8读取仍在该容量内；未给容量尾行增加初始化承诺。

新增QA参与核会推迟自己后继的Indexer投影，完整L1净影响必须实测。区间内总逻辑权重读取不增只是选区间依据，不是性能结论。

## 有限源码审计

[audit.py](./audit.py) / [audit.json](./audit.json) 已通过。去除新字段、constexpr接线及新helper，并选择False分支后，五文件的其余AST与参照一致；新helper数学与原M16尾块在局部变量改名后AST相同。T0..96及127/128/129/144/168/192/384检查了有效行×八列恰好一次、pad16写入边界、真实owner与等待集合；Host另外覆盖Q上界1/6/8、历史8192/131072，新增开关之外的全部controls与9d相同。这里没有导入设备库或运行NPU。

## 最多两次完整 CPU

复用既有含DQ、同33源的CPU基线：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/query_dequant_group_20261018/cpu_v1`

其source_candidate与本参照33文件逐文本相同。两次候选CPU均采用基线同控长G6/panel64/score1024/prefetch1/roots1、OA128/OB96、attention-query-metadata off。False的 [controls_false.json](./controls_false.json) 故意省略新字段，验证旧调用的默认关闭；True只多一个 `QA_M16_PARALLEL=true`。不提供样例shape，不把CPU通过当设备精度或性能。

以下为本轮复现命令；输出目录独占，不能覆盖重跑：

```bash
bash '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/reproduce_cpu.sh' '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/cpu_false_v1' \
  --attention-query-metadata off --spmd-source '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/source_candidate' \
  --controls-json '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/controls_false.json' > '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/cpu_false.stdout' 2>&1
bash '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/reproduce_cpu.sh' '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/cpu_true_v1' \
  --attention-query-metadata off --spmd-source '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/source_candidate' \
  --controls-json '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/controls_true.json' > '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018/cpu_true.stdout' 2>&1
```

关闭和开启各一次CPU均已PASS，见 [关闭结果](./cpu_false_v1/result.json)、[开启结果](./cpu_true_v1/result.json) 及各自原始stdout。两者均130根ABI、58符号维度、单MIX且sync_start；未提供样例shape，未执行NPU。两次编译额度已用完，源码保持冻结。

## 实际生成物核对

[codegen_audit.json](./codegen_audit.json) 保留可复算比较方法和精确路径。False与9d同控基线的原始PTOAS C++、AIC物理C++、AIV物理C++、orchestration四份文件，仅删除整行注释后逐字完全一致；不归一变量名、event编号、地址或空白。前三份各有3171行PTO追踪注释发生变化，orchestration原文也相同。原始PTOAS文件净长度差14字节只是多处注释变化相抵的结果，不能表述为“仅14字节内容变化”。实际C++完全相同，因此本控制组合没有v1旧分支事件排序扰动。

False的2370条allocation与基线按序全同。True为2325条：删除原QA两套dense/尾块生成路径，仅保留15条全局M16分支allocation，qkv以外分配按序不变。三份生成物的各空间最大地址末端相同：Acc131072、L1475136、Left65536、Right65536、UB163872字节。这是 `max(addr+声明形状字节数)`，不是累加占用或运行时live峰值。

True的原始PTO 432–516行直接使用运行时arg132计算 `ceil(T/16)*8`，按24跨波次；没有T24静态展开，也没有原M32/M64 QA数学分支。K循环仍0..16、step2，每对K256按K0/K128/K256/K384执行四次K128 MMA，首段init和动态valid_rows保留。实际M16 Acc在0；两份L1 X在196608/204800、权重在65536/131072；Left四份K128在0/4096/8192/12288，Right两槽在0/32768。源码AST、owner模型与实际130根/58轴共同支持后续独立设备验证，不预记精度或性能通过。

自动Host选择已实际调用验证（history131072、max_query_tokens6、requests=ceil(T/6)）：

| T | QA_M16_PARALLEL | 原Indexer分组 |
| --- | --- | --- |
| 16 | False | G2 |
| 17、18 | True | G2 |
| 24、30、31 | True | G6 |
| 32、48 | False | G6 |

因此T18/T24/T30无需手工policy override。上述False生成物一致性限定于实际比较的长G6/OA128/OB96 controls；其它策略保持源AST相同，不冒充已有逐产物对照。

主会话已提交v2独立128K/B4(T24)与B5(T30)：`task_20261018_050316_2729667380` / `task_20261018_050316_27300510000`。本目录当前只交付CPU与源码证据，不包含这两个任务尚待收取的设备结论。未由本代理提交或新增NPU任务。

归档见 [ARCHIVE_FILES.json](./ARCHIVE_FILES.json)：小文本与33+33源可复制；两次完整PTO/物理C++仅列本地路径和大小，不复制大生成物。
