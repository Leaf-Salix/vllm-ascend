# 小 T 的 QA 全局 M16 分工候选

候选仅在实际 T<=48 时按全局 M16 块编号分工，让每个块的八个 N128 列任务分配给不同 AIC；T>48 保留原 M32/M64 完整块与 M16 尾块。源码已冻结于 [source_candidate](./source_candidate)，来自当前正式执行源的 33 份 Python 文件，只有 qkv_stages.py/vector_schedule.py 改动。对照为 [source_reference](./source_reference)，完整文本差异见 [source.diff](./source.diff)。不叠加其他候选，不修改正式源或依赖，不自行上卡。

## 分工与同步合同

新 helper 的任务数为 8*ceil(T/16)，logical=block*8+column：

- row=(logical//8)*16，column=(logical%8)*128。
- owner=logical%24；T<=48 至多24个任务，每个正式 AIC 至多一个 M16 块。
- T24 的 row0 由核0..7生产，row16 由核8..15生产；T48 的三块由0..7、8..15、16..23生产。
- T0 不进入数学循环；全部24个 AIC 的 PHASE_QA 发布仍保留。
- QA 容量仍至少 pad16(T)，工作量只看真实 T，不随较大的工作区容量增加。

只把原 _q_a_worker 的 DENSE_M 改成16不能解决 T24：floor(24/16)=1。只把该 floor 改 ceil 也不够：其原 tail 起点 full_rows+m_group*16 仍把 row16 分给 group0，使第二组全空。本候选使用统一 global block ordinal，避免这个问题。

原 M16 tail 的 create_tensor、K256×16 pipeline、valid_shape、matmul_acc(init_cond=k_block==0)、assemble 数学 body 原样复用，仅局部 tail_* 改名为 small_*。完整/不满16行均走同一原有效行合同。T>48 的旧 helper 与策略分支 AST 不变；没有 split-K、权重转换、累加顺序或初始化变化。

Cube 的 PHASE_NORM 仍等待 [0,T) 的全部有效八行 owner，足够覆盖所有新分工；所有 PHASE_QA 发布的位置和数量不变。Vector 新增 T<=48 分支，等待 ceil(T/16)*8 个真实 owner；T24 从8扩大到16，T48从8扩大到24。若漏改这个等待，QR 就可能读取尚未生产的尾行。其余 wait/notify 保持原样；T49 恢复原 M32+M16 分工和原8核 QA等待。

[AST/整数审计](./audit.json) 已通过：只改上述两文件；新数学与原尾块仅局部名差异；整个其余 qkv AST 以及 vector AST 还原分支后完全一致；T0..96 加127/128/129/192/384有效行×8列块恰好一次、全部输入/输出在真实行与 pad16 容量边界内、实际 owner 均被等待；T49及所有检查过的 >48 点分工相同。检查不导入设备库。[独立只读审阅](</data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/qa_small_m16_review_20261018/README.md>) 也未发现源级阻塞；该审阅不替代设备验收。

## 实际代价与已有阶段证据

新增参与 QA 的 AIC 原本可以更早进入同核后继 IDX_PROJECT，不是整层空闲核。QA 的并行度提升可能延后该核的 Indexer 压缩/query 投影，完整 NPU 结果才能判断净影响。

按源码逻辑，完整1024×4096 BF16权重每个 M 块读取一次：

| T区间 | 旧 M 块 | 新 M16 块 | 逻辑权重读取变化 |
| --- | --- | --- | --- |
| 1..16 | 一份M16尾块 | 一份M16 | 不变 |
| 17..31 | 两份M16尾块 | 两份M16 | 不变 |
| 32 | 一份M32 | 两份M16 | 2倍 |
| 33..48 | 一份M32+一份M16 | 三份M16 | 1.5倍 |

这些是源码层请求量，不能当作真实 HBM 字节或据此推算延迟。T24总数学/逻辑读取相同，只改变owner；T48总逐行数学相同，但增加一轮权重请求及更多真实 QA 依赖。

现有正式 d253 的长B16/T96诊断 [hc_tail_clock_audit](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/hc_tail_clock_audit/README.md>) 三轮为：

| 端点/μs | A0 | B1 | A2 |
| --- | ---: | ---: | ---: |
| COMB 最晚发布结束 | 73.04 | 72.18 | 69.56 |
| QA 最早发布开始 | 76.46 | 73.06 | 72.26 |
| QA 最晚发布开始 | 79.70 | 76.08 | 75.98 |

这说明该已有前段里 QA→QR 有约束，不能把 COMB 更快直接当主机会；但 T96 已有24个 QA owner，本候选不改这个 T。现有完整 phase_clock 集合没有 T24/T48，所以不能从上表证明本候选的关键路径收益，也不将插桩时差当无插桩性能。

## 历史排重

已查 MASTER 的 Q_A/QR动态迁移记录（14873起）、历史 ATTEMPTS、[dense projection审计](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/dense_projection_d253_audit/README.md>)、[HC_MIX_QPROJ_EXPERIMENTS](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_QPROJ_EXPERIMENTS.md>)。未找到 T<=48 统一M16 global block ordinal+ceil(T/16)*8 owner 的同方案设备实验。

旧 HC mix 驻留是 T96 的 AIV 数学，文中“主Q N128/L0预取”是 Q_B INT8 投影，不能当作 QA 小M 分工失败。动态QA迁移的 M64/M16 同名 accumulator 类型冲突属于原编译失败，本候选新增独立 small_acc 名称；没有修改依赖规避该问题。

## 唯一 CPU 与冻结

源码在首次完整 CPU 命令发出后冻结。唯一一次全链 CPU 已通过，见 [result.json](./cpu_v1/result.json) 与 [原始 stdout](./cpu.stdout)：58个符号维度、130个编译 ABI 项、single_mix=true，未提供样例 shape，未执行设备或测量性能。下列命令是原始复现记录，已经执行，不重复运行覆盖已有输出。

```bash
python '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/audit.py'
bash '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/reproduce_cpu.sh' '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/cpu_v1' \
  --attention-query-metadata off --spmd-source '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/source_candidate' \
  > '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/cpu.stdout' 2>&1
```

只执行一次全链CPU。主会话已提交独立长B4/B8对照：task_20261018_044522_413498115985、task_20261018_044522_413502612248。本报告尚未纳入其设备结果。参照和候选仍为 d253，未叠加后来正式采用的 DQ 规则。原 task 降低30%的目标不因源码/CPU通过而视为达成。

## 生成物与片上资源

[只读生成审计](./codegen_audit.json) 使用既有 `ultra_attention_tick_clock_20261018/cpu_v1` 的原始 PTO/C++ 作基线；其33份 source_snapshot 与本目录 source_reference 逐文本相同，controls 与130项 ABI 也完全相同。这里读取的是 `kernel/ptoas/dynamic_csa_full.cpp`，未使用后续 tick 插桩物。完整文件仅保留原路径，不复制大生成物。

本次默认完整 CPU controls 为 G6/balanced、OA256/OB128、attention-query-metadata off；它完整包含动态 T 分支。设备入口会按历史/批量选正式策略，不能把这次 CPU 分配审计冒充所有 constexpr 策略的实际生成物。

原2,365条 allocation 按生成顺序均保留：其中 qkv 121条、其它阶段2,244条；物理地址、类型、形状和静态有效维度均一致。比较仅归一内联编号、SSA编号/版本及源路径/行号，动态有效行关系另由 AST 和整数 owner 审计覆盖。新增小T分支15条，UB分配没有新增。

| 空间 | 两份生成物的最大地址末端/字节 |
| --- | ---: |
| Acc | 131072 |
| L1/Mat | 475136 |
| Left | 65536 |
| Right | 65536 |
| UB/Vec | 163872 |

表中是 `max(addr + 声明形状字节数)`，不是把全部 tile 相加，也不是运行时 live 峰值。小T分支的实际地址为：Acc M16×N128 FP32在0；两份 X M16×K256 BF16在196608/204800，两份权重在65536/131072；Left的四份M16×K128在0/4096/8192/12288；Right K128×N128的两槽在0/32768，下一K256 pair继续复用。没有扩大既有地址末端。

生成 [PTO](./cpu_v1/kernel/ptoas/dynamic_csa_full.pto) 430–516行与 [C++](./cpu_v1/kernel/ptoas/dynamic_csa_full.cpp) 6285–6656行保留真实 T<=48 判断、`logical//8` 行块/`logical%8` 列块及动态 valid_rows。K pipeline每次展开两个K256，各拆两个K128 MMA，顺序为K0、K128、K256、K384；每M16块共32次K128 MMA，首块 init 语义、累加顺序及最终M16×N128写回沿用原M16数学。Vector的对应 owner 等待在 PTO 6349–6408行，新增 `8*((T+15)//16)`，后两条原策略仍在。

## 事件变化的边界

QA之前86条同步调用顺序相同。QA新增小T分支有59条静态事件/屏障调用；这些是生成文本记录，不是每次执行的动态次数，更不是性能收益。T>48保留源码分支的272条事件调用按pipe方向/类型计数相同，但局部set/wait排序与event ID有变化，因此不能声称T>48生成指令或时序完全不变。

QA之后4,716条本地事件、FFTS、pipe_barrier、DSB和DCCI调用，在SSA与局部event ID归一后顺序相同；实际仅紧接的三个投影M→FIX配对编号由5/6/7顺延为6/7/0，具体行号列于审计JSON。QA最终发布仍在C++ 8155–8186行保留原drain、DCCI ENTIRE、DSB和信号写入；没有源码级增删任何其它notify/wait或代际begin。CPU通过与这些记录仅支持该候选可进入独立设备验证，不证明同步活性、设备精度或收益。

精简归档清单见 [ARCHIVE_FILES.json](./ARCHIVE_FILES.json)。源码、源码diff、两个小审计及CPU结果/日志可归档；PTO/C++完整生成物只列路径和大小，本地保留。

## 补审：旧 M32/M64 分支的事件归属

本节只读上述两份现成C++，未重跑原审计、CPU或设备。对保留分支解析常量值，并按实际一一对应关系归一局部变量及event编号后，两路各261条非同步语句均同序相同，包含地址计算、tile声明、TLOAD、TEXTRACT、MMA、TSTORE及循环条件；两路各136条事件调用仍在相同的相邻非同步语句之间。以下范围均指前述原始 `kernel/ptoas/dynamic_csa_full.cpp`。

| 旧分支 | 原物行号 → 候选行号 | 实际变化 |
| --- | --- | --- |
| M32（49<=T<128） | 6182–6928 → 6659–7405 | 8组相邻事件调用换序：入口seed一组、dense K循环seed一组、dense末尾drain一组、分支末尾drain一组；dense及M16尾块内各两组双目标发布交换 |
| M64（T>=128） | 6930–7676 → 7407–8153 | 4组相邻事件调用换序：dense及M16尾块内各两组双目标发布交换 |

共同的双目标交换发生在同一Right `TEXTRACT`之后：原先先发 `MTE1→M`、再发 `MTE1→MTE2`，候选交换这两个相邻set。例如M32原6426/6427→候选6903/6904，M64原7173/7174→候选7650/7651；尾块为原6798/6799→候选7275/7276及原7545/7546→候选8022/8023。它们源pipe同为MTE1，目标分别是MMA消费与下一轮MTE2复用许可，不能笼统称为“彼此独立、必无代价”。其它交换只在连续seed/drain调用内发生；没有事件跨过任何业务搬运或数学语句。

四个保留K循环仍为0..16、step2，源级stage=2及实际两份L1 X/权重槽、Right两槽、Left四份K128 tile都不变；没有发现pipeline降深、搬运先后、K累加或物理地址变化。结论仅限这些生成物的指令安排：相邻事件换序仍可能改变流水获许可的时刻，设备效果不能由文本证明。

若小T结果值得继续，原七档范围内最少补两个未命中新分支的代表：128K/B16（T96，M32 dense及该路seed/drain）与128K/B28（T168，M64 dense加多个M16尾块）。使用当前冻结d253参照/候选独立对照，不叠加后来DQ。该选择覆盖两条旧策略及上述事件变化类别，不等同七档全部验收；原七档没有M32尾块点。若采用承诺扩展到任意动态T，还需另以一个49..127且非32整除点（例如T72）实际覆盖M32尾块，不能用源码相同代替该路径验证。本节仅给验证建议，不提交任务。
