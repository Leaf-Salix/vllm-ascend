# 过程065：真实CSA后到通知命中与32B候选

2026-10-11。私有vLLM覆盖驱动`a07a5091`、通知对照驱动`3d7e05fb`，
PyPTO实验驱动`a7ae94b5e`；生产生成器仍为`21a3fbcfe`，Simpler `5085a6ce0`、PTOAS `0c21f984`未改。
[完整版本及增量](versions.json)。本轮得到真实CSA覆盖证据；32B通知未形成收益，不采用。

## 真实128K/B16覆盖

任务`task_20261011_233739_300023629127`，单卡0，退出0并释放卡。
第2层HC_pre+norm+CSA+HC_post、TMR/CANN9.2/NZ2/atomic0，两轮完整执行，第二轮改变hidden输入。
复用过程055私有覆盖驱动，用现行通知协议的trace位补充统计，不改队列、派发、ACK、核选择或数学表达。
没有开启chip swimlane、PMU或参数dump。专属插桩写时钟和GM，会影响到达次序，不能当作正常性能测试。

| 项目：每个核上的task实例 | 第一轮 | 第二轮 |
| --- | ---: | ---: |
| 总执行数 | 973 | 973 |
| gated early-dispatch | 584 | 588 |
| 上项进入安全前缀路径 | 584 | 588 |
| 派到忙核后的交接资格 | 331 | 424 |
| 实际同核交接 | 286 | 339 |
| 其中gated后继 | 109 | 133 |
| 其中普通ready后继 | 177 | 206 |
| 通知提交 | 676 | 653 |
| 完整回调后交接 | 6 | 36 |
| 提交通知但未交接 | 670 | 617 |
| 具备pending资格但未交接 | 45 | 85 |

两轮合计1329次通知、42次回调后交接，命中约3.16%。这是插桩运行中的次数比例，
不是可节省时间的比例；也不能把1287次未命中全部认定为纯浪费，因为通知可能等待了原计算。
625次同核交接中583次没有经过完整回调。9项输出/缓存/状态逐bit一致，各31项保护区通过。

必须区分一个task“接收前驱交接”和“向后继交接”，否则会误读MIX覆盖。两轮合计：

| 成员 | 执行数 | 接收同核交接 | 向后继交接 | 通知提交 | 回调后向后继交接 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qr_proj_matmul` | 48 | 0 | 48 | 0 | 0 |
| `qk_pv_aic` | 48 | 0 | 48 | 0 | 0 |
| `indexer_score_topk_native_pair_aiv` | 96 | 6 | 0 | 96 | 0 |
| `qk_pv_aiv` | 96 | 0 | 0 | 96 | 0 |
| `_proj_b_mm_nz_kernel__2` | 128 | 104 | 56 | 93 | 21 |
| `proj_b_act_hc_post_0` | 96 | 0 | 0 | 96 | 0 |

后到新增交接主要集中在O-B投影自身的SPMD block接续。MIX QK/PV的AIC作为前驱已经向O-A交接，
不能因为它自身“接收交接”为0便说MIX没有作用；Score AIV也确实收到6次同核交接。
完整逐成员表见[by_function.csv](coverage/by_function.csv)。

45/85条pending未交接中，11/7条前驱没有异步尾部；其余34/78条前驱均执行了通知。
这些后继入口均晚于已记录的前驱完整完成，但没有AICPU实际派发时间戳，
不能进一步证明它们都在计算结束后才派发，也不能把它们全部当作漏掉的重叠机会。
尾等待包括新增通知DMA，本轮不据此声称某个CSA task是Scalar bound。

## 32B通知候选

既然通知调用很多，先做一个局部且可隔离的减负候选：两个DMA burst由64B减到32B。
每核独占的64B缓存行和L1/UB scratch仍保留，通知只读首个64位字；交替值、全部流水汇合和轮询不变。
没有按task名称关闭能力，没有新增early-resolve或派发路径。
PTO-ISA A2/A3的`TPrintCopyMat2GM`以`Numel*sizeof(T) >> SHIFT_BLOCK_BYTE`计算burst，
`TLoadInstrGm2L1`也以`BLOCK_BYTE_SIZE`计算相邻burst地址；小用例实际验证32B通知可运行。

任务`task_20261011_234650_339784719759`，卡0，先做AIC/AIV loops=4两组输入及图重放，再做CSA。
AIC后到交接15/16次，重放18次；AIV两组没有命中，重放有3次。输出、fanin、保护区及退出通过。
不能把本轮AIV未命中当作32B不能交接，或把回调计数当作原计算重叠时长。

128K/B16、第二层真实权重、HC_pre+norm+CSA+HC_post，TMR/CANN9.2/NZ2/atomic0，无诊断；
预热5次后各30次单次NPUGraph重放，轮换次序，reset在NPU事件计时之外。单位μs：

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通PTO | 973.14 | 990.39 | 1030.04 |
| 原64B完整通知 | 975.20 | 994.48 | 1030.66 |
| 32B完整通知 | 977.50 | 1028.13 | 1870.62 |

候选较64B mean多33.65μs，14/30配对更快，中位差+0.45μs；相对普通mean多37.74μs。
1870.62μs长尾显著影响均值，其根因尚未定位，不能直接归因于DMA长度。
典型配对也没有显示收益，因此不采用，不为寻求好结果继续重复本候选。
9项状态及换输入重放逐bit一致，每路31项保护区通过；任务退出0，设备已释放。
[原始结果](timing/results.json)、[90条样本](timing/samples.csv)、[配对差值](timing/paired_comparison.json)。

## 数据定义及复现

覆盖[results.json](coverage/results.json)保留函数映射、配置、汇总与逐成员计数。
两个`round*_records.csv`保留全部1946条16列原始记录，不重复提交大型稀疏trace Tensor。
`handoffs.csv`和`pending_without_handoff.csv`分别保存交接关联及未交接列表。
CSV中的clock均为原始tick，未假定频率：

- `tail_flags`低8位7是trace完成标记，不是通知event编号；bit8为已发通知，bit9为回调后的成功交接。
- `prefix_flags` bit0为pending交接资格，bit1为tagged args ABI，bit2为gated安全前缀能力。
- `mode_flags` bit0为实际Wait回调，bit1为实际本核前驱上下文，bit2为当前task有异步尾部。
- `full_tail_end`包括通知完成，不是独立的原Vector/Cube计算终点。

恢复[versions.json](versions.json)的私有环境。覆盖驱动需要本进程trace地址，取得单卡后才编译；
该产物不能跨进程复用。`repro/run_csa_late_coverage_128k_b16_20261011.sh`使用过程063普通产物作数值参考。
后续增加出/入两个方向的统计仅离线重读同一trace，没有再次占卡。

32B先在私有PyPTO仓source `../env.sh`，设置私有PYTHONPATH，运行：

```bash
python tests/st/runtime/scheduling/task_boundary_callback_probe.py --variant notice32 \
  --compile --work 4 --compiled "$SCALAR_OVERLAP_ROOT/artifacts/notice32_probe_compiled_20261011"
```

再到私有vLLM仓重新source `../env.sh`，以`tmr_csa_callback_compile.py --variant notice32`
编译CSA候选，`--reuse`指向过程063 compiled目录。将[repro/](repro/)脚本复制回私有根的`artifacts/`
后通过task-submit执行。增量patch使用`git apply --unidiff-zero`。只保存生成通知片段，避免复制整份CSA计算体。

本轮没有新Native、七档或16卡测试，公共默认未改，局部Wait及诊断回退保持。
下一步以真实通知调用多、命中少的证据审查通知入口及资源合同能否精简；
不通过跳过MIX或低命中task缩小功能目标，也不把当前白名单、未知事件范围与其他规划器缺口视为理论禁止。
