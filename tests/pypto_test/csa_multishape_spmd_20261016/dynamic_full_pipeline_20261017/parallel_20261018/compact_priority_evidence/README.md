# 动态 CSA：优先准备 Indexer compact key 与 scale

基线593c5d25，2026-10-18。只在独立worktree修改两份动态编排源码；
没有改生产数学helper、共享依赖或安装，也没有将输入shape固化。
完整过程已在主工作区`parallel_20261018/compact_priority_evidence/`归档；
同源码七档均完成当前S6精度/图/协议验收，详见[SEVEN_CASES.md](SEVEN_CASES.md)。

## 问题与实际改动

当前阶段时钟显示，Indexer compact norm已在主QB前就绪，但其Hadamard
仍排在独立主压缩投影之后；AIV的compact量化/scale又排在主Query及主压缩
写回之后。候选将两个完整阶段块前移：

1. AIC：主QB→compact Hadamard→主压缩投影→Score。
2. AIV：Indexer query量化/系数→compact key量化/scale→主Query/主压缩→Score。

AIC可以用既有两槽提前生产Score，随后若AIV尚未到达，第三槽仍须等双AIV
ACK。没有提前等待整段Score，也没有取消任何发布、获取、cache维护或FFTS。
前移IDX_NORM/Hadamard等待可能推迟其它工作，不假定所有形状都获益。
这不同于历史prioritize_indexer：后者未提前这里的compact Hadamard/scale。

`make_candidate.py`从冻结基线移动整块语句，并核对顶层AST多重集合：
Cube68条、Vector159条均相同；仅两个调度文件变化。数学、owner和槽容量未变。
完整33个Python源码在`source_snapshot`冻结后才提交设备，不在运行中修改。

## 验证与适用边界

- `schedule_model.json`：144组实际调度AST解释通过，覆盖G2/G6、空/奇数/
  mixed/padding及多wave，HC行依赖和输出组流水开启。这是phase偏序模型，
  不模拟真实FFTS、缓存或数值。
- `cpu_v1/result.json`：不提供样例shape，58动态轴、130 roots、single MIX
  完整CPU编译通过。long G6/N64/1024；CPU输出策略256/128与设备B16的
  128/96不同，不能称整份分配完全相同。
- `review.md`：独立审阅数据依赖、cache/state别名合同、两槽Score跨leaf
  排空及空lane ACK，未见新增等待环。指定前段/Score的Left/Right地址与
  当前长B16基线相同；未把输出阶段策略差异掩盖为全文件等价。
- `task_20261018_013723_19521519782`：128K/B16/S6完整PASS。
  输出/TopK/score/cache-state、A→B→A、同址图与历史/页表重放、保护区及
  72核代际均通过，再进入同进程三方五窗完整L1测量。

## 长历史同轮结果

单位μs。完整L1为incore最早开始到最晚结束，包含内部等待及阶段空隙；
不含外围orchestrator，不使用Native性能作为分母。原task、593旧SPMD、
候选共享正式第4层W8A8权重及根输入，各SPMD有独立工作区/信号。
测试两侧均使用已有失败轮询100tick退避hook，未据此改变正式默认。

| 实现 | 五窗 | 中位数 |
| --- | --- | ---: |
| 原动态分task | 852.68 / 847.48 / 863.38 / 861.48 / 881.46 | 861.48 |
| 593旧SPMD | 913.06 / 917.82 / 901.38 / 902.34 / 917.62 | 913.06 |
| compact优先 | 888.22 / 890.76 / 889.94 / 885.92 / 888.38 | 888.38 |

相对同轮旧SPMD降低2.703%（24.68μs）；相对原task仍慢3.123%，30%目标
未达到。不与其它任务物理plan或L0候选的数字相加计算组合收益。
这段数字只对应长B16；完整七档已补齐，见[SEVEN_CASES.md](SEVEN_CASES.md)。
七档仅验收各自当前S6 fixture，对Native精度/整网token、整层mixedQ/空rank或
连续状态轨迹没有新结论。

## 短历史代表通过，迁回正式动态源

8K/B32另一策略任务`task_20261018_014358_34138622015`已完成exit0，
全部数值、图、保护及代际检查通过。同进程五窗完整L1：

| 实现 | 五窗μs | 中位数μs |
| --- | --- | ---: |
| 原动态分task | 998.48 / 1041.28 / 1010.42 / 985.76 / 983.26 | 998.48 |
| 593旧SPMD | 1024.52 / 1024.26 / 1027.80 / 1010.24 / 1029.14 | 1024.52 |
| compact优先 | 1001.86 / 999.22 / 1005.68 / 989.46 / 1001.56 | 1001.56 |

相对同轮旧SPMD降低2.241%（22.96μs），相对原task仍慢0.308%，未达30%。
两个策略代表均有改善后，主会话只将cube/vector_schedule两份文件搬入正式
动态包；33份Python源码与已验证冻结候选逐字一致，见formal_migration.json。
没有叠加物理plan、本地score或L0候选，没有新增静态shape/测试点白名单。

同源码其余128K B4/B8/B24/B28及8K B16已各完成一次exit0，当前fixture
完整精度、图、保护区与代际协议均通过；已完成长B16/短B32未重测。
[SEVEN_CASES.json](SEVEN_CASES.json)保留七档三方共105窗引用、全部原始五样本
及逐项验收状态。七档相对同轮593降低1.73%–4.98%，但仍比原task慢
0.31%–6.09%；相对原task至少降低30%的门槛全部未达。Native/token不在验收范围。

## 复现与本地大产物

从主工作区调用`dynamic_full_pipeline_20261017/run_device.sh`，不能从私有
worktree复制的包装脚本推导环境根。设备须经`task-submit --device auto`：
`--history 131072 --batch 16 --spmd-source formal_migration.json中的tested_snapshot
--source-matrix-reference 主parallel目录/baseline_source --wait-backoff --l1-windows 5`。
短档替换history/batch为8192/32。代表任务记录见device_job.json及
`device_medium_job.json`；补齐五档的精确命令、脚本和stdout/stderr见
[remaining_five/compact_priority_remaining_jobs.json](remaining_five/compact_priority_remaining_jobs.json)
及同目录文本日志。所有原始日志保留，没有hash校验。

CPU大编译目录、原始设备构建和二进制本地保留，只登记路径/大小清单；
七档build_output、编译binary和大CPP/PTO见
[LOCAL_ARTIFACTS_REMAINING.json](LOCAL_ARTIFACTS_REMAINING.json)。
本次归档只复制文本和结果摘要，未复制二进制、未运行Git操作、未重测。
