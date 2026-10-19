# O_B 有效行裁剪：已采用并完成收尾

NZ 大 M128 路径原来按 ceil(T/128) 计算完整行块：B24/S6 的 T144 按 256 行
计算，B28/S6 的 T168 也按 256 行计算。上游量化会把尾行补零，但 O_B 仍然
执行整数矩阵乘并写出零 partial。HC post 的唯一 partial 消费者按真实 T
逐行读取，因此可以裁掉实际 T 之后的算术 padding。

已合入正式 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`：
activation slice 显式携带 `min(128,T-t0)` 有效行，INT8 matmul 与最终 assemble
沿用这一动态范围。NZ 权重 K/N、权重驻留、整数点积、八组反量化左折叠、
BF16 边界、HC post、分核与通知等待均保持。T 内的 dummy/inactive 行照常计算。

工作区容量及上游 INT8 补零合同保留。ND 权重路径和 ROW_TILE≤96 的小中路径
未改变。大 M128 路径的 `[T,P_MM)` partial 尾区由原来的写零变为保持原值，
消费者不得读取该区域；没有放宽输出、cache/state 或保护区门槛。

## 真机结果

四个任务均 `completed (exit=0)`。每个报告包含 973 项 PASS；输出与 score
比较使用 atol=rtol=0，浮点精确对照的 max_abs 全为 0。Top-K、cache/state、
保护区、同址 metadata/padding 变体、图重放和信号代际均通过。

下表为同轮完整 CSA L1 incore 首尾的五次采样中位数，单位 μs；不是 O_B
子阶段耗时，也不与其它日期的绝对时间混算。耗时降低比例以优化前为分母。

| 场景 | T / 最后一块有效行 | 原单 SPMD | 裁剪后 | 耗时降低 | 逐次更快 |
| --- | --- | ---: | ---: | ---: | ---: |
| 128K/B24 | 144 / 16 | 1070.32 | 1065.94 | 0.409% | 4/5 |
| 128K/B28 | 168 / 40 | 1230.20 | 1208.10 | 1.796% | 5/5 |
| 8K/B32 | 192 / 64 | 935.14 | 929.54 | 0.599% | 3/5 |
| 128K/B22 | 132 / 4 | 1056.04 | 1057.66 | -0.153% | 2/5 |

采用理由：三个受影响目标档位同轮中位数均有改善；B22 是 T132、4 行尾块的
边界验证，慢 1.62 μs（0.153%），记录这一结果，不将其当作收益，也不据单轮
微小差异新增孤立 B22 分支。保留简单、覆盖输入区间的动态有效行公式。
七档中其余四档的既有策略选 ROW_TILE≤96；本改动不触及它们的数学分支。
本轮没有重跑不受影响的档位，也没有新增其他优化方向。

## 编译、迁移和证据

- `source_reference/`：测试前的正式源码冻结，37 个 Python 文件。
- `source_candidate/`：实际真机测试的冻结候选，仅改 O_B 文件。
- `ACTIVE_ROWS_AUDIT.json` / `ACTIVE_ROWS_IR.pto.txt`：CPU lowering 确认同一个
  active_rows SSA 传播到 activation、Acc 和写回；该文件只表述 CPU 证据。
- `ALL_RESULTS.json`：四个场景的终态、逐次采样、两种比例定义与采用决定。
- `DEVICE_SUBMISSIONS.json`：四个队列任务的命令、身份和终态。
- `ADOPTION_AUDIT.json`：正式目录 37 个文件去除文档字符串后，逐文件 AST
  与实际测试候选完全一致；仅补充概览和同步注释。数学变化仅一个文件，
  ND、小中 NZ 路径与 policy 保持原样，因此迁移后不重复真机测试。
- 各 `device_*/result.json` 及 `source_matrix/l1_*/*.json`：完整对照与原始 L1
  时间戳。保留可复查证据，不以新增测试替代源码等价核对。
- `ARCHIVE_FILES.json`：入 Git 的文本记录与本地大产物路径/大小；没有文件 hash。

本次按用户的“O_B 修改按照测试效果，按需合入”目标收尾。采用后停止调测；
不据此宣称此前七档相较多 task 降低 30% 的整体目标已经完成。
