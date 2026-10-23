# HCA Q_B 大T分档：本轮收口，候选未采用

2026-10-23。用户要求本轮优化后停止，goal已暂停。本文仅汇总已经完成的测试，
没有继续补测、没有修改正式算子、没有push，也没有修改公共SDK。

## 分档依据与实现

允许参考CSA增加区间策略，但不能只复制它的档位标签。原HCA Q_B使用M128/N256，
在129≤T≤192时需要两轮M，对32MiB INT8权重遍历两次。
候选改为M192/N128，一轮M覆盖该区间，权重遍历一次；小T和T>192仍调用原函数。
有效行由动态描述符与valid_shape控制，不假设T=B×6，不修改数学或归约顺序。
这里减少的是GM→L1逻辑加载量，没有PMU证据证明HBM流量也恰好减半。

四个head组、每组4个Cube、K512/K128分层双缓冲不变。
INT32累加不溢出，反量化、RMS和RoPE直接复用原函数。
估计缓冲：L1 A/B双缓冲320KiB，L0A双槽48KiB、L0B双槽32KiB，Acc96KiB。
CPU完整编排及53份incore C++编译通过，生成代码确有M192/N128分支。

完整候选位于`candidates/q_projection_tall.py`，冻结包位于`experiments/q_tall_v1`。
`prepare.py --q-projection-variant`把候选和QKV包装放到冻结TND目录内，
不会覆盖公共HCA的Q_B模块，因此同进程控制组继续使用原公共代码。
控制组为`experiments/native_output_ranges_v1/operator`，两侧使用同一公共SDK
`csa-shared-20261022-logical-codegen`，正式第3层W8A8、NZ2。

## 完整L1结果

同进程交错采样，每侧7窗；单位μs。性能提升按用户定义`control/candidate−1`。

| 场景 | 原策略 | 区间候选 | 性能提升 | 候选更快的配对 |
|---|---:|---:|---:|---:|
| 128K/B28/S6，T168 | 777.96 | 771.64 | +0.82% | 4/7 |
| 128K/B16/S6，T96，原计算分支 | 522.36 | 532.20 | −1.85% | 1/7 |

两档输出、cache/state与控制组逐bit相同，Native最终输出在原0.01/0.01容差内通过，
保护区和重复直接调用通过。Native严格storage 1e−5原差异仍存在，报告`passed=false`，
不把候选精确保持控制结果写成Native全面精度通过，也不宣称30%目标达成。

## 为什么目前不采用

B28的Q_B单核平均耗时的七窗中位数从76.22降至60.85μs；Q_B任务结束位置
从250.20提前至225.42μs。但Q反量化/RMS/RoPE结束仅从313.10提前至307.68μs，
Attention AIC结束从553.68提前至546.24μs，完整层最终只净省6.32μs。
这说明局部供数收益没有等量传递到完整关键路径，不能直接拿Q_B局部百分比报告整层收益。
各项都是独立中位数，不可逐项相加构造一条虚假的真实时间轨迹。

B16仍执行原M128/N256，但新入口包含额外的运行时策略分支，编译后的编排并非逐指令相同。
其Q_B起点从104.10后移至111.50μs、Attention起点从193.82后移至202.60μs；
仅凭这轮不能把原因确定为某一条AICPU指令或JIT开销。
目前仅确认整层有回退，候选不进入正式算子。后续若恢复工作，再隔离入口编排与核内策略。

## 本轮纠正的假设

共享PyPTO的`pl.spmd`默认`sync_start=False`，在
`python/pypto/language/parser/ast_parser.py:4567`和`language/scope.py:232`均可核实。
此前认为HCA大多仍有默认整组起跑等待的判断不成立；没有为显式补写False浪费一轮设备测试。
现有attention三槽与两步QK提前执行也已经存在，不当作本轮新增优化。

## 复现与证据

```bash
python tests/pypto_test/hca_tnd_mpmd_20261022/prepare.py <新实验名> \
  --q-projection-variant tests/pypto_test/hca_tnd_mpmd_20261022/candidates/q_projection_tall.py
```

完整CPU和真机参数参见`Q_PROJECTION_TASKS.json`及`evidence/q_tall_cpu_v1.log`。
所有设备任务必须通过`task-submit --device auto`，不能裸跑；本轮两个任务均退出0。
`Q_PROJECTION_SUMMARY.json`保留原始7窗、逐核阶段统计及精度边界；
`evidence/q_tall_*_result.json`保留完整原始结果，原始泳道和构建产物仅留本地，
路径/数量/大小见`Q_PROJECTION_ARTIFACTS.json`，未做hash校验。
`summarize_q_projection.py`只分析历史文件，不启动设备或重新编译。
