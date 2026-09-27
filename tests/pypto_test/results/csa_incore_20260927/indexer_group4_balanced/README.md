# Indexer 4＋2 query复用与核内工作量配平候选

状态：设备验证结束，配平版及进一步L0驻留版均退化，已撤回四query候选。
基于V10性能版，只有decode_indexer.py变化，Sparse Attention维持V10。

目的：同一个768行Key块、同一个L0B的128列panel服务四个query；S6的尾组只处理两个query。
QK仍为M128、FIXPIPE FP16量化及第二次Cube head规约不改；选择和尾部掩码按各query独立计算。
四query分组只在压缩历史超过8192候选且query总数至少48时启用，其他情况仍选择双query策略。
两个分支共用当前函数的constexpr特化，不由测试脚本选择历史算子版本。
通用函数也有重构，阶段出口仍须验证整套当前源码，不能直接把旧双query记录当成新分支实测。

## 为何先配平再上卡

四query组的QK计算约为双query组两倍。直接沿用等item轮转会给部分核分配过多重组。
以下是从现有循环边界推导的单核Score-step×query-pair计数，**不是时间或设备性能结果**：

| 128K档位 | V10双query最忙核 | 4＋2直接轮转最忙核 | 4＋2配平最忙核 |
| --- | ---: | ---: | ---: |
| B8 | 45 | 66 | 46 |
| B16 | 90 | 101 | 92 |

因此取消尚未启动的直接轮转任务task_20260927_143001_121697315147，未占卡测量该候选。
配平规则：先分完整leaf的四query重组，再按剩余query-pair配额分配双query轻组，最后单列短尾leaf。
这是固定24个incore worker内部的工作划分，不改变PTO任务派发、融合或跨任务依赖。

## 编译与测量

CPU编译使用已有fixture及`csa_split_optimization_20260927/compile_contiguous.py`。
已在算子侧避开动态有效行slice的PTOAS类型校验问题、调度C++的And表达式限制和条件变量作用域问题，
未修改PyPTO、PTOAS或ISA，也没有让这些可在CPU定位的问题占卡。

设备任务：`task_20260927_143729_131008131569`，128K/B16代表档。
正式layer4权重＋合成输入历史，单卡S6/TP1/mode2/atomic1/确定性0，无EPLB，复用第二层metadata。
5次预热/20次无profiler计时，另采4个DFX图重放窗口。
原始结果目录：`../../csa_split_optimization_20260927/indexer_group4_balanced/h131072_b16/`。

## 设备结果与撤回

单位μs；两项任务均退出0。新增L0驻留版任务为task_20260927_144233_134499614331。

| 实现 | Score AIC四窗口均值范围 | CSA本体均值 | 本体p50 / p95 | 同轮Native本体 |
| --- | ---: | ---: | ---: | ---: |
| V10双query | 356.79–367.98 | 1304.87 | 1222.98 / 1577.50 | 1317.00 |
| 四query配平 | 441.99–444.93 | 1411.08 | 1315.38 / 1737.88 | 1320.19 |
| 四query配平＋L0驻留 | 426.17–432.73 | 1348.57 | 1290.84 / 1698.40 | 1314.50 |

两项候选保护区/索引结构通过，非有限值0，Top-K集合替换均670，输出max_abs均0.0390625。
RMSE分别0.0041767871、0.0041769822；零容差仍FAIL，不据统计量相近认定逐bit一致。

配平解决了静态工作量偏斜，但生成代码中每个panel重新准备L0A Query/系数。
驻留版把它们移到整个leaf的Score循环之外，使用L0A子视图复用，追回部分时间，仍未优于V10。
两版均撤回，不能仅凭Key逻辑读取量减少就保留；更完整的流水或Vector开销分解仍有待分析。
未扩展其他六档或16卡，没有开展跨任务调度优化。

[配平版计时和泳道统计](report.json)；[候选补丁](candidate.patch)；
[L0驻留版统计](../indexer_group4_resident/report.json)；[L0驻留版补丁](../indexer_group4_resident/candidate.patch)。
