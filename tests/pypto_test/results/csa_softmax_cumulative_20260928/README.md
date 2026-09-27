# 累计softmax最大值候选：保留性能版N128和流水

动机：真实EP16下PTO改变后续专家工作量，两类GMM增量抵消部分CSA收益。
关闭atomic的长档干预仍慢于Native且DSpark不一致；继续检查算法舍入差异，不能把数值差异只算作CSA内部误差。

基底2a740c1f，独立工作树`.cache/csa-softmax-cumulative-2a740c1f`，只改Sparse Attention。
[候选补丁](candidate.patch)做以下相关算术调整：

- 概率转BF16前，最大值由本N128块的局部最大值改为本query累计最大值，初态包含sink。
- 概率BF16使用Native的`round`边界；最终输出及其他量化不改。
- softmax最大值与PV归并分别持有状态，下一query的softmax不会等待上一query的PV排空。
  query首块重置softmax状态，空块保留状态，继续复用既有三槽及预发机制。

与pypto-lib性能版的差别：上游使用局部最大值后归并以支持分块独立处理。
本候选为了检验整网MoE代价，在同一worker顺序处理的softmax块之间保存累计最大值；不新增跨worker依赖。
**仍使用N128，Native以N512归约并量化；不是完整复刻Native，也不保证更高精度或更快。**

CPU lowering/PTOAS/CCE编译通过；[日志](compile.log)。
单卡task_20260928_011145_268616913908退出0，[命令](run_sparse.sh)。
复用现有Native Q/cache/Top-K输入（B40/H8192正式层诊断），原性能版与候选同输入比较：

| Sparse输出指标 | 原性能版 | 累计最大值候选 |
| --- | ---: | ---: |
| Native零容差不一致元素 | 3328649/7864320 | 1554385/7864320 |
| RMSE | 6.3484211e-5 | 4.0564183e-5 |
| max_abs | 0.0009765625 | 0.0009765625 |

RMSE降低36.10%，仍未逐bit一致，不定义为完整数值验收通过；它只是支持进一步测试的证据。
B3/H255均匀attention解析值589824个元素完全相同，覆盖query尾部和无效块，不代表全部边界。
[原性能版报告](baseline/report.json)、[候选报告](cumulative/report.json)、[尾块](tail_b3/report.json)。

task_20260928_011550_276473021789退出0，正式layer4、B16/H8192的完整CSA、metadata/保护区和图重放检查完成。
同轮原性能版→候选，atomic1/det0/mode2，5次预热20次计时：[命令](run_layer.sh)。

| 整层指标 | 原性能版 | 候选 |
| --- | ---: | ---: |
| Native均值 μs | 925.906 | 934.391 |
| PTO均值 μs | 786.753 | 803.294 |
| PTO P95 μs | 806.660 | 817.880 |
| 完整HC输出对Native RMSE | 0.0033201673 | 0.0033204923 |

PTO均值增加2.10%，Native控制也波动0.92%；没有观察到性能收益，整层误差也基本没变。
metadata/保护区/有限值/Top-K结构通过，默认atomic下零容差差异仍保留。
因此**不合入、不做16卡扩测**：局部Sparse误差下降不足以支撑本候选的整模型代价。
下一步单独定位HC pre/post与归一化，避免重复根据局部误差试整网。
[原性能版整层](layer_b16/baseline/report.json)、[候选整层](layer_b16/cumulative/report.json)。
