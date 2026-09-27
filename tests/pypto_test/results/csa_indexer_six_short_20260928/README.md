# 短档S6复用：B40核内收益，模型待验

基于已保留的9a01a276，隔离工作树`.cache/csa-indexer-six-short-9a01a276`。
只改变短档策略选择，长档S6与正在执行的EP16源码均不动。
[补丁](candidate.patch)、[单卡命令](run_layer.sh)。

## 按工作量选择

压缩历史2048～8192行、单leaf路径，候选先要求整请求分组足以覆盖24个worker，
再比较最忙核的query工作量：`ceil(query数/分组大小/24) × 分组大小`。
只有S6相对双query增加不超过25%时，才选M384/N64、K384/N64；其他输入仍双query/M128/N128。
当前矩阵的8K/B24、B40会选S6，B16、B32保留双query。此规则是待测启发式，不是最优策略结论。

B40双query最多5组/核、10个query，S6最多2组/核、12个query；算术工作量上限增加20%，
同请求Key从三遍变一遍。仍须靠实测判断减少读取能否抵消计算与流水代价。
没有改变FP16/Cube、FIXPIPE、Top-K规则、cache布局、worker数、短档sync/early-resolve设置。

## 编译与筛查

最初在调度层使用复合`and`遇到`GenerateExprString not implemented for expression type: And`；
改为等价嵌套条件生成INT32选择值后，完整CSA lowering、PTOAS、CCE及链接通过，Ruff/shell语法通过。
未修改或更新工具链。CPU编译在本轮EP16正式采样前完成，不把编译通过当性能或功能通过。
[完整编译日志](compile_full.log)。复现：

```bash
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/results/csa_indexer_six_20260928/compile.py --full \
  --source ../.cache/csa-indexer-six-short-9a01a276 \
  --output tests/pypto_test/results/csa_indexer_six_short_20260928/compiled_full
```

task_20260928_035039_308109516510已排队，先只测8K/B40：
原版9a01a276算子与候选各5预热20次无profiler完整CSA，另各4个DFX窗口，保留所有样本/P95。
基线来源`.cache/csa-indexer-six-2a740c1f`是2a740c1f＋已合入9a01a276的S6补丁，算子相同。
两侧仍检查metadata、保护区、有限值和Top-K结构；没有声称覆盖B24或整模型。
有收益再决定后续受影响项；无收益就停止扩测。当前未合入生产。

## 单卡结果

task_20260928_035039_308109516510退出0。[完整原始样本和四窗口](report.json)。

| 指标μs | 基线9a01a276 | 短档S6候选 |
| --- | ---: | ---: |
| Native控制均值 | 1435.988 | 1569.268 |
| Native P95/max | 1481.560/1482.960 | 1756.320/2275.720 |
| 完整CSA均值 | 1320.763 | 1296.411 |
| PTO P95/max | 1373.780/1379.320 | 1325.420/1328.120 |
| Score AIC四窗口block均值 | 95.915 | 46.691 |
| Score AIV四窗口block均值 | 93.888 | 59.864 |

Score AIC下降51.32%、AIV下降36.24%，窗口范围无重叠，支持保留核内复用。
完整CSA下降1.84%，但Native控制均值上升9.28%且出现长尾，不能将相对Native的比例算作可靠净收益；
全部样本保留，没有剔除异常值。未改Sparse核内略升，不归为这次优化成果或确定退化。
metadata/保护区、有限值、Top-K结构通过；Native零容差仍FAIL，输出RMSE0.003291863→0.003292732，
Top-K集合替换均901，不表示跨版本逐元素一致。
与长档B8策略合并后，先补B24/H8192固定规约状态/图检查，再做真实EP16；B24性能尚未实测。
