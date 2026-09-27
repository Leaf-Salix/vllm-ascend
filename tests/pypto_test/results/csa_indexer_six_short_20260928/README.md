# 短档S6复用：独立候选，尚无设备结果

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
