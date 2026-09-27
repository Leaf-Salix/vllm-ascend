# 128K/B8三query复用：独立候选

基于已保留的9a01a276，隔离工作树`.cache/csa-indexer-mid-9a01a276`。
长历史的query数48～95时选择三query/M192/N128，至少96时保留S6/M384/N64，更小输入保留双query。
短历史不变；这是一套算子内的输入策略，不按测试档切换历史版本。
[补丁](candidate.patch)、[单卡命令](run_layer.sh)。

三query此前只在128K/B16实测；B8还没有设备收益证明，且本次arena沿用9a01a276的288行。
B8最忙核Score-step×query：双query为66、三query为75、S6为96；较小分组可能更好地保持并行度。
这些是工作量推导，不是实测耗时。FP16/Cube数值策略、Native物理cache、sync/early-resolve均不变。

完整CSA CPU lowering、PTOAS、CCE和链接通过，Ruff及shell语法通过；没有设备执行。
[编译日志](compile_full.log)。复现：

```bash
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/results/csa_indexer_six_20260928/compile.py --full \
  --source ../.cache/csa-indexer-mid-9a01a276 \
  --output tests/pypto_test/results/csa_indexer_mid_20260928/compiled_full
```

task_20260928_040125_325427313063：仅128K/B8，同轮基线/候选各5预热、20次无profiler完整CSA，
各4个DFX窗口。基线工作树`.cache/csa-indexer-six-2a740c1f`的算子与9a01a276相同。
保留全量样本/P95、metadata/保护区、有限值与Top-K结构检查。当前尚未合入生产。

完成后复用通用收集器：

```bash
python tests/pypto_test/results/csa_indexer_triple_20260928/summarize.py \
  --root tests/pypto_test/results/csa_indexer_mid_20260928 \
  --task-id task_20260928_040125_325427313063 --baseline-revision 9a01a276
```
