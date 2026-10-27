# Query前段调度/流水证据

本目录是2026-10-27独立候选的冻结记录，不是已采用的运行源码。运行源码仍为11b380e3f。主结论见上两级的`DSV4_FLASH_CSA_TND_QUERY_FRONT_PIPELINE_20261027.md`。

## 只用公开归档离线复现阶段分析

需要Python3.11及`regex`；不需要vLLM、Torch、NPU。先`python -c 'import regex'`确认。以下命令在本证据目录执行，使用独立临时目录恢复，保留原文件。

```bash
QUERY_EVIDENCE_ROOT="$PWD"
QUERY_REPRO_ROOT=$(mktemp -d)
mkdir -p "$QUERY_REPRO_ROOT/collected/query-island-v14-run1"
mkdir -p "$QUERY_REPRO_ROOT/collected/query-dependency-v1"
mkdir -p "$QUERY_REPRO_ROOT/collected/query-v14-timing-jit"
tar -xzf executed_helpers.tar.gz -C "$QUERY_REPRO_ROOT"
tar -xzf generated_cpu_evidence.tar.gz -C "$QUERY_REPRO_ROOT/collected"
tar -xzf query-island-v14-run1.tar.gz -C "$QUERY_REPRO_ROOT/collected/query-island-v14-run1"
tar -xzf query-dependency-v1-evidence.tar.gz -C "$QUERY_REPRO_ROOT/collected/query-dependency-v1"
tar -xzf timing-jit-maps.tar.gz -C "$QUERY_REPRO_ROOT/collected/query-v14-timing-jit"
cp analyze_query_dependencies.py "$QUERY_REPRO_ROOT/"
python "$QUERY_REPRO_ROOT/analyze_query_dependencies.py"
cmp "$QUERY_REPRO_ROOT/QUERY_DEPENDENCY_ANALYSIS.json" "$QUERY_EVIDENCE_ROOT/QUERY_DEPENDENCY_ANALYSIS.json"
```

`LOCAL_REPRO_CHECK.json`记录实际按上述归档恢复后结果逐字节一致。脚本先验证保留构建/CPU/dep的leaf、严格alpha编排与全部配置，再条件性关联task token，不把L1注册ID当funcID。原计时缺少逐轮registration直接见证，因此细分端点有映射假设，不作精确因果证明；完整event/L1结果不受该函数命名假设影响。

## 硬件复现和口径

`executed_helpers.tar.gz`内为独立helper源码和manifest，不含已采用源码变更。CPU收据和实际生成文本在`generated_cpu_evidence.tar.gz`；完整原始CPU归档SHA另记`RAW_CPU_INPUTS.json`，二进制未发布。硬件复现需要主报告和前一阶段报告指定的SDK、源码快照、真实权重、静态13包。绝对路径变化必须重新binding/manifest并CPU编译，不能重用旧身份声明。

所有NPU任务outer780s/inner750s，先event性能、再独立L1，dep另起任务。没有性能winner就不执行Native精度扩测。源数学相同或exit0均不等同于新Native精度通过，更不等同于新整模型token一致。

原始JSON的部分`criterion`/`test_order`仍沿用最早harness的query-island文字；每次真正被测改动以`candidate_experiment`、`candidate_variant`、`isolated_candidates`及冻结SOURCE/manifest为准。其条件代码没有因此改变。本轮Query融合、Q_B流水、HC pack都是独立候选，不曾组合。

## 准备后取消的诊断

`executed_helpers.tar.gz`保留本轮已运行和CPU准备过的helper，不代表每个helper都已上NPU；逐项状态见`TASKS.json`。其中`qa-parallel-dependency-v1`仅完成静态审核，等待设备时取消，无新设备结果。取消原因是它不能补回原计时的registration缺口，而QA候选已因完整回放不稳定否决；没有发布其未执行的阶段分析。

`qa-parallel-timing-jit-maps.tar.gz`是保留构建证据，不宣称它提供逐轮graph/registration直接见证。A/A的只读实际Worker登记观测已通过，后续新计时须沿用这一观测方式再关联阶段。
