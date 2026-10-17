# 动态 CSA：S5 尾组布局修正及验证记录

日期：2026-10-17。正式代码位于 `ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py`。

## 结论与范围

- 修复了单 SPMD Indexer 尾组的实际计算错误：有效 query 少于分组 tile 的 query 数时，QK 的紧凑 L0C 布局与当前 A3 `TINSERT` 的源行距不匹配。
- G2 的 768/1024 score tile、G6 的 768 score tile，完整组与尾组的第一条 query 分数及 Top-K 根均精确一致；CPU 整数点积参考也精确一致。
- 128K/B16、8K/B32 的 S6 完整 CSA 对原动态分 task PTO，A→B→A 三轮输出、Top-K、score、cache/state、保护区及代际检查通过。此次没有采性能，也没有重跑七档。
- S5 长短历史的第 5 行异常消除，前四行输出和 Top-K 完全未变；eager/graph 自身精确一致、保护区和代际通过。**S5 与 Native 仍存在差异，不能称完整精度验收通过，更没有验证生成 token 一致。**
- 本次不修改 PyPTO、Simpler、PTO-ISA、共享安装、服务路由或旧分 task 数学代码。

Dspark 配置上限保持 5，本轮实际 query 数设为 5；不是改成 Dspark=4，也不把 S5 假装成 S6。性能目标仍是 S6 七档对原动态分 task PTO，每档完整 L1 耗时降低至少 30%。

## 为什么单独使用 Native 正确性参照

旧分 task 的 `NativeCSACall`、`DECODE_SEQ`、compact metadata、两条 compressor、Indexer 分组和 Attention 请求定位均仍有 S6 假设。不能仅改 fixture 的 token 数后让旧算子充当 S5 oracle。

`native_query_probe.py` 复用真实 Native builder、正式 W8A8 第 4 层权重和缓存布局，机械绑定现有 device Tensor 到动态 SPMD。Native 仅作正确性参照，不测 Native 性能。零容差报告用于保留完整差异；原性能版 PTO 本来就不与 Native 逐 bit 一致，不能将零容差失败直接解释为本次所有改动都算错。

`dsv4_csa_single_layer.make_fixture` 新增仅关键字参数 `query_tokens=6`，默认行为不变。`default_s6_source_compatibility.json` 记录代回 6 后的算法 AST 等价核对，另有两档真实 S6 回归。

## 故障定位

最初两档 S5 的结构、保护区及 graph 检查通过，但每请求第五行明显偏离：

| 历史 / B | 原输出 max_abs | 第五行每行 Top-K 平均替换项 | 前四行平均替换项范围 |
| --- | ---: | ---: | ---: |
| 128K/B16 | 0.1875 | 454.1875 | 5.56～6.44 |
| 8K/B32 | 0.1953125 | 331.5625 | 2.84～3.53 |

`score_tail_probe.py` 用正式 Cube/AIV helper 隔离问题：同一第一 query、相同 Native key/scale 页、相同实际可见历史，固定地址执行 `live=2→1→2`。没有重新实现设备数学。CPU 使用 INT32 精确点积、显式 FP16 缩放/ReLU 舍入和 FP16 head 系数，辅助核对第一 query 的 2048 个 score。

修正前 `score_tail_v1/result.json`：G2 两种 tile 下 `live=2` 均为 0 差异；`live=1` 均有 1792/2048 个 score 错误。每个 128 列 panel 只有前 16 列正确，说明是分形间步长问题，不是 Top-K 并列顺序。

生成 C++ 的 QK Acc 类型是物理 `M=128`、有效 `M=64`、`CompactMode::Normal`。`mad` 将结果按有效 M 紧凑存放，16 列分形之间的源步长为 64 行。实际构建依赖：

```text
/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include/pto/npu/a2a3/TInsert.hpp
```

其中 `TInsertAccToMat` 仍使用：

```cpp
constexpr uint16_t srcStride = SrcTileData::Rows;
pto_copy_matrix_cc_to_cbuf(
    dstAddr, srcAddr, 0, nSize, SrcTileData::Rows,
    dstStrideD, srcStride, unitFlagCtrl, QuantPre, ...);
```

源步长及搬运行数仍为物理 Rows，没有按 CompactMode 处理有效行。因而第二个分形起错读。相邻 `TStore.hpp` 已对 `CompactMode::Normal` 计算有效行对齐后的源步长，但不能据此假设 TINSERT 也支持。此结论限定于本次实际依赖版本；未修改或验证上游最新版。

## CSA 内的修正

1. GM 仍只加载 `live×64` 个有效 query head，不读取下一请求或越界地址。
2. 在 L1 中，把不足 G 条的 query 槽用本组第一条有效 query 补齐；通过已有 `gather_row` 实现，未增加 GM workspace。
3. QK 使用完整物理 M，保证 Acc 实际行距等于 TINSERT 所用行距。
4. head 系数生产者已经完整写零到失效 query 槽；归约使用完整物理 K，补齐项乘零。INT8 点积经 1/1024 和 ReLU 后有限，不引入 `0×Inf`。
5. Vector 仍只发布 live 条 query 的 score/Top-K。请求边界、动态循环、页表、核间协议和输出覆盖不变。

完整组不增加补齐搬运，数学顺序不变。尾组多做补齐的片上计算与重复 query 加载，这是当前布局限制下的正确实现，不宣称没有代价。未来底层支持紧凑 Acc→Mat 后可评估去除补齐；需重验尾组数值及完整 CSA 性能。

修正前后源码分别保存在 `score_tail_layout_fix/before_indexer_score_stages.py` 和 `after_indexer_score_stages.py`。

## 修正后的证据

| 用例 | 任务 | 结果 |
| --- | --- | --- |
| G2，tile768/1024，2→1→2 | `task_20261017_233838_195151931247` | 6 轮 score/根/CPU 参考全部零差异 |
| G6，tile768，6→1→2→3→4→5→6 | `task_20261017_234124_2018905507` | 7 轮第一 query 的 score/根/CPU 参考全部零差异 |
| S6，128K/B16 | `task_20261017_233941_197286026429` | 完整 CSA 对原分 task，A→B→A 全部精确通过 |
| S6，8K/B32 | `task_20261017_233942_197336924422` | 同上 |
| S5，128K/B16 | `task_20261017_233926_196680127927` | 结构/两轮 graph/保护区通过，Native 数值仍有差异 |
| S5，8K/B32 | `task_20261017_233925_19662801044` | 同上 |

G6 helper 检查没有验证整层混合请求，也没有覆盖 balanced G6 的所有尾组路径；不扩大结论。

S5 修正后两档全局输出 max_abs 都降至 **0.03125**；第五行平均 Top-K 替换降至 **6.625 / 3.5625**。修正前后 Native 输出/Top-K 完全一致，PTO 的前四行输出/Top-K 也完全一致，变化只出现在原先有布局错误的第五行。剩余差异与前四行量级相当，仍需单独确认来源及验收。

按既有 `atol=0.01, rtol=0.01` 冻结输出门槛，S5 仍有 **5113 / 1606** 个超差元素；cache/state、INT8 和 Top-K 也未完成 Native 精度验收。这个统计不是新放宽的标准，不能把“尾组修复”改写成“整层精度通过”。详见各 `native_*_s5_v4/tail_fix_analysis.json` 和 `result.json`。

## 复现

工作目录为本仓库，所有 NPU 执行必须经队列。输出目录必须不存在。

```bash
case_dir="$PWD/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
task-submit --device auto --max-time 900 \
  "bash '$case_dir/run_score_tail_probe.sh' '$case_dir/score_tail_repro_new'"
task-submit --device auto --max-time 900 \
  "bash '$case_dir/run_score_tail_probe.sh' '$case_dir/score_tail_g6_repro_new' --group-size 6"
task-submit --device auto --max-time 1200 \
  "bash '$case_dir/run_native_query_probe.sh' '$case_dir/native_s5_repro_new' --history 131072 --batch 16 --query-tokens 5"
```

当前正式源码应通过 helper 检查；Native 整层探针仍会因保留零容差差异而返回 1，请读 `functional_passed`、`precision_exact`、逐项比较和 graph 记录，不把退出码等同为崩溃。

失败复现使用 before 源码快照，并在隔离 checkout 中替换对应 helper；不要覆盖另一个会话的共享工作区。任务号、前两轮驱动问题及最终结果统一保存在 `jobs.json`。大 `.pt`、编译产物、完整源码快照和二进制仅本地保存，路径及大小见 `local_artifacts.json`，不做 hash 校验。
