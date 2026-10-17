# OA pair：S1 / S5 独立边界审阅

独立验收完成：两档均**功能通过、原PTO因果前缀零容差通过、Native零容差失败**。两任务均已complete且无运行异常，队列均为completed (exit=1)，原因是Native精度字段失败。本目录不修改候选、驱动或依赖，不编译、不提交设备任务。三个口径分别保留，不由退出码或顶层 `passed` 代替。

## 两个边界成立

使用主进程已冻结的 `parallel_20261018/oa_pair_manual_evidence/source_candidate`，两档均为OA128、组流水开启，实际进入 `cube_schedule.py:409` 的 `0<T<=96` 新路径。

| 规格 | 策略与新增覆盖 | 任务 |
| --- | --- | --- |
| H8192 / B16 / S1，T16 | G2/768、OB32；每请求一个live1尾组；OA有效M16；量化2×8行，INT8尾部[16,32)补零 | task_20261018_060343_145773520215 |
| H8192 / B7 / S5，T35 | G2/768、OB96；每请求2+2+1；OA有效M35；量化跨32行任务至3行尾，INT8[35,96)补零 | task_20261018_060343_145777419564 |

两档都使用pair helper的物理M128、实际valid T，不把T实值化。第二档覆盖非8/16对齐尾部与跨32行量化任务。每个输出组均覆盖3pair+2single五owner，g7的single回绕到AIC0/1；量化等待同一并集。依据候选 `output_projection_a_stages.py:270–400`、`phase_sync.py:159–182`、`vector_schedule.py:938–939`。没有将两个等长请求规格当作mixed Q覆盖。

## 三个结果字段分别解释

主驱动是 [native_query_probe.py](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/native_query_probe.py:198>)。

- `functional_passed`（233–234行）表示有效shape/dtype、双方无非有限值、cache/state/metadata保护，以及TopK结构无非法项；不是Native数值相同。`compare_topk` 返回 `MEASURED` 时仍可能有排列或集合差异，functional允许该状态。两轮graph另有硬断言，验收仍显式核对两轮记录完整且全部通过。
- `precision_exact`（235行）要求Native与SPMD的输出、TopK及各cache/state视图全部零容差通过。Native本探针不采性能。顶层 `passed=functional_passed and precision_exact`，因此既有Native差异可使任务exit1。
- `causal_extension.passed` 是 [causal_extension.py:74](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/causal_extension.py:74>) 的独立结果。原分task真实执行S6，前Q行、position、初始KV及共同页表列精确对齐；额外query分别用X和−X。两次均零容差核对真实SQ输出、TopK、score及五份allocation公共字节，第二次还核对前缀不受额外query改变。

**注意现有汇总结构：causal结果未并入顶层passed。** 驱动在因果对照前已写入passed；因此验收必须另外要求 `stage=complete`、无error、两轮graph PASS、functional=true、causal=true，再单列Native结果。不能反向把因果PASS写成Native通过，或把Native导致exit1写成算子崩溃。此轮只记录这个边界，没有改驱动。

## 保护、代际和因果比较的范围

`native_query_probe.py:212–232` 在两个graph replay前恢复相同初始cache/state并把内部workspace填37，检查输出/TopK及所有cache/state视图与eager零差异、buffer地址不变。score不在 `collect_state` 的graph逐项比较中，而由因果扩展单独核对。两次replay使用相同输入和metadata，不是A→B→A输入更新，也不是连续decode状态轨迹。

协议断言为72核phase0代数各自+1，以及全部60个phase×72槽的第1..127字保持0；没有逐项验收所有业务phase第0字的owner集合。内部Prepared allocator分配首尾各512B保护区均须为37。外部 `x_out/idx_topk/idx_topk_scores` 用普通torch分配，没有同类独立外侧红区。

[guard_checks](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/dsv4_csa_single_layer.py:206>) 对五份cache/state allocation的所有非allowed槽字节做保护比较，并精确检查只读position、Q/L、页表、slot/start_pos等metadata未变。内部工作区仅测外侧保护区，没有逐字节验证全部未使用内部padding；有效输出与因果精确比较补充其数学约束。

因果掩码仅排除 `S6.allowed & ~SQ.allowed`，明确断言覆盖SQ全部合法写字节；其余共同、SQ独有及两侧不应写的字节均比较。S6自身也做完整guards。S1可比S6少state页表尾列，但SQ可访问的共同列逐项相同；分配容量必须相同，否则直接拒绝。均由 `causal_extension.py:32–55` 检查。本轮没有非法页、短表、metadata变化或混合请求长度探针。

## 既有 Native 负结果的适用限制

旧 `native_h131072_b16_s5_v5/result.json` 与 `native_h8192_b32_s5_v5/result.json` 均为 `stage=complete, functional=true, causal=true, precision_exact=false, passed=false`；输出max_abs均0.03125，Native TopK结构合法但存在集合差异。精确路径位于主仓 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/`。MASTER的“2026-10-17 23:49/23:55”及 [S5历史说明](</data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/s5_native_validation_v1/README.md:76>) 保留冻结容差仍超差的事实。

历史仅说明Native零差与PTO数学一致性需要分开验收，不能预先豁免本轮差异；本轮若因果失败、保护失败或graph不稳定，均应独立认定为失败。不能凭旧Native差异把新的回归自动归因Native。

## 本轮结果位置

主PAR为 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/`。

- `oa_pair_manual_h8192_b16_s1_v1/result.json` 及 `causal_extension.json`。
- `oa_pair_manual_h8192_b7_s5_v1/result.json` 及 `causal_extension.json`。
- 任务/命令登记：`oa_pair_manual_evidence/boundary_registry.json`。

## 本轮独立验收结果

逐项读取原始result和causal_extension；两档独立causal JSON与result内嵌对象完全相同。每档记录的33份候选Python源码逐文件直接文本比较，均与冻结source_candidate一致。

| 检查 | S1：H8192 / B16 / T16 | S5：H8192 / B7 / T35 |
| --- | --- | --- |
| stage / error | complete / 无 | complete / 无 |
| functional_passed | true | true |
| causal_extension.passed | true | true |
| precision_exact / 顶层passed | false / false | false / false |
| graph | 2轮，每轮8项零差、31项guards、70份内部保护区全PASS | 同左 |
| graph协议 / buffer地址 | 2轮均PASS / 不变 | 2轮均PASS / 不变 |
| 因果扩展 +X / −X | 各8项零差、31项guards全PASS；−X另8项前缀不变检查PASS | 同左 |
| Native/eager guards | 全PASS | 全PASS |
| 非有限值 / 非法TopK行 | 0 / 0 | 0 / 0 |
| performance_measured | false | false |

因果扩展的五份allocation均确认覆盖实际SQ的全部合法写字节，共同页表前缀相同。S1的state与indexer_state页表为[16,4098]，真实S6为[16,4100]；S5两侧均为[7,4100]。S1的compressed/indexer合法写字节为0，只覆盖保持与保护；S5分别有7168/910个合法写字节，补充了压缩cache实际写入边界。没有把S1的零写入结果当作该分支完整覆盖。

Native比较均使用atol=rtol=0，浮点项是逐元素零容差比较，不额外宣称signed-zero等原始位模式比较。主要数值如下；“—”为整数项不计算max_abs。

| Native比较项 | S1差异元素数 | S1 max_abs | S5差异元素数 | S5 max_abs |
| --- | ---: | ---: | ---: | ---: |
| x_out | 102636 / 262144 | 0.03125 | 235583 / 573440 | 0.0234375 |
| idx_topk | 7008 / 8192 | — | 15488 / 17920 | — |
| swa.0 | 2403 | 0.0078125 | 5277 | 0.0078125 |
| compressed.0 | 0 | 0 | 41 | 0.00390625 |
| state.0 | 32720 | 0.0006489753723144531 | 71585 | 0.0007456839084625244 |
| indexer.0 | 0 | — | 59 | — |
| indexer.1 | 0 | 0 | 6 | 0.00006103515625 |
| indexer_state.0 | 8182 | 0.0006167888641357422 | 17900 | 0.0007491409778594971 |

输出RMSE分别为0.0030309620525652958、0.0031819391062775423。Native TopK集合分别在16/16、35/35行不同，replaced_indices分别50、124；order_only_rows均0，invalid_rows均0。结构合法不等于选择结果一致。

### Native差异的归因限制

已检索主仓`tests/pypto_test/csa_multishape_spmd_20261016/`及本私有worktree的测试证据，共读取890份result类JSON；另检索MASTER及主动态测试树的规格/路径记录。精确H/B/Q为8192/16/1、8192/7/5的结果仅找到本轮两份，**未找到同规格正式源旧结果，不能把全部Native差异称为既有差异，也不能量化相对旧正式源的Native精度变化；该归因未完成。** 前述旧S5负例规格不同，只作背景。

本轮能直接确认的是：候选与原分task真实S6的SQ因果前缀，在两个额外query符号下，对已检查的输出、TopK、score及SQ合法cache/state写入完全一致。因此这些输入下的Native差异也出现在该原PTO因果前缀参照中；这不替代同规格“旧正式单SPMD vs 候选”对照，不能宣称Native通过或精度差异完全定位。

验收结论限定为上述两个OA启用边界的功能、保护和原PTO因果一致性通过。未扩大到mixed Q、非法表/页、长历史、连续decode轨迹或这两档性能。

## 可归档文件

本目录可整体复制。两子目录各保存原始`result.json`、`causal_extension.json`以及只读查询所得`task_status.txt`、`task_log.txt`；根目录保存任务登记副本`boundary_registry.json`及查询记录`QUERY_STATUS.json`。`LOCAL_ARTIFACTS.json`逐文件列出归档文件以及两原输出目录内文本/本地大文件的绝对路径和大小；未计算校验摘要。原始build_output、源码快照和numerical_diagnostic.pt保持在主输出目录，只列路径、不搬移或改写。
