# TND完整精度版：HC-pre面板复用优化（2026-10-18）

## 状态与版本

以临时精度交付`c87e278e55cea4becd627f174bd5e3a382e9d895`为黄金，
仅修改HC-pre线性分片生成。原性能版和冻结精度快照不覆盖。
本文件所属提交为本轮源码版本；实测源码SHA另行冻结，不能以移动分支HEAD替代。
分支仍为`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，变体仍为`tnd_full_precision`。

模块、完整CSA单层及真实整模型8K/B40均已通过本轮精度验收。
本轮不跑Native2、128K或BSH；不改SDK、codegen、静态包、权重dtype或精度门禁。

## 从nalinaly性能版借鉴什么

只读更新的参考为[nalinaly a919f1194](https://github.com/nalinaly/vllm-ascend/commit/a919f11942d93627441c8fa694e310dcbd1c671b)。

| 参考做法 | 源码位置 | 本轮判断 |
| --- | --- | --- |
| T16/K256、stage2流水和Cube矩阵乘 | `deepseek_v4_flash_csa/hc_pre.py` | 搬运复用值得参考；其4×4096分片不能直接替代本精度版22分片及HF32每8积舍入 |
| BF16加宽时融合RMS、减少GM重复读取、提前派发 | `deepseek_v4_flash_csa/decode_csa.py::hc_widen_rms` | 融合搬运可借鉴；512列平方和及rsqrt不能替代已验收Native统计/舍入边界 |
| 多query共享Key，N128预取面板、双缓冲、Key/Score独立L1存储与AIC/AIV事件 | `deepseek_v4_flash_csa/decode_indexer.py` | 后续优化方向；须保留真TND边界及Native score、量化、TopK语义，不能按固定S6照搬 |
| 单根、scratch作用域、减少调度等待 | 同包`decode_csa.py` | 当前已保留；本轮不改runtime或作用域生命周期 |
| 独立HC-pre的分档与48核分工 | `vllm_ascend/ops/pypto/hc_pre.py` | 是另一入口，其性能数据不能当成CSA内HC-pre或本版逐bit数据 |

本轮采用其“独立行合并处理、共享搬运”的思路，独立实现，不移植其数值计算路径。

## 实际改动及精度边界

- `hc_pre.py`：按**真实token数T**分流，T<16保留原标量行路径；T≥16使用8-token面板。
- 新增`hc_pre_panel.py`：FP32输入先做布局转置；每份权重同时供8个独立token使用，
  连续载入8个token，使用合法8×1 ColMajor载体进行广播。
- 不改变22分片、768/末段256的K范围、每8乘积与旧累加值的精确和及RNE，
  或最终22分片顺序。185条精确和/舍入语句与原路径逐AST一致。
- 两个路径均初始化padding partials；新面板显式清零内部转置缓冲的尾行。
  这是kernel内部tile对齐，**不把真实请求补到S6**，也不改变外部query bounds。
- HC-pre后续统计、gates、Sinkhorn、输入Norm及其他CSA模块不变；
  仍为HC-pre到HC-post完整纯DSL替换、单一顶层PTO入口、62tensor ABI。
- 原冻结实现仍在c87提交及原运行快照中。本轮没有重新安装或编译vLLM/PyPTO/Simpler；
  仅对修改的DSL生成JIT产物。

T16是保守启用阈值，没有宣称已搜索所有可能的分界点。
不保留全T统一启用方案：原始panel在T1和T3分别回退约131%和42%。

## 单层完整CSA：精度和性能

真实L4模型权重，合成hidden/history，历史8191；单卡TP1，确定性level1及HCCL开启，
NZ2、13个静态包及实际reselect、SK关闭。`npugraph_ex fullgraph=True`实际捕图并raw replay，
不是eager。各图固定query边界，输出NaN/TopK毒化恢复、A/B/A及保护区检查通过。

单位ms/完整HC-pre至HC-post图回放，8轮×10次，正反执行顺序交错，三臂同进程同卡。
所有边界A/B均对Native逐bit检查后才计时，计时后再次核验状态和保护区。

| 配置 | Native均值 | 冻结精度版均值 | 候选均值 | 候选延迟下降 | 精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| 混合1～6行，B6/T21，原始panel实验 | 0.5533 | 6.1123 | 4.7701 | 21.96% | A/B逐bit一致 |
| 混合1～5行，B5/T15，正式分流候选 | 0.5103 | 4.4752 | 4.4359 | 0.88%，按无退处理 | A/B逐bit一致 |
| B40/S6/T240，正式分流候选 | 1.3803 | 55.1257 | 28.1117 | **49.00%** | A/B逐bit一致 |

上述每组候选最慢样本仍小于冻结版最快样本，完整原始样本保留。
T15微小差别不推广为普遍收益。T21是形成分档策略前的独立panel根，不能冒称正式候选另一次复测。

逐bit范围为最终输出、TopK及六类cache/state，共八类外部边界；
独立CPU审查实际读取三臂A/B见证，逐uint8字节复算。
Native/候选初始hidden及cache相同，禁止PTO fallback；保存独立注册schema、62参和实际调用证据。
这不等于所有内部临时tensor或任意输入均有形式化等价证明。

## HC-pre模块单因素证据

单位ms/raw graph，8轮×10次。真实L4 HC权重、合成hidden；
完整partial含padding对冻结实现、mixed/post/comb对Native，A/B/A/zero及guard全部通过。
模块Native为直接HC-pre raw graph；不把它伪称为整模型静态FULL服务口径。

| T | 冻结HC-pre | 原始8-token候选HC-pre | 延迟下降 |
| --- | ---: | ---: | ---: |
| 1 | 0.7974 | 1.8455 | 回退131.44%，不启用 |
| 3 | 1.3007 | 1.8471 | 回退42.01%，不启用 |
| 16 | 3.6509 | 1.8535 | 49.23% |
| 21 | 4.9290 | 3.4995 | 29.00% |
| 40 | 8.8198 | 5.1414 | 41.71% |
| 240 | 50.2854 | 23.2729 | 53.72% |

HC-pre模块成本与完整CSA的量级，说明精确乘加是当前重要开销；
两套不同fixture/任务的时间不能直接相减当成严格的关键路径分解。
候选B40仍约为本组Native的20.4倍耗时，不能宣称已达到nalinaly性能版速度。

## 真实整模型8K/B40

任务已正常终态，实际执行8分38秒。仅执行1次192-token预热加1次192-token测量，
复用原已验收Native黄金，不重新生成bank或Native2。
每rank逐请求检查完整token、完整有序接受事件和零抢占；
21层CSA完整替换、20层HCA及FFN仍Native；TP1、DP=EP16、EPLB关闭。
本轮只有一轮整模型测量，与旧三轮数据只能作历史参照，不能伪称同会话受控性能实验。

| 真实整模型8K/B40指标 | 冻结精度版历史值 | 本轮优化版 | 变化 |
| --- | ---: | ---: | --- |
| target forward，ms | 1218.433 | 651.470 | 延迟下降46.53% |
| 解码周期，ms | 1234.349 | 667.174 | 延迟下降45.95% |
| 对冻结Native黄金的token/接受事件 | 原已通过 | 预热及测量全部一致 | 零抢占 |

Native历史target forward为89.482ms，本轮候选仍约7.28倍耗时。
这是完整模型forward设备事件，不是单层CSA，也不是服务吞吐。
本轮20个对齐step逐step取16rank最大，forward范围650.211～652.508ms；
周期19个样本666.252～668.103ms，不能把单轮窄分布当作跨进程置信区间。

独立CPU审计重读16rank的预热与测量：每轮122880 token，两轮共245760 token、
完整有序接受事件和计数器与原Native黄金相同；21层CSA及6/240两档单根warmup/capture、
13包实际reselect一次/验证复用、计时期间无新增编译/缓存错误、CPU入场几何均通过。
16rank是相同40道问题的副本，不是640道不同问题。
本轮没有新增整模型逐层tensor hook，不将token一致扩大为已逐bit观测所有内部tensor。
历史性能和协议见[原单入口精度版性能报告](DSV4_FLASH_CSA_TND_SINGLE_ENTRY_DSL_REAL_MODEL_PERFORMANCE_20261018.md)。

## 限制与复现原则

本版仍要求原已验收A3/CANN9.2私有SDK、静态资格bootstrap、dtype/layout条件及T≤256。
新转置缓冲上限16MiB，在HC-pre阶段作用域内使用；本轮未验收新的128K配置或其他SOC/SDK/SK1。
旧c87的128K验收不自动改标为新优化版验收。

复现使用本文件所在提交及证据中的精确SHA；先还原临时精度交付报告指定的SDK、
模型和bank，逐文件验证冻结身份及本轮仅两文件数学差异。
先模块，再混合短query，再B40完整CSA；发现首差即停。整模型须先通过这些门禁。
每个NPU任务单独提交`--device auto --max-time 780 --ptoas 0.63`，显式传入原
`PYTHONNOUSERSITE=1`、CANN、PTOAS和CCACHE环境；每5分钟内检查进度，不创建自动任务。
精确命令、助手、原始JSON/log及SHA已列于下节及本轮证据索引。

## 复现入口、证据与任务

[证据索引](evidence_tnd_precision/hcpre_panel_performance_20261018/index.json)、
[源码两文件SHA](evidence_tnd_precision/hcpre_panel_performance_20261018/source-manifest.json)、
[全部结构化结果](evidence_tnd_precision/hcpre_panel_performance_20261018/summary.json)、
[任务终态](evidence_tnd_precision/hcpre_panel_performance_20261018/tasks.json)。

[单层小归档](evidence_tnd_precision/hcpre_panel_performance_20261018/single-layer-evidence.tar.gz)
包含各次原始助手/manifest、JSON、日志及无设备失败记录；
大体积三臂tensor见证保留本地和227，逐文件SHA/字节数列在单层索引中，不放进Git。
[整模型归档](evidence_tnd_precision/hcpre_panel_performance_20261018/whole-model-evidence.tar.gz)
包含16rank完整输出/事件/计时/日志、Native黄金文本、执行助手、候选patch和全源manifest；
[独立审计结果](evidence_tnd_precision/hcpre_panel_performance_20261018/whole-model-qualified.json)
可从原始JSON独立复算，归档不含模型、KV bank或SDK二进制。

原227实验根为：
`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/precision-perf-20261018`。
单层助手`formal-v1`对应其`staged-v1`，`formal-v2`对应`staged-v2`；
模块和raw-panel助手对应实验根。每次换新输出标签，不覆盖已验收结果：

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
J=$BASE/precision-perf-20261018
submit_one() {
  task-submit --device auto --device-num 1 --max-time 780 --ptoas 0.63 \
    --env PYTHONNOUSERSITE=1 \
    --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
    --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
    --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
    --env CMAKE_BUILD_PARALLEL_LEVEL=4 "$1"
}
# 各项完成并检查首差后，再提交下一项；NEW替换为尚未使用的标签。
submit_one "bash $J/run-hc.sh hc-short-NEW '1 3 21'"
# 模块大T：bash $J/run-hc.sh hc-large-NEW '16 40 240'
# T15完整CSA：bash $J/staged-v1/run-csa.sh short-NEW 1,2,3,4,5 4 8191
# B40完整CSA：用40个6组成lengths，而不是在脚本中补请求。
lengths=$(python3 -c "print(','.join(['6']*40))")
# submit_one "bash $J/staged-v2/run-csa.sh b40-NEW $lengths 4 8191"
```

执行前先按原[冻结精度交付报告](DSV4_FLASH_CSA_TND_TEMPORARY_PRECISION_DELIVERY_20261018.md)
核验私有SDK、原全源和静态包；解压单层归档后，核对选中助手manifest。
目录名`module-short/module-large`只区分当时预检快照，实际模块运算代码相同；
整套`raw-panel`助手也包含这两项模块执行链。保留原预检，不因助手文件缺失跳过SHA检查。

整模型助手在整模型归档`helpers/`中，先运行`preflight.sh 8k`，通过后执行
`run-case.sh 8k pto`，设备数改为16、其余提交环境和max780秒相同。
**原run-case使用固定本轮输出目录，不可直接重跑覆盖**：未来复测复制执行助手、
`patch/`与`reference/`到新实验目录，修改env/preflight/run-case中的job、SOURCE_ROOT及输出路径；
保留bank和Native黄金路径、参数、精度门禁和候选patch内容。
原`helpers/snapshot/manifest.json`作为原始见证单独保存，不预放入新运行snapshot目录；
新目录执行`prepare_snapshot.py`重新复制原冻结树，并由`verify_assets.py`确认恰好两文件变化。
按助手中现有SHA256方式重新生成新目录的`helper-manifest.json`，先做无设备预检再提交。
不要把旧qualified-result或其他生成结果当作执行助手，也不要运行历史报告生成器。
迁移SDK/模型/主机须先还原对应资产并重新验收，不能只改路径声称同环境。

| 任务 | ID | 卡数 | CLI记录实际执行 |
| --- | --- | ---: | --- |
| 模块T1/3/21 | `task_20261018_231147_53132715565` | 1 | 1分28秒 |
| 模块T16/40/240 | `task_20261018_231502_65379123384` | 1 | 1分23秒 |
| 原始panel混合T21 | `task_20261018_231801_76454827264` | 1 | 3分26秒 |
| 正式分流T15 | `task_20261018_232734_111955621503` | 1 | 3分30秒 |
| 正式分流B40/T240 | `task_20261018_233308_134813512086` | 1 | 3分39秒 |
| 真实模型8K/B40 | `task_20261018_234207_169125428019` | 16 | 8分38秒 |

六项均exit0、单项max780秒，全部终态，无本轮遗留任务或自动调度。

## 审查与本地门禁

源码、测试助手、原始tensor见证和16rank原始token/events均独立复核通过。
真实模型独立复核还重算52576条接受事件、max-rank forward/cycle及全部静态资格。
保留无设备预检中因嵌套目录误解析bank路径而失败的日志；修正路径后重做完整预检，
没有改变gold或减少检查。11项CPU计时/资格测试通过。
生产两文件及报告/索引的增量pre-commit通过，报告的manual Markdown检查通过；日志见证据索引。
隔离全仓`bash format.sh ci`仍被历史文件格式/拼写/导入问题及本机缺shellcheck阻塞；
未把这些历史格式化改动带回正式源码或原始证据。
