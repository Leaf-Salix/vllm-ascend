# G2/768 的 N128 已是正式配置

2026-10-18 只读核实：当前 G2/score_tile768 不是 N64，而是 N128。
因此“仅把短 G2 的 QK_PANEL64 改为128”相对正式参照 d56c9eb6 没有新增改动，
无需 proposal、CPU 编译或设备试验。本次只新增本说明；未更改源码或运行模型。

正式 [policy.py 第95行](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/policy.py:95)
为 `qk_panel=64 if group == 6 else 128`；第96、97行分别选择短档768与PREFETCH=0。
[d56c9eb6 冻结参照的同一位置](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/adopted_source_d56c9eb6/policy.py:95)
及本归档 `evidence_v1/source_snapshot/policy.py:95` 均保留该配置。

既有完整动态单 SPMD 设备三方矩阵的实际 controls 可直接核对：

- 8K/B16，`task_20261018_003818_31105785398`：
  [source_matrix/result.json 第8行](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/indexer_h8192_b16_v1/source_matrix/result.json:8)。
  候选第8/9/10行依次为 G2/N128/768，旧 SPMD 参照第23/24/25行相同。
- 8K/B32，`task_20261018_004709_328630619605`：
  [source_matrix/result.json 第8行](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/indexer_h8192_b32_v1/source_matrix/result.json:8)。
  候选第8/9/10行、旧 SPMD 参照第23/24/25行均为 G2/N128/768。

两个原始结果第2行都是 `passed: true`，第4行均为同进程；短档 controls 同时为
PREFETCH=0、BALANCED=false、ROOTS=2。这些作业比较的是 AIV 半叶 score 缓存，
不是 N64→N128 单旋钮消融，因此不能从其耗时推导 panel 改动收益。它们足以证明
N128 已在当前这条完整动态路径运行，不能将长 G6/M384 的 N128 容量失败当作这里的
否决依据。后续按主 agent 要求结束，不扩展模型、资源审查或编译。
