# 当前目标：动态单SPMD CSA相对PTO多task提升10%

用户新目标：参考https://gitcode.com/cann/cannbot-knowledge/tree/master/knowledge/ops，
尽力而为地做性能优化，目标比PTO多task的incore L1性能提升10%。
2026-10-21用户明确重点档为128K/B1、B16、B28。

## 指标与范围

- 公式沿用户已指定口径：(PTO多task耗时−动态SPMD耗时)/动态SPMD耗时。
  目标>=10%，等价于动态耗时<=多task/1.1；不能混写成耗时降低10%。
- 两者均采用完整L1最早incore开始到最晚结束，包含内部等待，不计外层AICPU头尾。
- 保留八档：128K/B1/B4/B8/B16/B24/B28、8K/B16/B32。优先128K/B1/B16/B28；
  128K其余档随后覆盖，8K兼顾且不能明显退步。测试清单PERFORMANCE_CASES.json。
- B1允许针对性分档；其它规格用batch、历史长度、资源等区间策略，禁止B16/B28单点白名单。
- S6主测，必要的S5/S1、动态metadata、边界、图重放、同步代际回归保持。

## 实现与比较合同

最终为一个完整动态单SPMD CSA，正式源码仍在ops/pypto/deepseek_v4_flash_csa_spmd。
不改为历史静态shape；不要求一份二进制支持全部batch。真实长度/位置/页表/slot
从device Tensor读取。正式W8A8/NZ权重，精度、Top-K/score及同分规则、cache/state、
保护区合同不放宽。数值非逐bit优化采用前仍单独确认。

同轮冻结输入、权重、多task、原动态参考、候选和工具链。交替五窗，不择优窗口、
不跨轮拼接收益。公共多task已在其它会话更新，新目标建立独立快照，保留旧实验。
先依据已有关键路径与机制差异选择候选，取消20μs最低收益门槛，但不能把波动当收益。

参考cannbot知识时核对其适用DSL、芯片和限制；不把Triton/TileLang案例收益直接外推到PyPTO。
优先减少重复计算、无效搬运、串行交接，并核对实际生成代码/资源峰值与完整L1收益。
旧负例只有新机制或证据才复测。用户此前放弃Attention方向，不能仅因新总目标自行恢复。

PyPTO、Simpler、PTOAS可在独立worktree改进，既有隔离工具链能力须继承，共享安装与pypto-lib不改。
组件bug及需要转达的能力问题提对应上游issue附复现建议，不提PR、不加标签；Simpler非bug扩展留本地。
过程持续记录本目录及ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md，详细中文阶段commit。
不提交权重、二进制和大编译快照，不做hash。未经新授权不push。

## 恢复点

正式源码已采用B1原半叶/分数与归并根留UB，当前快照source_after_b1_half_v2。
采用记录B1_HALF_SCORE_ADOPTION.json，最新重点档表GOAL10_FOCUS_PERFORMANCE。
B1直接配对净省29.36μs，但相对新多task仅4.21%，总体10%目标仍未完成。
B16列池正向、B28混合，未整体采用。所有原任务已终态，下一步见CURRENT_CHECKPOINT。
旧快照与失败记录保留，不能把旧4K同分失败候选或旧多task基线当当前结果。
