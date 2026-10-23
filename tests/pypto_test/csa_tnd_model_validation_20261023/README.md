# CSA TND 真实模型功能验证续作

本轮接续 `csa-repro-20261021-clean/kv-repro` 中断的真实模型验证，
先完成 128K/B16/S6 的 Native、TND-SPMD、TND-MPMD 三侧对照，再验证 B24。
不改变已完成的 PyTorch 单卡复现手册，也不重复生成已审计的历史 KV。

## 固定口径

- 真实权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，TP1、DP=EP16，DSpark 出5验6。
- CANN 9.2.0-beta.2；PyPTO/PTOAS/Simpler 从公共清单绑定，框架和算子按提交冻结。
- 只切换 CSA；20 层 HCA 始终 Native。NZ mode=2，EPLB 关闭。
- 保留 `npugraph_ex`、`FULL_DECODE_ONLY`，关闭静态编译，不加入 SuperKernel。
- 确定性 level=1、HCCL_DETERMINISTIC=true、AIV 展开模式、event_work_mode=1 三侧一致。
- 同一批混合问题、每请求192输出token；固定历史实际长度131104..131346，来自128K公共前缀加问题后缀。
- 首轮 B16 显存预算三侧均0.92。真实S6满档、无抢占是功能门槛，不能缩小实际并发来绕过OOM。
- 功能检查与精度对照分开：功能检查要求实际21层CSA路由、完整token和DSpark事件；
  精度对照报告逐请求token、首处分歧、DSpark接受长度分布和简单问题答案，不宣称浮点bit一致。

## 历史中断点

旧 `decode-native_clean2` 已生成256请求×192token，平均接受3.80650。
旧 `decode-spmd_clean2` 在首个S6 decode申请210 MiB时OOM，MPMD随后未执行。
另一次8K MPMD尝试也在相同申请量处OOM。两次都使用0.95显存预算及静态编译。
这些是诊断线索，不能当作当前版本的结果或与本轮直接计算性能收益。

历史 KV：`/data/pyptouser/qinchuanyu/pto-eager/csa-repro-20261021-clean/kv-repro/mixed128k_fixed`。
只读使用前16/24个不同问题；完整40问题及生成、审计证据仍在原目录。

## 执行

`prepare.py <实验名>` 冻结 HEAD、当前测试入口和公共SDK清单，
不会带入其他会话未提交的算子修改；不复制SDK、权重或KV。
用 `preflight.sh <冻结源码> <结果JSON>` 先排单卡核对Native依赖和两种TND依赖图。
然后通过 `task-submit` 为 `run.sh <冻结源码> <输出目录> <native|spmd_tnd|mpmd_tnd> <16|24> <预算>` 分配16卡。
每次使用新输出目录。B16三侧及输出对照通过后，才能开始B24。

`check.py` 检查真实路由和功能完成，不代表Native/PTO输出一致。
各次实验的任务号、结果及对照结论写回主 `DSV4_FLASH_CSA_VALIDATION_LOG.md`。

## 当前证据

- `native_b16_summary.json`：Native 16卡B16功能通过，无抢占，平均接受3.80650；
  32个简单答案通过，跨DP及旧Native输出token差异均为0。
- `workspace_validation.json`：两种TND的临时工作区复用单卡验证通过。
  128K/B16的S1/S6及两层交替，eager和graph的输出、Top-K、cache/state逐bit一致，保护区通过。
- `workspace_pool_estimate.json`：CPU meta容量估算，不是模型设备峰值。
- `jobs.json`：两种PTO B16整模型任务及其前置验证路径。整模型和B24尚待完成。

有工作区修复的冻结包使用`prepare.py <实验名> --revision 99268437c --workspace-fix`；
启动`run.sh`时第六个参数提供单卡`result.json`。该结果必须包含两种实现的PASS，
否则启动器直接退出。该门槛不替代后续整模型正确性检查。
