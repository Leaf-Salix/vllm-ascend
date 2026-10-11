# 预置完成标记实验：功能通过，候选已撤回

本目录是失败候选的证据归档，不代表当前保留实现。仅 O-A 不再作为独立版本测试或维护。

## 改动与对照

编译器识别已有的无条件局部 Set，在提交时增加内部能力标记；Simpler 按 AIC/AIV
预置完成类型。正常 Set 不再写 GM 标记，未被交接或诊断修改的控制行省去清零与写回。
Wait 取消时清除预置标记，避免等待未发出的事件；真正的流水等待、FIN、任务依赖、
算术、tiling 和 sync_start 均保留。PTOAS 本轮没有修改。

三组均使用同一份候选 Simpler SDK `2fb78a6ee9cdc7f365a5ebf926e146ce6cb34b26`：

- `baseline`：普通同步尾部。
- `previous_all`：原扩展版，Set 写 GM 完成类型；仍由同一候选运行时执行旧协议。
- `all`：新预置标记协议。

PyPTO 基于 `3d88caef9` 加 [候选补丁](pypto.patch)，PTOAS 为 `4ef2f93c`，私有 vLLM 为
`2cd33a9a`。原协议由 [编译驱动](compile_compare.py) 在测试期间恢复旧 Set adapter，
并移除对应的预置提交字段。它没有改变算子源文件。
67 个 PTO 文件在一一对应重命名 SSA 后相同，调度源文件除该提交字段和自动变量名外相同；
共有 60 处提交字段，见 [范围检查](source_scope_checks.json)。

## 测量结果

任务 `task_20261011_121205_282836328294`，单卡 0，退出 0，设备已释放。
CANN 9.2、TMR、NZ=2、atomic_add 关闭；128K/B16 第二层真实权重，
计时包含 HC_pre+norm+CSA+HC_post。三组各 30 次单次图重放，轮换顺序，reset 在计时外，
关闭 profiler。此处的普通版也是 PTO 路径，不是 Native 算子基线。

| 路径 | min μs | mean μs | max μs | P95 μs |
| --- | ---: | ---: | ---: | ---: |
| 普通版 | 976.26 | 989.97 | 1015.86 | 1005.48 |
| 原扩展版 | 959.40 | 983.40 | 1007.92 | 1002.02 |
| 预置标记版 | 970.04 | 986.54 | 1103.26 | 1002.04 |

新候选相对原扩展版均值 **+3.14 μs（+0.32%）**，16/30 配对轮次较慢，
配对差值中位数 +0.61 μs。三个轮换顺序下的均值差分别为 +2.22、−3.99、+11.18 μs，
没有稳定方向。第 5 轮候选出现 1103.26 μs 最大值；P95 基本不变，不能称为 P95 整体恶化，
也不能在没有额外诊断的情况下将这一次最大值归因于某条指令。

候选相对普通版均值 −0.35%，但原扩展版已经更快，不能据此接受新改动。
本轮绝对值亦不能与旧 SDK 的历史均值直接相减，宣称某项修复的净收益。

## 功能与处理

8 项调度器 CPU 检查、3 项 PyPTO 定向代码生成检查、3 项基础场景编译和 3 项基础真机用例通过。
基础诊断捕获 AIV 同核计算重叠 16/48、48/48，AIC 为 24/24、24/24；这些不是 CSA 的命中率。
两个候选各 9 项输出/状态与普通版逐元素精确一致（torch.equal），改变输入后的图重放也一致，
三组各 31 项保护区全部通过。未进行整模型 token 测试，也未额外验证真实退出竞态。

预置标记候选已撤回，Simpler `9d6757269` 恢复到 `58f6581eb` 的源码内容；
PyPTO `9daa07428` 恢复原接口实现，仅更新匹配恢复后 SDK 的 pin 和 gitlink。
私有前端、绑定、TMR 运行时及 torch 适配库均已重建，三方 SDK 检查通过；
恢复源码与此前已验证实现一致，未再占卡重复测试，见 [恢复检查](restore_checks.json)。
保留 [Simpler 候选补丁](simpler.patch) 和数据供分析，不将其列为已接受优化。
后续应先获取 CSA 实际交接及关键路径证据，再决定减少哪段协议成本；净收益目标仍未完成。

原始产物在私有根 `.worktrees/scalar-overlap-demo-20261009/`：

- 编译：`artifacts/csa_prearmed_tail_compile_v1/`。
- 设备入口：`artifacts/run_csa_prearmed_tail_v1.sh`。
- 设备结果：`artifacts/csa_prearmed_tail_device_v1/results.json`。
- 基础设备验证：`artifacts/prearmed_tail_local_device_v1/`。

本目录保留 [全部 90 个性能样本及配置](results.json)、[配对统计](analysis.json)、
[检查摘要](checks.json)、[调度器检查](scheduler_checks.txt)、
[AIV 原始诊断](local_aiv_trace.json) 与 [AIC 原始诊断](local_aic_trace.json)。
