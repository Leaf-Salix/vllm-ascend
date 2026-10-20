# TND精度版：纯DSL独立计算前移与按依赖等待对拍（2026-10-21）

## 结论

保留正式精度基准，未修改SDK、生成C++或默认路由。所有实验算子均由现有DSL生成；没有采用nalinaly的C++等待helper退避。

**找到形状相关的有效改动：只等所需WO-B生产者，T240两次独立任务比冻结后缀快17.52%和16.64%，每次12/12窗口领先。**
但T126仍慢冻结2.37%，T144仍慢4.78%。额外提前系数计算后T126仍慢0.85%；尚未满足本轮同时覆盖短T与两个目标形状的推广门槛。

因此本轮只归档实验候选，不接入正式完整CSA，也不启动8K/B40或128K/B24的整模型token测试。
这里的逐位通过仅指真实第4层权重、合成后缀输入的五边界对冻结PTO基准，**不是新增整模型Native token验收**。

## 分支与基准

- 分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
- 本轮文档起点：`8fe011517431e11e20e27d9ad64f9c9569b19db3`。
- 正式数值源：`48c58ce7c467dfce63d5cb5c41afd225488d6012`，保持不变。
- owner2父候选SHA256：`399dbc67a1709a5e2fda636afb245d6e6478a5f1cec20763471caae6fd98e54b`。
- 被测运行目录为固定SHA的私有实验副本；不将overlay说成独立Git提交。

## 单因素与依赖检查

| 候选 | 父版 | 唯一策略变化 | 设备前检查 |
| --- | --- | --- | --- |
| channelpreload | owner2 | 原通道scale载入、16×512系数展开移至phase4等待前，并复用于本worker后续块 | 列恒定、跨等待UB不覆盖、AST逆变换一致 |
| obowners | owner2 | 每个16×512反量化块只等所需4个N128 WO-B生产者，再保留原payload acquire；先处理不依赖其余生产者的块 | 实际AIC写者→AIV读者T1…256覆盖、无反向依赖、每块acquire |
| postowners | owner2 | 每4token HC-post块只等所需8个512列反量化生产者，再保留原payload acquire | 4token不跨16行、8owner完整覆盖、原pipeline不变 |
| coefearly | obowners | 先等覆盖本块有效scale行的两phase3槽并获取payload；提前原系数计算，再等原4个phase4槽读取integer | 尾部第二槽可为idle发布者；系数跨等待UB不覆盖，原FP32乘法/舍入顺序不变 |

前三个候选分别对同一owner2测试，未将它们组合。第4个只以obowners为父版，不叠加channelpreload或postowners。
T1…256枚举是源级覆盖证明；不等于每个T都做过设备测试。4个候选均完成实际PTOAS/CCEC编译、ABI16注册、KernelArtifact.load及独立PTO/C++只读审查。

原TNOTIFY、代际、cache写回/获取和实际owner的完成约束保留。减少TWAIT调用数不等于按同一比例减少失败轮询或GM事务；本轮未采PMU/泳道，不能据此拆分纯搬运、计算或轮询贡献。

## 性能与精度结果

单位μs，为同任务12窗×100次FULL graph replay的中位数。表中“较父版/冻结”是耗时变化，负数表示更快；不跨任务相减作为组合收益。

| 候选/独立重复 | T | 冻结后缀 | 同任务父版 | 新候选 | 较父版 | 较冻结 | 领先冻结窗口 | 精度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| channelpreload-t126-v1 | 126 | 404.66 | 497.51 | 459.22 | -7.70% | +13.48% | 0/12 | 160项逐位一致 |
| channelpreload-t240-v1 | 240 | 472.04 | 525.10 | 549.87 | +4.72% | +16.49% | 0/12 | 160项逐位一致 |
| obowners-t126-v1 | 126 | 402.01 | 499.44 | 411.53 | -17.60% | +2.37% | 0/12 | 160项逐位一致 |
| postowners-t126-v1 | 126 | 402.01 | 468.65 | 476.65 | +1.71% | +18.56% | 0/12 | 160项逐位一致 |
| obowners-t240-v1 | 240 | 470.15 | 531.40 | 387.78 | -27.03% | -17.52% | 12/12 | 160项逐位一致 |
| obowners-t240-v2 | 240 | 466.58 | 566.74 | 388.96 | -31.37% | -16.64% | 12/12 | 160项逐位一致 |
| coefearly-t126-v1 | 126 | 399.64 | 413.40 | 403.03 | -2.51% | +0.85% | 0/12 | 160项逐位一致 |
| obowners-t144-v1 | 144 | 452.72 | 618.33 | 474.37 | -23.28% | +4.78% | 0/12 | 160项逐位一致 |

channelpreload T240回退、postowners T126回退，均不保留为通用优化。coefearly T126对obowners有局部收益，但仍未胜冻结，不继续扩大。
obowners是本轮最有价值的实验结果，但只确认T240稳定领先；不能据此默认T≥某阈值全部有效，更不能把T240等同完整8K/B40工作负载。
T144对应B24×S6的最大真实行数，仅检验对应后缀形状，没有加载128K历史、TND请求边界或完整模型。

## 精度与测量方法

227单卡、TP1、DeepSeek-V4-Flash-0731-w8a8真实第4层WO-A/B，NZ2。heads/residual/post/comb为固定seed合成值。
Native数学与K归约起点/顺序来自冻结PTO；本轮gold为冻结后缀，不新增Native完整算子精度结论。

每任务先eager，然后FULL A/B/A输入扰动及恢复。五边界为WO-A BF16投影、INT8量化、FP32 token scale、WO-B BF16输出、HC-post最终输出。
所有外层guard、只读SHA、NaN/反码毒化和代际均通过；每个resident臂最终1204代。8个成功设备任务合计**1280项逐位指标**，未发现首差。
每窗计时后继续复核；保留全部样本，不开启DFX/依赖图/PMU，不删离群值。三臂顺序轮转与反转，每臂每位置均4次。
设置Torch确定性算法后再设置Torch-NPU确定性level1，并启用HCCL确定性。

## 版本、时限与复现

PyPTO `2dfa7f120ba5e577566046d920845aa241bfa7bb`，Simpler `57e499238fd4ee3548947ad860be456aaeeb16b5`，PTO-ISA `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。
Python3.12.14、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2。
实际PTOAS为隔离0.66 wheel，SHA256 `d121b5b5db9a23a98eb3c0acc74d0abaef8aaa90e4df0385e79feb67636ef997`，wheel源提交未证明。
调度命令的`--ptoas 0.63`不代表实际编译器版本，脚本显式设置PTOAS_ROOT；每个raw JSON记录实际导入/DSO SHA。

所有NPU任务内层750秒、外层780秒；8项最终均exit0，实际launcher耗时97～115秒，最终状态与日志随证据保存，未遗留自有任务。
复现前置条件、源码/父链、完整task-submit命令、CPU生成物和原始逐窗数据见[证据目录](evidence_tnd_precision/woa_hoist_20261021/README.md)。

## 后续门禁

保留冻结精度版和独立obowners实验。先解决T126/T144剩余差距，或对可交付的分流方案单独证明其完整CSA开销与数值正确性；本轮没有实现该分流。
完整CSA集成还需显式初始化所有信号stamp、强依赖和写回、实际内存峰值与Native逐位对拍，不能直接把本轮外部workspace后缀接入模型。
只有满足推广性能门禁并完成完整CSA精度检查，再按8K/B40→128K/B24顺序检查192-token Native一致性，每项任务继续小于15分钟。
