# 四路Top-K：按实际长度选择独立二路/多路核

来源仍为最新ops-transformer b5b33e14的QLI V2 A3 ProcessLD。
[首轮通用核](../csa_topk_fourway_20260928/README.md)在128K/B16的merge核内降低21.53%，
但8K/B16核内增加7.14%、完整CSA/P95也升高；长档收益保留为依据，短档回退不忽略。

本候选冻结在`.cache/csa-topk-fourway-adaptive-e58ddc94`，e58ddc94基底。
仅性能版decode_indexer.py变化。已有orchestration已计算max_topk_cache_len，
据此在单leaf时传constexpr False，生成原二路循环核；多leaf时传True，生成首轮四路核。
阈值为压缩后实际cache长度8192，沿用TOPK_CANDIDATES_PER_LEAF，非硬编码测试H8192/128K标签。
极短输入原single_leaf_publish分支不变；混合长短请求由批内最大长度选择，多路核仍按每query可见根数处理尾块。
同一套源码和捕获图支持两条分支；任务worker数、依赖和early_resolve不改。
不叠加Score/Sparse的sync_start，不修改Native cache、精度版或运行时。

二路核的目标是消除无用多路分支及临时量，不能提前断言首轮全部20.54μs回退都由此造成。
完整CPU lowering/PTOAS/CCE/链接通过。归并原语及tie规则未改，复用首轮五组真实NPU边界证据，
不重复运行相同小用例；本轮补长度分支接入后的长短完整层状态、A→B→A及计时。

任务`task_20260928_130855_23326697870`，一张卡，最长2400秒。
先8K/B16后128K/B16；两侧各20次正式图计时、5次预热、四个独立DFX窗口，两档交替候选/基线次序。
layer4真实权重/合成历史，S6、mode2、atomic0、det0、EPLB关闭、第二个CSA层metadata复用。
基线为本轮同卡e58ddc94，不拼接上一任务的计时数据。八类状态/保护区/图重放要求精确；
det0 Native浮点只作控制，未保存idx_topk_scores，五组原语小用例另有score/index位模式检查。
当前设备任务进行中，尚未合入；核内、完整CSA、P95和最终EP16分别验收。

[源码补丁](candidate.patch)、[CPU编译入口](compile.py)、[编译日志](compile.log)、
[设备命令](run_layer.sh)、[汇总命令](summarize.sh)。
