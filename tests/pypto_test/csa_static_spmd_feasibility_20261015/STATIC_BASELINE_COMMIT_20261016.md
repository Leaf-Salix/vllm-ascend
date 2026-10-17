# 静态单 SPMD 提交边界（2026-10-16）

本次提交目标：nalinaly/vllm-ascend 的 dsv4-flash-pto-v0.25.1rc1。
基于远端 f7ee8dda 建立独立提交 worktree，不混入其他会话的未提交代码。

- static/ 保留七档原始静态模板；static_local_signal/h131072_b16/ 保留已通过的临时notify/wait版本。
- benchmark_static_vs_dynamic.py 及性能报告保留128K/B16/S6同卡动态PTO对照，单SPMD当前更慢，未写成优化成功。
- evidence/ 保留逐次尝试的结果、日志、源码快照和六份L1泳道，包括失败尝试。
- reference_operator_source/ 是本次测试冻结参考算子的完整Python源码，不改变生产算子。
- references/symbolic_ir/ 保留两份生成器需要的符号IR；其他重复编译生成快照本地保留。
- GIT_RETENTION.json 记录提交/本地保留的路径与大小；artifact_inventory.json 保留全部大产物本地路径。
- component_changes/ 保留独立PyPTO和Simpler修改补丁；没有向这两个组件仓推送。
- 共享安装、生产CSA/HCA入口及pypto-lib均不在本次修改范围。

静态结果已经验证固定输入ACLGraph重放，不代表动态batch/seqlen已完成。
动态shape候选在独立目录开发，不覆盖这里的静态实现和历史结果。

复现环境仍使用已有bootstrap_isolated.py和env-dsv4-0251rc1.sh。
若原机器results/hc_pre_layer_integration_20261012/baseline_ops不可用，可将
HC_PRE_TEST_OPERATOR_SOURCE 指向此目录的reference_operator_source/；不得误用后续改动过的生产算子作为原静态基线。
