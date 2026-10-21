# TND CSA 多 task 实现

以 `84751e34` 的 `deepseek_v4_flash_csa` 多 task 编排为骨架，结合同提交
`deepseek_v4_flash_csa_spmd_tnd` 的真实 Q/L、混长、padding 和核内数学实现。

`entry.decode_csa_tnd` 包含独立的 HC 前段、QKV、两条 Compressor、Indexer
投影与 cache 写回、Score、Top-K、Attention plan、Attention、O projection
和 HCpost。仅 Score 与 Attention 各自使用局部 MIX；完整 CSA 保持 MPMD。
跨 task 使用 Tensor/TaskId 依赖，不沿用常驻单 SPMD 的 GM phase 轮询。

`binding.PreparedDynamicCSA` 在捕获前准备工作区、绑定 Native 原存储，直接
接收 `hidden[T_capacity,4,4096]`、设备 `query_start_loc`/`seq_lens`、页表、
slots 和 compact metadata。活动请求为 S1～S6；T/B 分别表示 token/请求
容量，不能相除推导 query 长度。失活请求及尾部 padding 由设备元数据排除。
Host 只使用原有描述符，逐请求计划由设备 task 生成并计入完整 L1。

当前为独立算子验证阶段，尚未注册新的 vLLM variant。功能迁移与性能目标
分别记录：初版具备多 task 路径，并不表示已超过冻结 SPMD 或达到 10% 目标。
进度与设备证据统一追加到
[`DSV4_FLASH_CSA_VALIDATION_LOG.md`](../../../../tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md)。

单卡验证工具与固定场景在 `tests/pypto_test/csa_mpmd_tnd_20261021/`。
先通过 `prepare_experiment.py <名称>` 冻结实验副本，再通过 `task-submit`
执行该副本的 `launch.sh <实验目录> <新输出目录> <场景名>`。CPU 预编译
增加 `--cpu --compile`，无需占卡；CPU 成功不代替设备数值验证。

单卡以完整 HCpre→RMSNorm→CSA→HCpost 的 L1 首尾跨度验收，记录五个
交替窗口的 min/mean/max。最终目标还要求真实 vLLM TND 路径与 16 卡验证。
