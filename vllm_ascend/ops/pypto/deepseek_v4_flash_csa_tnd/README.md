# TND CSA 多 task 实现

以 `84751e34` 的 `deepseek_v4_flash_csa` 多 task 编排为骨架，结合同提交
`deepseek_v4_flash_csa_spmd_tnd` 的真实 Q/L、混长、padding 和核内数学实现。

`entry.decode_csa_tnd` 保持多个可独立调度的task，包含HC局部流水、metadata、
Q_A/QR、Indexer和Main两条前段、Score/Top-K/Attention以及输出局部流水。
长历史BALANCED路径把Score到Attention按行衔接，其他路径保留四个独立task；
BALANCED分支12个task，其余15个（本轮长B4也走15个task）。跨task使用
Tensor/TaskId依赖，局部MIX内部
使用独立代际信号和实际生产者等待，没有把完整CSA改为单SPMD。

`binding.PreparedDynamicCSA` 在捕获前准备工作区、绑定 Native 原存储，直接
接收 `hidden[T_capacity,4,4096]`、设备 `query_start_loc`/`seq_lens`、页表、
slots 和 compact metadata。活动请求为 S1～S6；T/B 分别表示 token/请求
容量，不能相除推导 query 长度。失活请求及尾部 padding 由设备元数据排除。
Host 只使用原有描述符，逐请求计划由设备 task 生成并计入完整 L1。

当前代码采用已完成固定八档设备验证的v49组合：HC按有效行分工并局部衔接，
大容量保留8行gate/norm；Indexer完整M8的head分组与固定RoPE索引准备优化；
长G6 Score按实际live选择固定M64/128/192，保持六行FIFO及双半累加顺序。
Indexer/Main默认8/16个AIC核组，KV纳入Main，输出逐组衔接O_A/量化/O_B/HCpost。
信号由准备对象首次清零，各角色每调用推进一次代际，不能跨并发流共享工作区。

固定八档完整L1相对冻结SPMD：G128=0.964896、G8=1.017087、综合G=0.975115，
尚未达到0.90目标。所有八档的eager/图/采样后输出、Top-K、cache/state逐bit
一致、保护区通过；B28存在前段启动波动，短B32仍慢于参考。具体min/mean/max、
全部样本和限制见验证日志735.35～735.42及
[`hc_score_rope_matrix.json`](../../../../tests/pypto_test/csa_mpmd_tnd_20261021/hc_score_rope_matrix.json)。

已注册独立 vLLM variant：`PTO_CSA_VARIANT=mpmd_tnd`，服务输入、图与回退
合同复用 TND 路径，权重签名、策略和实际提交使用本包。CPU 合同检查已通过，
新组合的同址metadata、连续状态及真实整模型验证仍待完成；单卡结果不代替
这些验收，也不表示已达到10%性能目标。
进度与设备证据统一追加到
[`DSV4_FLASH_CSA_VALIDATION_LOG.md`](../../../../tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md)。

单卡验证工具与固定场景在 `tests/pypto_test/csa_mpmd_tnd_20261021/`。
先通过 `prepare_experiment.py <名称>` 冻结实验副本，再通过 `task-submit`
执行该副本的 `launch.sh <实验目录> <新输出目录> <场景名>`。CPU 预编译
增加 `--cpu --compile`，无需占卡；CPU 成功不代替设备数值验证。

单卡以完整 HCpre→RMSNorm→CSA→HCpost 的 L1 首尾跨度验收，记录五个
交替窗口的 min/mean/max。最终目标还要求真实 vLLM TND 路径与 16 卡验证。
