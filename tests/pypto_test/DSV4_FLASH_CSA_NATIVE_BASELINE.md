# Native基线：对齐decode部署模板

更新：2026-09-29。用户指定参考
[run_dp_template.sh](../../../vllm-ascend-main/tests/dsv4_perf_accuracy_20260827/runtime/decode/run_dp_template.sh)
及其[runtime/config.sh](../../../vllm-ascend-main/tests/dsv4_perf_accuracy_20260827/runtime/config.sh)。
本次只改测试入口和证据说明，不改变生产Native算子、权重或已测结果。

## 核查结果与修正

旧单层入口以enforce_eager构造Native对象，再手工捕获NPUGraph；这不是npugraph_ex/static kernel编译。
旧整模型554b3bca的Native实际日志确认npugraph_ex=True、static_kernel=False，CPU绑核已执行。
共享专家多流默认False，recompute默认False；两侧共同关闭了fuse_norm_quant，OMP线程数为4。
所以旧结果仍是其记录配置下的测量，但不能代表本模板优化配置下的Native基线。

| 项目 | 模板 | 修订后的离线decode入口 |
| --- | --- | --- |
| enable_npugraph_ex | True | 两侧显式True；eager诊断False |
| enable_static_kernel | True | 两侧显式True；eager诊断False |
| fuse_norm_quant | 默认True | 两侧显式True，移除旧环境绕行 |
| enable_cpu_binding | True | 两侧显式True；仍需设备日志确认绑核成功 |
| multistream_overlap_shared_expert | True | 两侧显式True；实际回退须报错，不能静默比较 |
| recompute_scheduler_enable | False | 两侧默认显式False；旧专项CLI只用于标明配置的诊断 |
| async_scheduling | True | 两侧显式True |
| disable_hybrid_kv_cache_manager | False | 两侧显式False |
| weight_nz_mode | 环境2 | CLI默认2；显式0/1仍用于布局诊断 |
| OMP_NUM_THREADS / OMP_PROC_BIND | 10 / false | decode一致；prefill保留原4线程 |
| HCCL_OP_EXPANSION_MODE / BUFFSIZE | AIV / 1800 | decode已一致，继续保留 |
| PYTORCH_NPU_ALLOC_CONF | expandable_segments:True | 已一致，继续保留 |
| VLLM_BATCH_INVARIANT | 0 | 显式0 |
| 模型加载线程 | 128 | decode128；prefill保留16 |
| gpu_memory_utilization | 0.95 | 默认0.95；128K/B24仍按已验证容量显式0.97，两侧相同 |
| VLLM_RPC_TIMEOUT / EXECUTE_MODEL_TIMEOUT_SECONDS | 3600000 / 30000 | 一致 |

每个Worker返回实际Ascend编译配置、共享专家开关、recompute、async/HMA、prefix及OMP/HCCL环境，
写入原有rank报告。decode在正式采样前核对请求值与生效值，编译或重叠被降级时不能继续产出合格结果。
这只是配置生效检查；静态编译和融合是否覆盖目标CSA算子，仍须查看编译/设备profile。

证据：[旧Native编译日志](results/csa_key_l1_seven_20260928/model/h131072/native/rank0.log)、
[新入口](offline_pd/run.py)、[Worker实际配置](offline_pd/worker.py)。

## 保留并明确记录的工作负载差异

- 正式权重仍为用户已锁定的`/data/model/DeepSeek-V4-Flash-0731-w8a8`；不换模板config中的另一份权重。
- 继续CANN9.2、TP1/DP=EP16、DSpark出5验6、mode2、EPLB关闭；PTO使用已验证私有整包。
- 七档仍为128K B4/8/16/24、8K B24/32/40。max_num_seqs、capture_sizes、token预算与max_model_len
  按场景显式记录；不把模板B32、1M容量配置直接覆盖到这些负载。两侧使用相同档位与容量参数。
- 模板是在线MooncakeHybridConnector；本测试使用已验收离线bank恢复，恢复在稳态forward计时之外。
  现有connector明确拒绝local prefix hit，测试要求请求独立KV，因此保留enable_prefix_caching=False。
  这不是在线prefix-sharing性能或容量验收；不能把结果宣传为模板完整服务吞吐。
- profile等长诊断的HCCL_EXEC_TIMEOUT可保留1800秒等待余量，常规decode仍为204秒；均不计入正常forward耗时。
- 不包含在线HTTP、tool/reasoning parser或网络传输；主指标仍为预热8步后10步decode forward。

## 环境缺口与下一步

本轮CPU符号检查：指定CANN9.2的libopapi.so及当前csa-native-ops-install的libcust_opapi.so
都没有aclnnAddRmsNormBias。norm_quant_fusion_pass的pattern通过real tracing调用本仓
npu_add_rms_norm_bias，所以只打开fuse_norm_quant不构成可运行环境。
已用当前release的csrc/moe/add_rms_norm_bias构建独立vendor补充包；不覆盖现有vendor，
不再静默关闭融合。构建记录见[对齐过程](results/csa_native_template_20260929/README.md)。

1. 已补齐AddRmsNormBias；单卡验证真实调用、融合注册、static kernel编译成功、静态包安装和图重放。
   CANN9.2只读OPP改用私有可写根，原tiling库内容不变；两侧source结果目录env.sh后运行。
2. 已补真实Native半层模板编译：128K/B16从同进程手工图1294.493降到1238.764μs（−4.305%），
   static_compile实际True、安装包和覆盖QLI/Compressor/Sparse的39个算子描述已确认。
   Top-K集合不变、3行顺序变化；浮点输出有差异，不能直接称精度通过。旧手工图报告继续标明范围。
3. 同卡同配置Native/PTO编译对照已补：128K/B16为1217.529/1090.391μs，8K/B24为1026.713/988.208μs。
   每侧独立空static_kernel目录；PTO真实custom-op调用确认。长档PTO P95异常，正在定位，不以均值代替稳定性。
4. 核内优化继续按长短8:2；阶段收口才覆盖七档及真实EP16，避免每个配置问题都占16卡调试。

当前已完成入口修订、10项CPU参数/Worker回归、单卡编译依赖及两档Native/PTO实际编译半层对照；
长档P95、两侧数值和整模型仍待验收，不修改旧七档表的百分比。
本单卡attention半层不含MoE，也未通过Worker实际绑核，不冒称整机模板已完整验收。
[新单层结果、数值差异与profile](results/csa_native_compiled_layer_20260929/RESULTS.md)。
[同配置Native/PTO对照及范围](results/csa_compiled_pair_20260929/RESULTS.md)。
