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
| super_kernel_optimize | 模板未设置 | 当前Native整体对照True；核内诊断False，两者均保留static compile；不再做收益消融 |
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
- 2026-09-29起正式七档为128K B4/8/16/24、8K B16/24/32，B40退出后续对比。max_num_seqs、capture_sizes、token预算与max_model_len
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
3. c93ec723同卡新七档真实编译对照已完整完成：128K B4/B8/B16/B24、8K B24/B32/B40。
   每侧独立OPP静态包/AOT缓存；Native实际安装和PTO custom-op调用已确认。
   长B16 Native/PTO为1232.308/1043.478μs，PTO P95为1061.640μs；
   全七档长短8:2均值变化−10.312%。本轮未复现历史1.4ms拖尾，不宣称修复。
4. 后续按长短8:2进行定向核内/调度A/B；阶段出口再覆盖受影响档位，真实EP16优先级后置。

当前已完成入口修订、CPU参数/Worker回归、单卡编译依赖及新七档Native/PTO实际编译半层对照；
历史间歇P95、两侧逐元素精度和整模型仍待验收，不修改旧矩阵原始读数。
本单卡attention半层不含MoE，也未通过Worker实际绑核，不冒称整机模板已完整验收。
[新单层结果、数值差异与profile](results/csa_native_compiled_layer_20260929/RESULTS.md)。
[同配置Native/PTO对照及范围](results/csa_compiled_pair_20260929/RESULTS.md)。

[当前七档真实编译结果](results/csa_coefficients_seven_20260929/RESULTS.md)、
[21份JSON](results/csa_coefficients_seven_20260929/download/README.md)。

2026-09-29核查：当前冻结compiler_interface只传static_kernel_compile，不设置super_kernel_optimize；
安装的npugraph_ex配置默认False，其图优化调用受该标志控制。现有Native收益应表述为
npugraph_ex、static kernel与融合的组合收益；已有同进程CANN9.2对照长B16−4.305%、短B24−6.415%，
没有只切static kernel的消融，不能拆出其单项贡献，也不能归因于未启用的super kernel。

## 最新入口要求与superkernel判据（2026-09-29）

4ffccb7b同源码新版七档已完成：128K B4/8/16/24及8K B16/24/32，长短8:2为−10.853%。
该轮Native仍使用vLLM Ascend编译包装，force_eager后手工外图捕获，superkernel关闭。
用户现在要求显式torch.compile(..., backend="npugraph_ex")；新对照由后端自行捕获/重放，
不再嵌套手工图。直接入口应用npugraph_ex自身passes，未走vLLM FX pass manager，二者须单列。
多流沿用torch.npu.stream及event/wait显式依赖，DSA自定义算子内部在捕获时实际下发，
结合profile检查真实stream；共享专家配置不代表单CSA包含MoE验收。
GitCode文档必须直连无代理，同event不得跨graph break，fullgraph编译失败不降级为eager。

先仅长B16/短B24对照static kernel+superkernel开/关；记录静态编译实际参数、图优化调用与profile。
有明确收益且数值检查通过才采用、再更新受影响Native基线；无明确收益则后续不再碰superkernel。
不将编译入口失败冒充superkernel有效负收益，不将API成功冒充所有kernel已融合。
[源码、冻结方法及队列任务](results/csa_native_superkernel_20260929/README.md)。

代表档开关对照已完成：长B16 1227.095→1117.058μs（−8.967%），
短B24 1209.602→1126.791μs（−6.846%），8:2为−8.543%；八类状态开关两侧零容差一致。
profile实际14/12个SuperKernel且保持两条计算stream，后续Native对照开启，不再进行开关调参。
这些为编译调用的图外事件区间。短档独立profile设备span关闭1036.250μs、开启957.500μs，
旧入口为1030.500μs；不能将新调用均值直接拼到旧CSA设备重放表。
同一个npugraph_ex已生成图的直接replay校准现已完成，superkernel固定开启。
[结果及边界](results/csa_native_superkernel_20260929/RESULTS.md)、
[校准方法](results/csa_native_graph_replay_20260929/README.md)。

## 当前采用的单卡计时口径

task_20260929_081604_23093924514正常auto单卡1退出0，保持显式named backend、static和superkernel开启。
固定地址/shape，唯一后端生成图、无主机更新节点，从实际owner取图直接replay；不新增外层capture。
长B16：均值1122.652μs、P50 1122.490μs、P95 1128.400μs、max 1131.380μs。
短B24：均值940.762μs、P50 940.520μs、P95 945.280μs、max 945.740μs。
每侧5预热20次，独立profile仍有两条计算stream和SuperKernel；八类状态与同初态同图compiled callable零容差一致。

后续单卡Native基线采用这套入口和设备重放边界，不再做superkernel开关试探。
用户进一步明确：**CSA整体性能比较开SuperKernel；核内细节诊断关SuperKernel，保留static compile**。
核内关闭组只采独立profile，其时间不填入主性能对比表，不重新进行开关收益消融。
新短档本体时间明显低于原编译调用1126.791μs，表明该调用区间不能直接充当设备本体；
异轮正式计时与独立profile不能相减得出精确主机开销。本轮不改变整模型动态输入执行接口。
旧PTO短B24约960–980μs，不能继续依旧Native基线断言PTO领先；新版七档在阶段出口统一更新。
[本轮结果](results/csa_native_graph_replay_20260929/RESULTS.md)、
[样本、检查与profile路径](results/csa_native_graph_replay_20260929/summary.json)。
