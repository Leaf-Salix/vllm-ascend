# EP16规约干预：仅关闭PTO atomic_add

复用已冻结`.cache/csa-source-ordered-2a740c1f`，与前轮相同分离布局、四页合并读取和新页排序。
仅将`VLLM_ASCEND_PTO_CSA_ATOMIC_ADD`从1改为0；仍用性能版，其固定K路径不等于完整Native算术。
Native确定性保持0、HCCL_DETERMINISTIC=false、mode2、EPLB关，双方完整批次入场。

先完成[单卡副本诊断](../../csa_duplicate_requests_20260928/README.md)，随后提交
task_20260928_010335_25771591179：[执行命令](run_model.sh)。128K/B16、8K/B40各10个正式forward及独立3步profile。
Native控制复用[ordered任务](../ordered/README.md)的实际结果；`model/h*/b*/native`明确链接至那一轮。
Native不使用PTO专用atomic开关，原始配置仍记录1，不改写为0。
[收集器](collect_model.py)只允许这一声明过的配置差别，其他门禁保持，逐rank核对10步请求位置。

当前待设备结果；不能据单卡副本没有差异提前排除或确定真实模型的atomic影响。
