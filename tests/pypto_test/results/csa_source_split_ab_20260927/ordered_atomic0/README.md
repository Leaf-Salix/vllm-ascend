# EP16规约干预：仅关闭PTO atomic_add

复用已冻结`.cache/csa-source-ordered-2a740c1f`，与前轮相同分离布局、四页合并读取和新页排序。
仅将`VLLM_ASCEND_PTO_CSA_ATOMIC_ADD`从1改为0；仍用性能版，其固定K路径不等于完整Native算术。
Native确定性保持0、HCCL_DETERMINISTIC=false、mode2、EPLB关，双方完整批次入场。

先完成[单卡副本诊断](../../csa_duplicate_requests_20260928/README.md)，随后提交
task_20260928_010335_25771591179：[执行命令](run_model.sh)。128K/B16、8K/B40各10个正式forward及独立3步profile。
Native控制复用[ordered任务](../ordered/README.md)的实际结果；`model/h*/b*/native`明确链接至那一轮。
Native不使用PTO专用atomic开关，原始配置仍记录1，不改写为0。
[收集器](collect_model.py)只允许这一声明过的配置差别，其他门禁保持，逐rank核对10步请求位置。

任务退出0；32组rank的10步位置配对通过，229376输出token无差异。

| 档位 | Native ms | PTO atomic1 ms | PTO atomic0 ms | atomic0对Native | PTO atomic1/0 P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 72.009 | 74.262 | 73.701 | +2.35% | 75.205/75.243 |
| 8K/B40 | 104.358 | 104.121 | 104.072 | −0.27% | 105.569/110.292 |

长档12rank DSpark统计不同，短档全部一致。长档仍慢于Native，短档基本持平且P95变差，暂不采用关闭atomic配置。
短档最后一步16rank的forward均值109.910ms、最大110.417ms；全部10步保留，不删除该步制造收益。
两个PTO版本正式轮及profile轮的rank0页序统计完全相同，新增干预确实保持cache策略和输入位置。

独立rank0三步profile中，两种专家GMM累计任务duration从atomic1到0：长档10.065→8.718ms，短档13.455→12.610ms。
说明关闭atomic值得纳入算术策略评估；但busy可以重叠，profile首层等待差异较大，不能据此宣布正式forward已胜出。
长档CSA body合计24.957→24.570ms，短档27.752→27.773ms。
不能根据单卡副本无差异排除真实模型影响，也不能将这些观测外推成atomic是唯一原因。

[正式10步与P95](model/RESULTS.md)、[逐rank原始样本](model/forward.json)、
[页序](model/page_order.json)、[独立主图分解](model/model_gap_rank0.json)。
CPU复现分解：`python ../ordered/analyze_profile.py --root model`。
