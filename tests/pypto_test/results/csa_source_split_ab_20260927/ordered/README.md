# 新页排序候选与严格批次入场

基底2a740c1f；[candidate.patch](candidate.patch)包含既有key/scale分离和四页合并读取，以及以下增量。

- 仅PTO performance的A3/C4 Indexer spec标记`prefer_contiguous_blocks`。
- 原分配器仍决定取得哪些页；仅将本次新取得的页排序后附加到请求，不移动已有历史、共享前缀，不复制设备cache。
- prefix命中、增量扩容、外部KV恢复均保留原逻辑映射；fragmentation仍可能存在，算子逐panel验证连续性后才走合并分支。
- 测试双方通过公开API暂停调度、整批入队、DP CPU barrier后恢复。level0不卸载模型、不清空cache。
- 每个正式计时step保存现有CPU position，收集器拒绝两侧位置不同的比较；不做设备复制/hash/同步。

CPU实测4项分配检查通过：[测试](test_indexer_ordered_blocks_cpu.py)、[结果](cpu_checks.json)。
测试入场的2项CPU检查通过；正式两档已完成，全部16rank的10步位置数组逐元素相同。

任务task_20260928_004008_231719416500，[命令](run_model.sh)。
顺序128K/B16 Native→PTO，再8K/B40 Native→PTO，mode2/atomic1/det0/HCCL确定性关/EPLB关。
任务退出0；CPU只读收集：`python collect_model.py`。进程退出0不代表性能或精度验收通过。

## 整模型结果：尚未胜出

| 档位 | Native forward ms | PTO forward ms | 变化 | Native/PTO P95 ms | token / DSpark |
| --- | ---: | ---: | ---: | ---: | --- |
| 128K/B16 | 72.009 | 74.262 | +3.13% | 73.052/75.205 | 0差异 / 14个rank统计不同 |
| 8K/B40 | 104.358 | 104.121 | −0.23% | 107.715/105.569 | 0差异 / 全部一致 |

229376个输出token无差异。短档0.23%不视为明显性能优势；长档性能和DSpark均未达标。
完整分布与实际配置见[RESULTS](model/RESULTS.md)、[逐rank数据](model/forward.json)。
这次统一了请求入场，与旧异步入场测试不跨表相减归因。主工作树继续保持原布局2a740c1f。

实际rank0正式计时轮，长档递增连续四页3954/4096（96.53%），短档412/640（64.38%）；
独立profile轮分别3771/4096、235/640。排序减少逆序，但不会消除已有碎片。
[页序统计](model/page_order.json)只描述bank恢复时CPU页表，不冒充硬件分支计数。

## 独立profile：专家矩阵计算是需要隔离的增量

下面为rank0三个profile step的均值，单位ms；不是正式10步forward的直接分摊。
Native实际event模式0、PTO为1，两侧profiler扰动并不完全相同。

| 档位 / 阶段 | Native | PTO | 增量 |
| --- | ---: | ---: | ---: |
| 128K/B16，21层CSA body合计 | 26.888 | 24.957 | −1.931 |
| 128K/B16，43层FFN span合计 | 33.729 | 36.816 | +3.087 |
| 128K/B16，首层FFN span | 1.905 | 4.111 | +2.206 |
| 8K/B40，21层CSA body合计 | 30.069 | 27.752 | −2.318 |
| 8K/B40，43层FFN span合计 | 56.127 | 58.323 | +2.196 |
| 8K/B40，首层FFN span | 1.627 | 1.508 | −0.119 |

长档两种专家GMM的累计任务duration合计增加3.984ms；任务可能重叠，不能直接当作critical span。
长档首层dispatch单项增加2.203ms，首层在第一个PTO CSA之前，明显包含全EP到达等待；
因此不能将FFN span增加3.087ms全归因于PTO数值，也不能靠删除首层得出新的性能成绩。
后续GMM增量与既有配对请求的活跃专家增加相符，但仍需同配置干预验证。

[CPU分析器](analyze_profile.py)从现有trace重建43层×3步，并核对FFN span总和，
输出[主图分解](model/model_gap_rank0.json)及`model/h*/ffn_breakdown_rank0.json`。
下一项仅切换PTO atomic_add，cache、Native确定性及测试配置不变；不是直接认定atomic为根因。
