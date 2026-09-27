# 新页排序候选与严格批次入场

基底2a740c1f；[candidate.patch](candidate.patch)包含既有key/scale分离和四页合并读取，以及以下增量。

- 仅PTO performance的A3/C4 Indexer spec标记`prefer_contiguous_blocks`。
- 原分配器仍决定取得哪些页；仅将本次新取得的页排序后附加到请求，不移动已有历史、共享前缀，不复制设备cache。
- prefix命中、增量扩容、外部KV恢复均保留原逻辑映射；fragmentation仍可能存在，算子逐panel验证连续性后才走合并分支。
- 测试双方通过公开API暂停调度、整批入队、DP CPU barrier后恢复。level0不卸载模型、不清空cache。
- 每个正式计时step保存现有CPU position，收集器拒绝两侧位置不同的比较；不做设备复制/hash/同步。

CPU实测4项分配检查通过：[测试](test_indexer_ordered_blocks_cpu.py)、[结果](cpu_checks.json)。
测试入场的2项CPU检查通过；新Native长档真实16卡窗口已完成，PTO及短档仍待当前任务。

任务task_20260928_004008_231719416500，[命令](run_model.sh)。
顺序128K/B16 Native→PTO，再8K/B40 Native→PTO，mode2/atomic1/det0/HCCL确定性关/EPLB关。
CPU只读收集：`python collect_model.py --available`。未完成前不宣称性能或精度达标。
