# Sparse末块发布修复后的完整CSA对照

基线4ffccb7b；候选融合末块发布，并显式抽取每16-head的mi/li统计量。
完整编译及[独立Sparse314万元素零容差](../csa_sparse_final_publish_fix_20260929/sparse_check.json)通过。
首版失败结果仍保留为失败，未将其较短计时当优化收益。

任务task_20260929_091005_151620710198已完成（exit=0），auto设备12：128K/B16和8K/B24，
CANN9.2/mode2/atomic0/det0；同卡两侧5次预热20次图事件，单独四窗DFX及完整状态零容差。
比较完整CSA/P95/max，以及Sparse加原merge的总AIV工作量和输出发布跨度。
生成代码、完整私有算子包、公共runner均保持冻结；验证通过后只移入生产性能版Sparse文件。
Native cache与精度版未改。

[来源](source.json)、[任务](task.txt)、[运行](run.sh)、[收集](collect.py)。
两档八类跨版本完整状态、各自图重放和保护区均通过；16个DFX窗口官方join/行数/block核对通过。
CSA均值1046.406→1015.430、961.513→947.139μs，长短8:2−2.667%；两档P95均下降。
Sparse加原merge的AIV总核时8:2−11.551%；保留这一真实核内收益，未将核·μs当CSA延迟。
每档20次PTO样本均无超过自身P50的105%，仅为本轮描述，不关闭历史间歇拖尾或EP16问题。
[结果](RESULTS.md)、[精简样本及检查](summary.json)。

边界task_20260929_092709_222811513859退出0，auto设备9，B3/H127，det1/atomic0/mode2：
八类跨版本状态精确通过；两侧同图active-B=3/2/1/3重放及padding/slot保护通过。
覆盖T18尾块、空闲QK核和末Sparse块跳过；[边界证据](boundary/summary.json)。
后续更新七档；整模型/EP16验收仍后置。
