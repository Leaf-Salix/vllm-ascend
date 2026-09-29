# Sparse末块发布修复后的完整CSA对照

基线4ffccb7b；候选融合末块发布，并显式抽取每16-head的mi/li统计量。
完整编译及[独立Sparse314万元素零容差](../csa_sparse_final_publish_fix_20260929/sparse_check.json)通过。
首版失败结果仍保留为失败，未将其较短计时当优化收益。

任务task_20260929_091005_151620710198正常auto单卡排队：128K/B16和8K/B24，
CANN9.2/mode2/atomic0/det0；同卡两侧5次预热20次图事件，单独四窗DFX及完整状态零容差。
比较完整CSA/P95/max，以及Sparse加原merge的总AIV工作量和输出发布跨度。
生成代码、完整私有算子包、公共runner均已冻结；未改变生产算子、Native cache或精度版。

[来源](source.json)、[任务](task.txt)、[运行](run.sh)、[收集](collect.py)。
边界B3/H127及padding脚本已准备，完整状态通过后才提交，不扩七档和EP16。
