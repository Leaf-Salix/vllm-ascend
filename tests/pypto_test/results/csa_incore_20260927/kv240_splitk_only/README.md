# 性能版 B40 KV 投影统一宽 tile / split-K

已保留。性能版删除 `T=240` 专用的 Native 精度遍历，B40/S6 与其他输入共用 N128/K256
及部署配置的 split-K=8；不改变总矩阵工作量。精度版保留原 Native K 顺序，
性能版 atomic=0 时仍以单分片提供固定顺序诊断。Indexer/Sparse Attention 均为 V10。

旧特例 N32/K64、16个block，每核遍历完整K4096，并按列组重排K256块。
新路径32个block，每核只处理K512；减小每核K工作量并提高tile利用率。
这不是只将相同耗时拆成更多任务：累计核内工作量也明显下降。
PTO WKV仍使用ND，不宣称已经复用Native的NZ权重；该布局差异另行处理。

## 单卡实测

正式layer4权重＋合成历史，8K/B40/S6/TP1/mode2/atomic1/确定性0，无EPLB。
预热5次/20次无profiler采样，另4个DFX图重放窗口；复用第二CSA层metadata。

| 实现 | KV投影block数 | block核内均值范围 μs | 累计核内工作量范围 核·μs | CSA本体均值 μs | 本体p50 / p95 μs | 同轮Native μs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V10 | 16 | 90.05–103.11 | 1440.86–1649.72 | 1428.20 | 1434.40 / 1462.48 | 1405.57 |
| 仅KV投影改动 | 32 | 11.32–12.00 | 362.22–384.04 | 1386.00 | 1378.45 / 1431.02 | 1410.61 |

本体相对V10下降2.95%，比同轮Native低1.74%。KV投影Worker跨度从97.54–140.16 μs
变为40.88–111.90 μs，仍受任务分派/交叠影响，不能把约8倍的block均值差解释为整段8倍加速。
完整PTO含拆分/写回为1736.41 μs，仍慢于Native，未达最终目标。

最初在合并WS的Indexer基底上试得本体1388.52 μs，
[该轮统计](../kv240_splitk/report.json)单独保存；不混入上述最终实现数据。
随后撤回WS候选，任务`task_20260927_150957_151486024668`对仅KV改动验证，退出0。
首轮及Native补采任务为`task_20260927_150612_14900281244`，退出0。

## 功能与误差

metadata/slot保护区及Top-K结构通过，非有限值0，Top-K集合替换仍901。
输出max_abs=0.03125，RMSE=0.003292748；V10 RMSE=0.003292742。
SWA cache max_abs保持0.015625，RMSE从0.000168518变为0.000168484。
其余浮点状态误差统计未扩大，逐项数据保存在[完整统计](report.json)。
零容差仍FAIL，不能据误差统计相近声称逐bit或整模型验收完成。
尚未做这一阶段的七档统一源码出口测量和16卡token/DSpark验收。

## 复现与证据

- [最终补丁](candidate.patch)，变更仅在性能版qkv_proj_rope.py。
- [计时、逐状态误差、四窗口核内统计及原始路径](report.json)。
- 原始结果：`../../csa_split_optimization_20260927/kv240_splitk_only/h8192_b40/`。
- 用既有`csa_split_optimization_20260927/run_case.sh kv240_splitk_only 8192 40 timing|swimlane`。
- CPU kernel ABI、PTOAS、AICPU C++编译先通过；无hash扫描或无关回归。
