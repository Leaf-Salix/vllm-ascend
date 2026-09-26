# Sparse plan 尾块修复

性能版在 B1/S6 只有 6 行时仍读写 8 行，导致内部 scratch 越界。修复显式传递尾块
有效行数，保留现有算术和流水策略。证据见 [tail_fix.json](tail_fix.json)；过程见验证日志第 129 节。

## 单卡回归

从仓库根目录提交，使用队列分配的一张卡；EPLB 关闭：

```bash
task-submit --device auto --device-num 1 --max-time 600 \
  'bash tests/pypto_test/results/csa_baseline_20260926/precision_review/run_regression.sh'
```

脚本先跑 B1/3/4 的无权重均匀 attention：Q/sink 为零，SWA=2、compressed=1，
输出与解析值要求 bit 一致。随后用固定正式第 2 层权重运行 B1/H255 全链及 A/B/A 图回放。
B1/3 覆盖 8 行 plan 的尾块，B4 覆盖整块；不依赖大张量快照。

需要单独隔离数值时，在已有单层命令加 `--save-sparse-case`，保存 Native 的真实
Q/cache/Top-K 和逆 RoPE 前输出；再用 `dsv4_csa_sparse_diagnostic.py --input .../native_sparse.pt
--variant precision|performance --output ...` 在单卡任务内回放。该模式输出零容差诊断结果，
只把无效张量作为执行失败；不能把任务退出 0 当作 Native/PTO bit 一致。

本次固定 Native 输入的 sparse max_abs 从 0.835657 降至 0.00390625，RMSE 从
0.0719584 降至 0.00008510。剩余差异和整模型验收独立记录，不用扩大容差隐藏功能错误。
