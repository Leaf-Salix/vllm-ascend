# HC 消费者就地归约：实现、失败尝试及完整 CSA 消融

日期：2026-10-18。该候选借鉴独立 `hc_pre.py` 中由消费者直接归约投影部分和的思路。
改动作用于正式动态 CSA 源码；没有从静态 B16 生成物开始优化。

## 做法与正确性约束

旧路径为 AIC 产生四份 K4096 部分和，AIV 独立归约到 RAW，发布 RAW，门控和
Sinkhorn 再读取。候选增加 `hc_consumer_reduce=False` 编译策略；开启时由两个
消费者分别从 partials 读取需要的列，按 `((p0+p1)+p2)+p3` 原次序归约。
省去 RAW 写回、重读及该阶段的通知。投影 K 切分、RMS、门值、Sinkhorn 数学
及舍入点不变，不添加全核屏障。

消费者按 M8 条带分配，投影按 M16×四份 K 分配，因此消费者等待当前条带
对应的四个 AIC owner。后续 wave 的 owner 映射也按实际动态行数计算，
不固定 B16 或 T96。inv_rms 来自本 AIV 已完成的 widen 阶段。

RAW 工作区容量暂留用于同进程开关比较；开启候选时其内容必须保持填毒值 37，
并检查 `PHASE_RAW` 没有任何发布，防止名义上删除交接、实际仍消费旧内容。
根 Tensor 数仍为 130，核间信号、代际、输入布局及其它阶段不变。

## CPU 尝试

1. v1：消费者 helper 返回 UB Tile，但门控其它输入还是 GM Tensor，混合算术
   被 parser 拒绝。修正为显式 `pl.load`。
2. v2：UB Tile 仍通过 Tensor 下标赋值写出，parser 拒绝。修正为显式
   `pl.store`；post 的有效四列通过 `pl.set_validshape` 表达。
3. v3：完整符号编译通过；不传入任何样例 shape，保留根动态维度，生成一个
   MIX 提交并设置 sync_start。CPU 成功不作为设备精度或性能证据。

上述两次失败都是本仓候选的表达写法问题，没有修改 PyPTO/Simpler。
源调度依赖模型 52 个用例通过，覆盖 T=0/1/7/8/15/16/80/96/160/192/384/385/1025，
开关两路及两个执行顺序种子；模型仅检查依赖，不代替真机。
原始日志为 `cpu_v1.log`、`cpu_v2.log`、`cpu_v3.log` 和 `schedule_checks.json`。

## 真机精度与性能

正式 W8A8 第 4 层，S6，同卡同进程、共享根输入地址、各策略独立工作区。
逐候选对原动态分 task PTO 精确比较输出、Top-K、score、cache/state；eager、
图重放、信号代际、输入/工作区保护区全部通过后，采五个完整 L1 窗口。
每轮轮换执行顺序，计时包含单 SPMD 内部等待及阶段空隙，不采 Native 性能。

| 历史 / B | 原分 task PTO | 单 SPMD 旧 RAW 路径 | 消费者归约 | 相对旧 RAW 路径 |
| --- | ---: | ---: | ---: | ---: |
| 128K / 16 | 881.26 μs | 901.20 μs | 910.74 μs | 慢 1.06% |
| 8K / 32 | 995.18 μs | 1039.60 μs | 1013.20 μs | 快 2.54% |

任务分别为 `task_20261018_000855_250681529778`、
`task_20261018_000857_250737420046`。原始五次采样、所有检查、RAW 填毒及
未发布证明在相邻 `hc_consumer_h*/hc_consumer_policy_matrix/result.json`，
泳道原始 records 和 trace 也保留，汇总见本目录 `result_summary.json`。

**默认保持关闭。** 两档收益方向相反，不能据此宣布所有区间获益，也不能直接
按这两个点建立白名单分档。长历史的退步原因尚未通过阶段时序隔离；删除一个
交接会同时改变消费者 load、归约与等待的时序，不能仅数减少了几个通知便断定
整层必然更快。短历史收益可作为下一阶段候选，需结合 T、历史长度和流水竞争
分析适用区间。两档仍都慢于原分 task，距离逐档降低 30% 的目标仍有差距。

## 复现与保存

```bash
case_dir="$PWD/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
task-submit --device auto --max-time 1200 \
  "bash '$case_dir/run_device.sh' '$case_dir/hc_consumer_repro_new' --history 131072 --batch 16 --wait-backoff --hc-consumer-matrix --l1-windows 5"
```

测试使用既有编译诊断 hook，在失败轮询间插入 100 syscnt tick 间隔，旧路径与
候选一致；它尚未成为生产默认。短历史改为 `--history 8192 --batch 32`。
`before/`、`candidate/` 保留本次前后源码，测试结果只声明以上验证范围。
大二进制、权重和 `.pt` 不入 Git；路径及大小清单记录在 `local_artifacts.json`，不做 hash。
