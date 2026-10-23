# HCA输出数学修正与O_B分块实测

全部为同任务、同SDK的直接调用L1首尾，单位μs。比例按用户定义`control/candidate−1`。
旧分组量化与Native整token量化的输出允许改变；三组cache/state必须逐bit保持。
Native storage严格1e−5检查仍有差异，因此以下均不是完整精度或30%目标验收。

## 旧分组尾链 → Native数学尾链

| 场景 | control | candidate | 性能提升 | 输出容差 | 控制storage精确 | 窗口数 |
|---|---:|---:|---:|---|---|---:|
| 128K/B16/S6 | 515.70 | 524.54 | -1.69% | PASS | True | 7 |
| 128K/B28/S6 | 759.74 | 756.26 | +0.46% | PASS | True | 7 |
| 8K/B32/S6 | 619.64 | 601.16 | +3.07% | PASS | True | 7 |
| 256K/B16/S6 | 649.72 | 659.14 | -1.43% | PASS | True | 3 |
| 128K/B16/混长T56 | 429.92 | 432.42 | -0.58% | PASS | True | 3 |
| 128K/B1/S6 | 230.82 | 227.96 | +1.25% | PASS | True | 3 |
| 128K/B8/S6 | 355.90 | 366.40 | -2.87% | PASS | True | 3 |
| 128K/B24/S6 | 696.56 | 706.56 | -1.42% | PASS | True | 3 |

## Native数学尾链 → N256、K分两段

| 场景 | control | candidate | 性能提升 | 输出容差 | 控制storage精确 | 窗口数 |
|---|---:|---:|---:|---|---|---:|
| 128K/B16/S6 | 518.92 | 515.56 | +0.65% | PASS | True | 7 |
| 128K/B28/S6 | 783.48 | 767.30 | +2.11% | PASS | True | 7 |

## 区间策略补验：T>96使用N256、K分两段

| 场景 | control | candidate | 性能提升 | 输出容差 | 控制storage精确 | 窗口数 |
|---|---:|---:|---:|---|---|---:|
| 128K/B24/S6 | 698.22 | 699.32 | -0.16% | PASS | True | 5 |
| 8K/B32/S6 | 602.36 | 596.72 | +0.95% | PASS | True | 5 |

## 服务与记录边界

- `native_output_service_s1_v2`：接管及三次调用顺序通过；Native输出PASS。
- `native_output_service_mixed_v2`：接管及三次调用顺序通过；Native输出PASS。

任务ID、全部样本、失败记录和原始报告路径见同名JSON。
`coverage_v1`中五项提交遗漏`--batch`，在fixture构造前被拒绝；只修命令后重提v2，未掩盖失败。
服务验证覆盖真实custom op及KV钩子，不包含FFN、整网token或实际PD传输。
