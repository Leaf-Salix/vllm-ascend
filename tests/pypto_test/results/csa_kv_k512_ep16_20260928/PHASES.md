# EP16入场分项

同一正式10步的可选主机分项，不从设备forward中扣减；不是kernel计时。

准备段含已有设备同步/忙等，墙钟与线程CPU之差不能单独证明OS抢占。gap是两个既有标记之间的范围，不能自动归因为单一函数。

下面仅展开设备相对入场异常>2ms的rank；所有正式样本仍用于性能统计。

| 档位/侧/step/rank | 设备迟到ms | 主机段 | 当步墙钟/线程CPU ms | 同rank十步墙钟中位ms |
| --- | ---: | --- | ---: | ---: |
| 128K/B4/native/10/14 | 2.844 | input_sync | 0.033/0.019 | 0.039 |
| 128K/B4/native/10/14 | 2.844 | state_update_gap | 0.200/0.200 | 0.232 |
| 128K/B4/native/10/14 | 2.844 | inputs | 24.862/24.762 | 23.468 |
| 128K/B4/native/10/14 | 2.844 | batch_coordination | 4.259/0.315 | 3.820 |
| 128K/B4/native/10/14 | 2.844 | attention_metadata | 3.931/3.911 | 3.926 |
| 128K/B4/native/10/14 | 2.844 | preprocess | 0.050/0.049 | 0.042 |
| 128K/B4/native/10/14 | 2.844 | forward_context_gap | 0.264/0.265 | 0.236 |
| 128K/B4/native/10/14 | 2.844 | observer_prepare | 0.021/0.021 | 0.014 |
| 128K/B4/native/10/14 | 2.844 | event_and_submit | 0.269/0.269 | 0.254 |

全部rank原始标记与分项：[phases.json](phases.json)。正式统计：[model/RESULTS.md](model/RESULTS.md)。
未复现不能证明已修复；本轮同时验证KV候选，不能把跨轮变化单独归因于事件预热。
