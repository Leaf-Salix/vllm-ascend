# 主Q_B Left双槽：完整精度通过，性能回退，不采用

只测128K/B16/S6一例，任务`task_20261018_024836_184487321209`，exit0。
正式W8A8；同进程原task、d56、候选交替五窗，计完整L1 incore首尾。
参考源包含compact前移，没有叠加HC通知guard；候选为`source_candidate_v2/`。

| 实现 | 五窗μs | 中位数μs |
| --- | --- | ---: |
| 原动态分task | 853.72 / 851.30 / 858.64 / 864.04 / 861.00 | 858.64 |
| d56 SPMD | 869.26 / 870.98 / 867.50 / 873.20 / 875.00 | 870.98 |
| Q_B Left双槽v2 | 882.96 / 876.70 / 871.34 / 875.26 / 880.42 | 876.70 |

候选比同轮d56慢5.72μs、0.657%，五个配对均更慢。不采用、不扩其它档位，
不与已有效的通知候选组合。不能将CPU两槽和合理事件位置等同实际收益，也
不能凭本轮完整L1把回退单独归因于Mat、Left或某条新事件。

原完整数值、Top-K、score、cache/state、A→B→A、图、历史/页表重放、
保护区及72核代际通过，参照是原PTO，未验收Native/token。v1只完成CPU，
因L1退化未上卡；v2保留Mat窗口后才执行本次唯一设备任务。原始结果在
`../qb_left_doublebuffer_h131072_b16_v2/result.json`及`source_matrix/result.json`，
五窗与配对差值也保存于`DEVICE_RESULTS.json`；复现命令在`device_registry.json`。

所有源版本、CPU失败预期、生成事件/地址摘录完整保留。大编译快照仅登记
原始路径/大小，见`ARCHIVE_MANIFEST.json`，不计算hash。
