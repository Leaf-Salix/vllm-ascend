# 独立前缀提前打包候选

仅将AIV input packing从H2后提前到H1/H2之间，与AIC平方和重叠。数学、舍入及10次handoff保持原样；无SDK或生成C++改写。

两轮8K/B32/T192、真实第4层权重、合成hidden/history8191，完整FULL graph：

| 捕获顺序 | 冻结Event μs | 提前pack μs | 降低 | Native A/B八边界与ABA |
| --- | ---: | ---: | ---: | --- |
| Native→冻结→提前pack | 1884.93 | 1835.62 | 2.616% | 全部逐位通过 |
| Native→提前pack→冻结 | 1869.62 | 1848.54 | 1.127% | 全部逐位通过 |

两次均12/12成对领先。只能认定单层跨捕获顺序有正收益，不能称恒定2.6%。未做128K、短T/混长、整模型token或与输出融合组合；默认实现保留。

原始result包含12窗×100回放样本、5窗L1和精度门禁；执行脚本及两数学文件原样保存在archive，校验表记录每文件SHA。复现前先用新隔离目录解包，参考外层README准备同SDK/源码资产，分别无设备compile.py early生成收据，再task-submit --device auto --device-num 1 --max-time 780提交run.sh（内部750秒）。具体命令参数以归档run.sh为准，不移除身份门禁。
