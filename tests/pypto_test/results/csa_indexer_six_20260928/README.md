# Indexer整请求S6复用：单卡收益与固定规约对照通过

独立基于2a740c1f，工作树`.cache/csa-indexer-six-2a740c1f`。
继承三query候选的constexpr分组实现，但长档query数至少96时改为一组S6：
INT8 QK采用M384/N64，FP16 head规约采用K384/N64。小长档及8K仍用双query/M128/N128。
单卡本体/核内收益及固定规约对照通过，已应用性能版；真实EP16正在验证。
它与QKV候选、source-split cache均独立，也没有修改工具链。

## 设计依据和代价

原版同一请求的Key读三遍。S6共用一遍，N64保证较大的M维仍能放入片上缓冲；
不要求物理页连续，仍从原Native cache按页加载Key及scale。
每query的causal mask、FP16输入策略、FIXPIPE缩放、Top-K树、worker数和调度标志保留。
规约K维变长，逐元素/状态是否一致仍未验证，不能按零系数推断逐bit一致。

按128K/B16、24worker、8192行leaf及实际尾leaf推导：

| 工作量 | 双query基线 | 三query候选 | S6候选 |
| --- | ---: | ---: | ---: |
| Key页次总量 | 50176 | 33280 | 16896 |
| 最忙核Key页次 | 2112 | 1568 | 800 |
| 最忙核Score-step×query | 132 | 147 | 150 |
| 最忙核QK指令次数 | 528 | 392 | 400 |

S6总页次减少66.33%，最忙核计算量增加13.64%；表中不是设备耗时或DDR流量。
若在B8也用S6，最忙核Score-step×query从66升到96，因此候选暂将门槛设为96 query。
没有把这一选择说成性能已最优；设备筛查先以长档核内收益为依据。

## CPU编译证据

Score/Top-K整体lowering、PTOAS、CCE及链接通过：[日志](compile.log)。
生成代码确认：

- L0A Query 48 KiB，地址0；系数12 KiB，地址49152。
- L0B Key 8 KiB与上一panel FP16分数48 KiB，共56 KiB。
- L0C QK 96 KiB，地址0；head规约4 KiB，地址98304。

完整CSA的lowering、PTOAS、CCE及链接也已通过：[完整编译日志](compile_full.log)。
两次编译均在模型短档正式计时前完成，不创建设备Worker、不执行NPU。
L0A容量更紧，不能把“编译能放下”当作流水加速；设备筛查结果另列下文。
score arena从240行增到288行，是为24个worker各自的两个AIV lane保留6个query的私有行。
共享的双query AIV也改为跨行加载后切片，旧版短档数字不能替代候选短档回归。

[算子补丁](candidate.patch)，[CPU编译入口](compile.py)。

```bash
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/results/csa_indexer_six_20260928/compile.py \
  --source ../.cache/csa-indexer-six-2a740c1f \
  --output tests/pypto_test/results/csa_indexer_six_20260928/compiled
```

## 原cache直接multi-ND加载的另一个方向

只读核对PTO-ISA 327cd58：`TLoadGm2L1Nd2nz`支持把Shape2映射到硬件ndNum，
可以表达四块32×128、块间4160字节的Key矩阵。A3要求矩阵stride为正且不超过65535。
但PyPTO当前3e87a843和2026-09-28以depth=1取得的main f997db72，
`FlattenTileNdTo2D`仍将自然Mat load的源窗口压到二维，不保留这种非连续三维视图。
目前不能直接替换原算子的四次gather_row；正式请求还存在倒序物理页，需要另处理逻辑顺序。
这只是能力边界，不是设备失败；没有改PTO-ISA/PTOAS，也没有将它叠加进S6候选。

## 设备筛查范围

三query已经在128K/B16测得Score AIC减少24.88%、完整CSA减少4.25%，因此继续尝试增加复用。
task_20260928_033316_2907819818：[命令](run_layer.sh)。同一个单卡作业：

1. 8K/B16原版和三query源码计时，只检查共用的短档双query路径。
2. 128K/B16重新测三query控制与S6完整CSA，仍为5预热＋20次图计时。
3. S6独立4个DFX窗口；三query核内控制复用前轮相同源码的4个窗口，明确不是同一作业。

各次保留metadata/保护区、有限值与Top-K结构检查，不扩大七档。

## 单卡结果及保留依据

task_20260928_033316_2907819818退出0。[汇总](report.json)、[汇总脚本](summarize.py)。

| 指标 μs | 原版/三query短档，8K/B16 | 三query/S6长档，128K/B16 |
| --- | ---: | ---: |
| Native控制均值 | 921.537/940.877 | 1309.385/1306.358 |
| PTO完整CSA均值 | 794.006/791.905 | 1191.899/1129.717 |
| PTO P95 | 818.980/808.280 | 1207.820/1144.020 |

S6相对同轮三query本体快5.22%，相对同轮Native快13.52%。
短档这次比较的是三query源码的共用双query分支；S6的arena分配行数240→288，
不能把该短档结果冒充S6源码的短档实测，模型短档仍单独验证。

S6独立四DFX窗口Score AIC均值286.124μs，对照前一作业三query350.587μs再降18.39%；
AIV299.148对365.664μs下降18.19%。其他未改任务的波动不算本次直接优化成果。
三query相比原双query的核内收益与本体收益见相邻目录，不能把跨作业数字称为同轮三版本实验。

完整输出对Native RMSE0.0041767351（三query本轮）/0.0041770055（S6），Top-K集合替换均为670；
零容差仍FAIL。metadata、保护区、有限值、Top-K结构通过；统计相近不是跨版本逐元素证明。

为补这一缺口，只做一个固定规约case：task_20260928_033929_295123724505退出0。
B16/H32768/S6、atomic0、det1，实际压缩长度8193，触发长档分支，覆盖8192行完整leaf及causal tail。
复用原有随物理行变化的scale fixture，原版和S6的8类输出/状态逐元素零差异，Native控制也精确一致；
候选A→B→A图重放通过。没有声称覆盖全部batch或跨batch padding图。
[固定规约结果](accuracy/comparison.json)、[运行命令](run_accuracy.sh)、[比较器](compare_accuracy.py)。

根据用户“核内有收益即保留”规则，应用性能版的S6长档分支；短档及较小长档继续双query。
三query保留测量与补丁，后续若小长档需要用它，须先测对应输入，不能沿用B16结果代替。
Native分配和算子流程、精度版算术保持原样。

## 真实EP16验证正在进行

task_20260928_034149_296992718490：[命令](run_model.sh)、[收集器](collect_model.py)。
隔离模型源码＝2a740c1f＋测试入场修复8dd737f4＋本目录候选；没有QKV边界或分离cache改动。
128K/B16、8K/B40各重新采集Native和PTO，mode2/atomic1/det0/HCCL=false/EPLB关闭，
warmup后10步无profiler forward，另3步profile。只有token/DSpark与真实forward结果可确认模型验收。
当前先完成128K/B16：Native73.033→PTO72.440ms，本轮快0.81%；
每步最慢rank均值73.577→72.930ms（快0.88%），P95 74.191→73.256ms。
65536输出token零差异，16组rank的DSpark统计一致；PTO P95/P50为1.010。
这是同轮重新采集控制后的实际EP16结果，但差额较小，尚不能宣布稳定优势或七档达标。
8K/B40仍在执行；[已完成档位和逐rank样本](model/RESULTS.md)。
