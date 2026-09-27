# Indexer整请求S6复用：CPU候选

独立基于2a740c1f，工作树`.cache/csa-indexer-six-2a740c1f`。
继承三query候选的constexpr分组实现，但长档query数至少96时改为一组S6：
INT8 QK采用M384/N64，FP16 head规约采用K384/N64。小长档及8K仍用双query/M128/N128。
当前未运行设备、未合入。它与QKV候选、source-split cache均独立，也没有修改工具链。

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
L0A容量更紧，不能把“编译能放下”当作流水加速；设备筛查尚未取得结果。
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

各次保留metadata/保护区、有限值与Top-K结构检查。尚无S6设备结果，也没有扩大七档或修改生产版本。
