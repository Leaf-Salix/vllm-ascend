# Top-K跨leaf归并保留UB累积根：独立候选

基底71153bb3，独立工作树`.cache/csa-topk-register-71153bb3`。生产与正在运行的atomic干预不改。
Native QLI在UB中维护累计Top512；当前PTO每合并一份半leaf都把根写回GM，再读回来。
候选以完整归并结果作为循环携带tile保留UB，下一次只取前512对；最终分数/索引直接从根发布。
新块优先的相等分数次序保持、leaf顺序保持、没有变更Score量化或归约策略、sync/early标志和cache布局。

每个query有H份半leaf时，省去(H−1)次根读取和(H−1)次根写回，每份4096字节，
为了匹配A3 TMOV物理形状，初始化额外读取相邻一个已分配半leaf槽位（4096字节）。
128K的典型8份净减少52KiB/query；实际存在causal尾leaf时按实际半leaf数计算。
这是搬运量推导，不是耗时或精度证明。先CPU完整编译；之后只在有空卡时单卡验证Top-K精确一致与merge核内代价，
不将其掺入atomic0的16卡对照，也不因预估收益直接合入。

## CPU算子侧编译处理

最初只携带1024-float切片，PTOAS将循环回边复制还原到2048-float底层存储，TMOV形状不符；
显式tile.assemble仍被降为同一无效TMOV。没有修改PTOAS/ISA或PyPTO来绕过检查。
改为携带2048-float完整结果，输入/输出物理形状匹配；每轮mrgsort仍只消费前1024个float。
第一次从相邻两个半leaf槽位读取2048个float，但只使用首槽；后一个槽只用于合法物理形状，
每个query按原布局本就至少分配两个槽，不跨query或扩大分配。

flatten只在orchestration建立同一pair arena的连续视图，incore保持显式tensor load；
没有新设备分配或复制、不更改Native KV cache。Indexer CPU lowering/PTOAS/CCE/链接已通过，
完整CSA集成也已通过CPU lowering、PTOAS、CCE及链接。设备精确比较与核内收益均待验证。

[候选补丁](candidate.patch)、[完整编译日志](compile_full_flat.log)、[单卡命令](run_layer.sh)。
task_20260928_051013_399892317506已排队，在正在运行的EP16干预之后用一张卡执行。
只128K/B8：双方atomic0/det1、第二层权重、物理行可变scale；保存8类状态并精确比较，候选A→B→A，
各20次独立图计时；之后单独各4个DFX窗口衡量merge核内，未与计时混采。

## 单卡结束：状态精确一致，本体改善，但merge核内收益未证明

task_20260928_051013_399892317506退出0。[8类输出/状态逐元素与图重放](accuracy/comparison.json)全部通过，
包括idx_topk、x_out和两类cache/state；Native控制也精确一致。此比较没有保存未用于后续计算的idx_topk_scores。
Native浮点零容差仍FAIL，未放宽阈值；本次不是Native数值全对齐验收。

单卡完整CSA872.664→856.161μs（−1.89%），Native控制1015.156→1016.520（+0.13%），
PTO P95 882.900→878.340、max897.980→897.200。中位数871.620→853.320μs。
但独立4窗口merge block均值11.814→12.068μs（+2.15%），两侧窗口范围重叠；
新增的2048-float UB TMOV仍有成本，不能把减少GM字节数当成核内加速证明。
未改的merge_norm窗口反而缩短约15%，可能涉及调度/内存分配连带变化，不能算本项的直接核内优化。
[全部计时和核内窗口](report.json)。

暂不合入、不扩模型测试，保留补丁和本体收益证据。新的七档EP16只验证71153bb3＋atomic0，
避免将未定位的本体差额与已经有效的固定规约方向混合。
