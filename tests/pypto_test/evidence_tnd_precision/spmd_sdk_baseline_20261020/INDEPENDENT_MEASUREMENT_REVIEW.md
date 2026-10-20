# 8K/B16效率基线与DFX拆分：独立测量审查

2026-10-20。只读原始JSON、CANN trace、DFX／依赖图归档及冻结源码；仅运行本机CPU重算。
未修改实现、SDK或已执行助手，未提交硬件任务。独立结果见同目录 `INDEPENDENT_MEASUREMENT_REVIEW.json`。

## 结论

本轮数据的映射与数值记录可用：无DFX计时、单独CANN profile、单独L1、单独dep-gen四种口径分开，没有将dep-gen混入L1时戳。
当前正式测得的Native是 **SK0**；不能把它称为用户截图中的SuperKernel优化Native。独立Native-only SK1任务尚不在本报告验收范围。

| 测量 | 重算结果 | 范围 |
| --- | ---: | --- |
| 无DFX Native SK0 | 中位940.826492 μs；均值938.525117 μs | 12窗×100次完整HC-pre→HC-post回放 |
| 无DFX冻结Overlap | 中位1333.726807 μs；均值1333.849208 μs | 同任务配对协议，没有新候选数学 |
| L1完整incore跨度 | 中位1244.14 μs | 单独3窗×1次采集，包含内部等待 |
| QK/PV AIV函数跨度 | 中位255.72 μs | 混合任务AIV侧，不能当纯Vector计算 |
| QK/PV AIC函数跨度 | 中位253.90 μs | 同一混合任务AIC侧，不能与AIV相加 |
| O-proj分组inclusive跨度 | 中位256.36 μs | A、BF16边界、量化、B和反量化等多个task间的完整跨度 |
| 前缀AIV函数跨度 | 中位177.22 μs | 单SPMD前缀，包含handoff及等待 |

用户提供的nalinaly动态SPMD 644.08 μs是待受控复现的8K/B16目标，不能与本次SK0结果拼成已复现的三臂比较。
本轮同形状测量仍不能代替真实8K/B40和128K/B24 token协议的最终回归。

## 精度与冻结身份

逐项检查SK0、DFX、dep三个结果的具体门禁记录，而不是仅读取pass：

- B16、每请求S6、真实／物理T96、L4、history8191，无逐请求padding。
- Native与PTO在A、B输入下各八个外部边界bit_equal且different_elements为0；恢复A的实际graph回放、TopK毒化恢复、compiled/raw一致及guards均通过。
- 实际FULL graph、Native静态编译、DSA多流开关开启；两臂 `super_kernel_optimize=False`。PTO资格通过且未回退。
- 三次任务的Native A隐藏输入、positions、初始cache以及八边界结果SHA完全相同；编译收据SHA及所有父源身份相同。
- 六份Overlap源SHA与上轮冻结清单及当前正式checkout字节完全相同。三个助手及各自manifest的实际SHA也匹配结果记录。
- 只读packet回放后门禁通过。本轮没有重跑跨bucket；旧六源的历史跨bucket验收不能重标为本轮又完成一次。

归档保存的是原始时戳、源／输入身份和数值比较记录，没有保存全部大tensor；本次CPU核验没有重新执行未保存tensor的逐byte比较。

## 每条时戳的task与角色映射

依赖图58个task ID无重复。L1归档有两份完全一致的**65函数kernel map**，另两份单函数map属于加载期权重预整理，未误当完整CSA map。
65是函数数目，不是核数；实际时戳metadata明确24个AIC和48个AIV，共72核。

对3个窗口逐条解析，共每窗1219条记录：AIC353条、AIV866条；每窗均覆盖全部72核，使用53个实际task token。
所有token都存在于单独dep图，所有fid都存在于完整map，并且记录核角色与map的core_type相同，没有遗漏或按task ID直接当func ID。

- AIC取 `deps.tasks[token].kernel_ids[0]`。
- AIV取其AIV槽；本组全部3个MIX task的两AIV槽相同，分别为 `[0,1,1]`、`[43,44,44]`、`[52,53,53]`，因此选择非负AIV槽没有丢失第二角色。
- 纯AIV task另一个槽为-1；所有实际记录都有唯一有效AIV函数身份。
- 检查50MHz时钟、单run边界、无丢失边界、时间区间合法；每函数的记录数、起止跨度和最长单核task均与现有分析逐项一致。

`analyze_dfx.py` 当前映射方法对本组证据成立。如果未来出现不同的两个AIV槽，它使用“第一个非负槽”的写法就不足以区分两个subblock，必须增加明确映射及断言，不能泛化当前假设。

## 分离采集与重算

实际助手与结果一致：

- L1：`enable_chip_swimlane=1`、`enable_dep_gen=False`，3个独立begin/end窗口，每窗一次raw replay，没有性能计时循环。
- 依赖图：`enable_chip_swimlane=0`、`enable_dep_gen=True`，单窗口一次raw replay，只作task图映射，没有用于耗时。
- 无DFX：另一个任务的12×100事件计时；CANN profile又在计时后单独采3次回放，不能把profile跨度代替12窗数值。

逐窗重新以最早task start到最晚task end计算incore跨度，并对所有记录区间作全局并集：

| L1窗口 | 完整incore跨度 μs | 无incore记录活跃的并集空隙 μs |
| --- | ---: | ---: |
| 初始窗 | 1229.46 | 105.18 |
| window_1 | 1244.14 | 111.50 |
| window_2 | 1248.26 | 110.12 |

中位数1244.14／110.12均匹配现有分析。另独立使用Decimal计算CANN trace的设备工作跨度和各窗口stream集合，与 `CANN_TRACE_ANALYSIS.json` 一致。

## 必须保留的解释边界

1. **阶段跨度不能求和。** Sparse attention分组包含很早调度的 `rope_cs/kv_touch`，到稍后QK/PV及merge才结束；其755.16 μs inclusive跨度不代表755.16 μs attention计算。
   同理Indexer Score/TopK分组363.74 μs不能当作纯打分排序耗时；实际score混合函数跨度约50 μs。
2. QK/PV的253.90／255.72 μs是相互重叠的AIC/AIV任务生命周期，含内部同步／等待，不可相加为509.62 μs或分别等同Cube／Vector执行时间。
3. O-proj的256.36 μs包含多个阶段、分批调度和任务间空隙，不是单核matmul耗时。
4. 110.12 μs只是“所有已记录incore区间的并集之外”时间，不能直接命名为纯调度／AICPU开销，也不能当作可全部消除的等待。
5. 无DFX完整计时与L1跨度来自不同任务及测量协议，其差值不是受控测得的框架开销。
6. `DFX_FUNCTION_MEDIANS.json` 的start、end、span分别取中位数，不构成某个真实窗口。画关键路径时应选取实际窗口，不能把独立中位数拼成一条时序。
7. DFX／dep的 `cross_bucket_scope` 沿用了“rechecks SK1 comparator”文本，这是遗留说明标签错误。实际options和计时单位均为SK0；原文件应保留，在新索引／报告纠正，不把标签当数值失败或SK1成功证据。

本轮可用来选择下一项测量，但不能仅凭这些inclusive跨度判定纯计算瓶颈已经锁定。下一优化仍须单因素、Native逐bit门禁及真实两档token回归。

## 独立核验文件

- `INDEPENDENT_MEASUREMENT_REVIEW.json`：重算结果、源身份、归档／trace哈希及范围限制。
- `independent-review/recheck_measurement.py`：只读CPU复算助手，不导入NPU、不改原分析文件。

没有发现原始记录映射或现有数值汇总的阻断性错误；上述标签与解释范围必须在最终结论中保留。
