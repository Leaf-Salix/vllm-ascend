# 当前动态CSA重验G6内部双M192/N128

从正式289cc072出发，先在ops/pypto的动态源码接线，再冻结source_candidate，
恢复正式已采用源码。不是继续优化冻结B16/S6 entry，也不是把六query改成
独立G3分组。所有device分组、leaf范围、query尾组补齐、两AIV Top-K以及
双槽READY/ACK次数不变，完整CSA仍只有一个MIX task。

候选每个N128 panel内，两个M192 INT8 QK共用Key Right，分别进行K192的
FP16系数归约。保持原INT32→FIX缩放/ReLU→FP16→FP32归约的舍入边界。
系数只加载本半组三条有效行/192列，源仍是原16×384矩阵，没有新增GM
系数布局或Host转换。数学等价不是精确验证，必须通过原冻结检查。

本候选有历史依据也有旧负例：tests/pypto_test/csa_multishape_spmd_20261016/
indexer_n128.py曾在旧静态框架测试，883μs不能直接与当前约815μs比较。
这次独立重新编写动态helper，使用已经修正INT8 L1子范围的PTOAS v3；
同时把旧候选“先WS STORE、后当前QK FIX”改为当前QK FIX先提交再写WS。
是否改善只能由同轮完整L1证明，不能从扩大N或少一次Key提取推断收益。

资源合同：两Query Left合48KiB，两系数Left合12KiB；Key Right16KiB加
WS Right48KiB，不允许第二Key也驻Right，因此候选明确TO_L0=False、预取
只到L1双槽。QK Acc96KiB加WS Acc8KiB。实际规划/CCEC仍须验证这些约束。

新开关score_split_query默认关闭，仅已有balanced G6、N128、score_tile1024、
prefetch1/L1组合进入。测试通过policy覆盖开启；尚未建立默认采用区间。
T/B/Q/L/position、页表、cache长度和循环次数均动态，无七档测试点白名单。

CPU用run_cpu.sh；设备由task-submit --device auto调用run_device.sh，并带
--batch/--history/--l1-windows 5。代表长B16先保留metadata/padding图检查，
精确/状态/保护/协议/图通过后才解释性能。三方使用同一个NEWP/PTOAS v3，
不依赖上轮GM子视图或PTO-ISA私有屏障。结果待实际任务完成。

完整CPU首轮通过58动态轴、单MIX和五个原权重CachePolicy绑定，未供样例shape。
符号入口使用QUERY_GROUP=6、QK_PANEL=128、TO_L0=False和显式SCORE_SPLIT_QUERY，
不是把开关存在当作实际分块改变；后续核对生成CodeGen的M192/WS K192和存储顺序。

## 完整结果

B16任务task_20261019_114829_82865120216 completed/exit0；A/B/A完整输出、
Top-K/score、cache/state、保护区、基础图及metadata/padding图、代际全部通过。
同轮五窗原多task861.02μs、当前参考803.00μs、双M192/N128候选827.70μs，
候选耗时增加3.07597%，不采用、不扩七档。不能拿旧静态883与本次827.70
相减作为收益，因为上下文、工具链和前后段都已变化。

实际PTO确认两Query Left在0/24576、两系数Left在49152/55296；WS的有效M
为3但物理Tile16，QK Acc96KiB、WS Acc8KiB。Right首个Key在0，其余七个
panel Key在49152；WS Right均在0。只有首panel Key/WS发生地址复用，
不能把一处重用泛化为八个panel全部缺少双缓冲。当前没有因此重提小调整。
原非N128叶数学和完整AIV函数AST与正式源相同，源码/CPU检查不扩大成其它
资源区间已完成真机验证。详见CODEGEN_AUDIT.json与DEVICE_RESULT.json。
