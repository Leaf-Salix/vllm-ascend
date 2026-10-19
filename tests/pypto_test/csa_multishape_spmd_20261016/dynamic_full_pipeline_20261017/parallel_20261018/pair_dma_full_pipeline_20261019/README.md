# 物理行 plan 上的双行 DMA 完整 CSA 候选

从 MGATHER 已测候选继续。现有 A3 MGATHER 的 row 模式仍逐行发 DMA；
本轮尝试对原 Top-K 次序的相邻两项按地址升序建立动态 pitch 描述符，
一次搬入两行，再在 UB 恢复原序。索引仍一次载入64项，所有负哨兵、
容量与完整 span/gap 检查保留；不满足时逐行采集。SWA与能力外不变。

实现先作用于正式动态 attention_stream_stages.py，然后冻结完整37文件，
正式目录恢复原已提交版本，供独立任务验证。没有改B/T/历史分档或metadata，
没有增加Host回读；所有数据相关地址在device生成。数学、Cube与事件协议
不变，SOURCE_AUDIT.json记录源级核对，不把它当作设备数值证明。

本候选依赖相邻 gm_subview_pair_dma_20261019 的独立 PyPTO gm_view 原型。
原型52项CPU单测、26组真实地址图检查通过；默认不开能力时，完整候选
AIC/AIV C++ 与旧PyPTO仅去整行注释后逐字相同。完整设备三方均使用同一
新PyPTO和PTOAS v3，不混用工具链，不修改共享安装。

成本假设：减少小DMA可能有益，但成对地址判断、排序和UB复制也可能抵消。
只有完整真机CSA L1五窗与冻结精度/状态/保护/图门槛通过后决定保留。
当前CPU准备，不预记真机通过或收益。大构建与轨迹本地保留，只记录路径/大小。

## 完整 CPU 首轮与独立最小复现

cpu_v1 在 codegen 报 pair_pitched Tensor view not found。逐 pass 追踪发现
首个 FREE_VAR 出现在 FlattenTileNdTo2D：非tile操作保留旧结果类型，
但类型中的动态 shape/stride Scalar 已被重写；消费者单独克隆变量。
最小 flatten_gm_view_repro.py 仅用已有 tensor.view，无新 gm_view 参数，
在原父工具链同样复现：before无FREE_VAR，after出现FREE_VAR。
当前上游main源码也保留同一分支；只读代码检查，不称为干净main构建复现。

本地新工作区将GM view/slice纳入已有remote_load类型+定义同步重写路径，
不修改父工具链；建议补丁flatten_view_only.patch，保留首轮失败及最小IR。
局部测试及完整CPU复编译待收齐，不声明真机或性能通过。

修正后局部4项CPU回归通过，完整cpu_v2通过58动态轴/单MIX。已提交
PyPTO #3046（含最小原接口复现、main只读核对、建议补丁；无PR/标签）。
首项128K/B16三方五窗设备任务见DEVICE_SUBMISSIONS.json，待终态；
该任务另含metadata/padding图检查，不用CPU成功预记设备通过。

128K/B16任务task_20261019_094452_378533414004 completed/exit0，全冻结
精确/状态/保护/图/metadata/padding/代际通过。同轮五窗中位多task860.74、
参考MGATHER803.96、双行+UB重排933.56μs，回退16.120%，不采用、不扩
短历史或七档。CPU短策略成功不代替性能。相邻pair_dma_direct_rows候选
只合并原序递增行并直写最终Tile，用实际生成物中少掉的MTE2→V/TMOV
提出消融；不能把它的预期当设备结果。所有首轮失败及修正继续保留。
