# QLI V2四路Top-K：保留精确尺寸的UB累计根

来源ops-transformer b5b33e14：
`attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h::ProcessLD`。
它把累计Top-512留在UB，每轮合并三个新列表并将有效前缀拷回，最终才解交织、发布索引。

候选冻结在`.cache/csa-topk-ub4-a66255ea`，只改性能版decode_indexer.py；基线d1f170ff与a66255ea生产源码相同。
不叠加QR UB驻留、PV L0B或主工作树的其他改动；不改Score、cache、四路/尾块的次序与tie优先级。
本次仍按当前PTO新列表优先，未照搬最新QLI V2的所有排序优先级或metadata/SyncAll流程。

## 与旧候选不同的依据

旧二路UB根候选受切片底层存储限制，必须携带2048-float结果，额外TMOV导致核内无收益，未合入。
新实现对四/三/二路归并结果用显式tile.extract得到真正的1024-float实体块，循环只携带512对。
CPU生成代码的root phi确为`Vec<float,1,1024>`；每轮前缀TEXTRACT直接落在同一根地址，**没有额外TMOV**。
不再逐轮写回/读取pair_arena，最终从UB根解交织后直接发布；不额外读取相邻槽，也不改PTOAS/ISA/PyPTO。
这是新的生成代码依据，不把减少GM访问直接称为设备加速。

当前四路GM基线H份输入需R=ceil((H−1)/3)轮。候选省R次根读取和R次根写回，
每次4096字节；H=8/10时推导净少24KiB/query。输入列表读取、归并本身及必要前缀拷贝仍存在。
与pypto-lib相比继续保留实际cache长度的二路/多路分核，以及当前Native物理页和多leaf处理；
只移植最新AscendC的累计根驻留，不将输入规模差异忽略后直接对比耗时。

完整CSA与独立merge探针CPU lowering/PTOAS/CCE/AICPU链接通过。
短档二路kernel正文去掉生成行注释后与基线一致；这不单独证明完整层性能不变。
五组小用例复用原独立CPU整体稳定排序，新增UB返回接口及arena全部只读的检查，
覆盖2/4/6/8/10列表、随机tie/全相等/单列表占优/mask及二/三路尾部；待设备执行，不引用旧PASS代替。

[候选](candidate.patch)、[完整编译入口](compile.py)、[编译日志](compile.log)、
[单卡入口](run_layer.sh)；单卡先小用例，再8K/B16与128K/B16各20次图计时/四DFX。
mode2/atomic0/det0，真实layer4/合成历史，两档交换执行顺序；未并入生产、未做真实EP16。

单卡任务`task_20260928_145106_4822976856`已提交，当前等待设备。
任务前置小用例复用[merge_case.py](../csa_topk_fourway_20260928/merge_case.py)，
通过CSA_TOPK_PROBE_SOURCE/OUTPUT/ROOT_IN_UB指定冻结源码、结果目录和只读UB接口，旧接口与新UB接口在Python层择一，均已通过CPU探针编译，旧接口见[日志](legacy_probe_compile.log)。
编译探针的命令为run_layer.sh中同一入口加`--compile-only`，见[CPU日志](merge_compile.log)。
实际设备输出、状态、核内和完整区间结果待收，不能引用旧版本的设备通过作为本候选结论。
