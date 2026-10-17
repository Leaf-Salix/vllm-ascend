# 大T融合机器码体积：现成产物只读审计

**结论：确认了长B16的新OA路径增加代码体积，但没有足够新证据把大T耗时归因于指令缓存、Scalar spill或可删除的控制流；本轮不提出算子修改。** 未编译、上卡、改正式源或依赖。全部数字来自已完成作业，未以源码行数替代机器码大小。

主PAR：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018`。

## 实际kernel image大小

角色由各作业`source_matrix/result.json.kernel_configs`读取，未按目录排序推断。两个reference源目录各33份Python源码直接文本比较，均与正式`66882634`相同。

| 规格 / 角色 | build_output子目录 | AIC字节 | AIV字节 |
| --- | --- | ---: | ---: |
| H131072/B16/S6，正式668 | oa_pair_manual_h131072_b16_v1 / decode_csa_spmd_jp7o4g2r | 50200 | 145680 |
| 同轮OA候选 | oa_pair_manual_h131072_b16_v1 / decode_csa_spmd_3u8uriiq | 53056 | 147200 |
| H8192/B32/S6，正式668 | qb_b_resident_h8192_b32_v1 / decode_csa_spmd_ll64q409 | 47884 | 148932 |

每个build下的`prebuilt/kernel_0.bin`是cube_schedule，`kernel_1.bin`是vector_schedule，映射由binary_manifest的name/func_id确认；未计算或保存校验摘要。Simpler现有`scene_test.py:1675–1684`和`elf_parser.py:12–24`确认设备缓存的是链接后完整`.text`字节，包含entry及可能的helper；这些不是ELF文件，不能据其长度拆出每个业务阶段的函数大小。

长B16 OA候选AIC增加2856 B（5.689%）、AIV增加1520 B（1.043%），合计增加4376 B。同轮既有L1中位数正式884.84→候选889.28 μs；它证明该档没有整体收益，**不能证明这4376 B导致回退**。两侧同G6/64/1024、OA128/OB96、QA=false与100tick wait-backoff，无phase clock插桩。

短B32的现成设备机器码只有正式668参照；没有找到新OA候选在G2/OA256的同规格设备image，不能补写为机器码相同。该档T192选OA256，新pair分支关闭；已有`oa_pair_manual_evidence/closed_branch_review/README.md`给出源级可达AST相同及G6/OA256唯一CPU的PTO/分配/局部事件等价。这不是G2/OA256机器码对照，本轮不为填空追加编译。

## 融合与分task不能只比较最大文件

上述两个作业的原分task均保留25个AIC、48个AIV image：AIC合计77920 B、最大9408 B；AIV合计231488 B、最大42532 B。融合后的单image较大，但合计代码反而更少。不同task由不同owner、循环和请求条件执行，保留image总量不是一核本次执行的指令工作集；单image最大值也不是I-cache miss或执行次数。

两作业各73个原分task image、3份融合build共6个image的完整路径/大小，以及策略、角色和既有L1样本，保存在`evidence.json`。没有将只占一个阶段的Attention image与完整CSA相除后用作瓶颈证据。

## 能确认的控制流变化

长B16候选实际PTO保留`0<T<=96`判断及pair/旧OA回退两条路径：候选`ptoas/dynamic_csa_full.pto:3848–3852`是AIC条件，`:13802–13804`是AIV条件；正式旧OA循环从`:3848`开始。这是生成物事实，并非只看高层源码猜测。

| 完整PTO静态位置数：正式→OA候选 | AIC | AIV |
| --- | ---: | ---: |
| scf.if | 167→173 | 389→390 |
| scf.for | 70→73 | 124→126 |
| arith.cmpi | 254→266 | 593→595 |

这些是生成IR中互斥路径也一并计入的位置数，**不是机器指令数或本次执行次数**。额外pair/single循环与五owner等待已有OA设计覆盖，不能把全部静态增加当作每拍重复执行，也不能由这些数字定位4376 B的具体机器码归属。本轮没有识别出尚未试过、且有周期证据支持删除的热控制流。

## 排重与证据边界

- `scalar_resource_audit/README.md`已经核对旧d253现成产物：没有寄存器、stack/spill或local scratch统计。此次所用6个融合image仍未保留`.o/.elf/.s/.ll/.su`，没有新的符号表、逐阶段机器码边界或资源报告，未知不填0。
- `attention_readonly_audit/README.md`及MASTER:16360已经区分整层体积和Attention局部数学/控制流。本轮不重开query metadata、physical plan、重复guard等已完成路线。
- 历史`tests/pypto_test/csa_multishape_spmd_20261016/OUTLINED_CACHED_WAIT_EXPERIMENT.md:44`：74个noinline helper与现选等待组合完整PASS，807.02 μs与当时现选约807持平，未采用。它是旧静态现选诊断，不等价于当前动态668，但足以排除“拆小函数必然更快”的无依据重试。
- 同目录`INDEXER_PANEL_GROUP_EXPERIMENT.md:11`：N64两面板成组循环完整PASS，中位920.58 μs，同轮原task909.50 μs，未采用；不能将循环化减少展开重新包装为新方向。
- MASTER:6141–6155记录旧静态balanced外部PMU的I-cache字段AIC 0.002→0.006、AIV 0.003→0.013。记录覆盖旧版本完整持久核，并有测量范围/长尾限制，不能外推为当前两档cache压力。
- 既有`A3_FFTS_WAIT_TIMING_READONLY.md`及其`wait_machine_audit/result.json`已保留官方objdump在真实A3控制对象上也无可用助记符的失败证据；本轮不重复包装/反汇编。当前不能统计真实分支、spill load/store或hot-PC取指停顿。

因此现有信息仅支持“OA新增动态路径扩大了长B16完整image”的有限事实。没有当前cache计数、可解码指令/函数边界或hot-PC归因，不能由字节数认定硬件cache压力，也不足以安排一个新的大T性能候选。
