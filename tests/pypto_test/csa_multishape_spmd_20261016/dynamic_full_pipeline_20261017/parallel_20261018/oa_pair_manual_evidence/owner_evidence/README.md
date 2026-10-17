# O_A pair 接入动态 CSA 的覆盖与等待合同（只读设计，2026-10-18）

**当前结论：采用第6节互异owner映射进入独立CPU候选；第2–5节保留最初映射的历史设计。仅替换数学 helper 不成立。** 至少同时调整 O_A 的pair/single分工、AIC组发布条件和AIV量化等待集合。已用有限模型核对T=1..96的所有输出唯一写、量化等待覆盖及业务phase无环；没有生产修改、完整CSA CPU或NPU验证。

当前源码基准66882634。相关实现：`policy.py:115` 的 `oa_cols=128 if tokens<=96 else256`；`cube_schedule.py:403–430` 的OA/quant/OB组流水；`vector_schedule.py:926–945` 的量化等待；`output_projection_a_stages.py:109–170` 的原NZ O_A组helper、`:180–245` 的量化。路径均在主仓 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`。

## 1. 区间与实际动态行

沿用 `OA_COLS=128` 的既有 **1≤T≤96**，此时 `ceil(T/128)=1`。不要写成T≤128，不新增B/S白名单，不把T固定为96。若手动controls指定N128却给T>96，必须进入原helper/owner分支或在既有入口明确拒绝；不能悄悄用本单M块分工。正式policy本就拒绝T=0；helper的空输入约定是无任务，不发布无人消费的组。

新pair helper采用动态 `PACK_TOTAL_ROWS`、`MM_ROWS`，实际packed组步长从Tensor轴求出，行有效值是T。不能携带资源探针里的固定3072/384根容量进入正式实现。P_MM可能只有32或96，不保证128；M128 Tile的load/store必须保持动态有效T，实际未写区和guard不能靠扩大buffer掩盖。singleton同样只写T行。

## 2. 每组3对+2单块的精确分工

对模型组g=0..7：pair p=0..2写N128块 `(2p,2p+1)`，owner=`3g+p`；single s=0..1写块`6+s`，owner=`2g+s`。每块输出覆盖列区间`[128n,128(n+1))`，GM输出加组列偏移`g*1024`。

| g | pair owners（分别写0/1、2/3、4/5） | single owners（分别写6、7） | 量化必须等待的新owner并集 | 原owner集合（禁止沿用） |
|---|---|---|---|---|
| 0 | 0,1,2 | 0,1 | 0,1,2 | 0,1,2,3,4,5,6,7 |
| 1 | 3,4,5 | 2,3 | 2,3,4,5 | 8,9,10,11,12,13,14,15 |
| 2 | 6,7,8 | 4,5 | 4,5,6,7,8 | 16,17,18,19,20,21,22,23 |
| 3 | 9,10,11 | 6,7 | 6,7,9,10,11 | 0,1,2,3,4,5,6,7 |
| 4 | 12,13,14 | 8,9 | 8,9,12,13,14 | 8,9,10,11,12,13,14,15 |
| 5 | 15,16,17 | 10,11 | 10,11,15,16,17 | 16,17,18,19,20,21,22,23 |
| 6 | 18,19,20 | 12,13 | 12,13,18,19,20 | 0,1,2,3,4,5,6,7 |
| 7 | 21,22,23 | 14,15 | 14,15,21,22,23 | 8,9,10,11,12,13,14,15 |

24个AIC各恰好一个pair；前16核各另一个single，因此core0..15为3个N128输出，core16..23为2个，不出现32pair方案中8核4输出的长尾。总共64个N128输出无缺漏/重写；总GM数学布局不变。

组发布是owner并集，而不是40个工作项逐次发布：g0的core0/1、g1的core3同时拥有pair和single。它们须做完该组的两项工作后才发布该组一次，不能第一次输出完成便发信号。新方案共37个真实 `(group,owner)` 发布，原N128方案为64个；这一静态计数不等于性能收益。

## 3. 必须一起修改的位置

1. `output_projection_a_stages.py`：新增实际dynamic轴的NZ pair数学体，使用已CPU通过的四K256普通range/局部Mat方案；另保留single的原N128完整K链。提供按group/worker处理pair与single的阶段helper。二者必须独立判断，不能写`if pair ... elif single ...`，否则重合owner漏掉最后单块。single应从现行同源N128数学抽取，不能通过虚构worker号跳进旧分工，也不能遗漏原首matmul与后续K累加语义。
2. `cube_schedule.py`：仅满足候选区间时改OA组分工判定为 `worker in [3g,3g+3) or worker in [2g,2g+2)`；调用该group所有本核工作后，用原 `publish_phase(PHASE_OA_GROUP_BASE+g)` 一次发布。`PHASE_ATTN_VECTOR` 的packed等待保留。OA组循环仍g升序，本核全部OA完成后才进入原OB组循环。
3. `vector_schedule.py`：仅量化消费者的OA等待集合改变，等上述并集后执行原量化helper。quant的 `(g*ceil(T/32)+row)%48` owner、最后32行块的INT8尾零责任及其发布保持不变。OB等待这些quant owner的公式也不变。
4. `phase_sync.py`（或等价局部专业等待helper）：采用两个短区间owner并集，先等`3g..3g+2`，再等`2g..2g+1`中不重合者；全部满足后只做一次原有 fence/cacheinvalid/fence。不能把新owner当作原连续logical%24，也不应为省事每等一核就重复整份业务cache维护。原信号代际、512B隔离、TWAIT、发布fence都不变。

无需新GM工作区、新phase或runtime Tensor根。策略默认值不改，候选接线可只使用现有 `OA_COLS` 和真实T谓词；同一谓词必须同时控制producer与consumer。若关闭 `OUTPUT_GROUP_PIPELINE`，最小候选保持整阶段原helper和原等待原样，以免把未审核分支顺带改造。N256区间也完整保留。

## 4. 依赖顺序与无环条件

```text
Attention AIV 写packed并发布
  → AIC按g升序做本核OA pair及single
  → 该组本owner所有输出写回、原发布fence/cache维护
  → AIV等待该组真实owner并集
  → 原量化、scale、INT8尾零完成并发布
  → AIC在本核全部OA完成后，按原OB组owner等待quant并算OB
  → 原PHASE_OB与HCpost链
```

无环证明依赖**OA阶段不反向等待quant或OB**。本模型保留此条件；不能为了更早做OB而把未完成的OA组置于一个可能等未来组quant的阻塞之后。所有核读的packed已按原真实AIV生产者完成；不按经过时间猜测可见性，也不删FFTS drain或普通phase缓存维护。

每个量化任务都读同一行的全部1024列，所以需要三个pair和两个single的owner并集。新g1例如需要core2，但旧等待集合8..15既漏core2又会等待不发布本组信号的核：不改等待会读陈旧数据或卡死。不能用旧集合“多等一点更安全”的理由保留。

`check_owners.py`仅执行Python有限模型：96个真实T的 `(group,row,N128块)` 唯一写，所有量化消费者的owner并集完整，按当前AIC组顺序/全部OA后OB形成的业务DAG无环。它不模拟硬件pipe事件、动态cache一致性或运行时信号，因此不能代替后续整链精度/图/保护/协议验证。

## 5. 性能风险与下一步边界

输入复用有明确字节来源：原64个独立N128任务各读一次A；新24pair+16single共40次，逻辑有效A字节由`64*T*4096*2`变`40*T*4096*2`。权重、MMA数、最终输出写入并未同比减少，不能把37.5%的A字节减少换算成完整L1收益。

**最早组就绪可能变慢。** 新g0的core0/1各做pair+single三输出才发本组信号，旧g0每owner只做一输出；g1的core3同样延后。因此即使每核全阶段总输出最多仍3块，新分工也可能抵消现有OA→quant→OB组流水的早发收益。当前没有新时序或设备证据能量化这个取舍。独立pair的v2 CPU通过也未证明整层同时活跃资源上界或same-math设备精度。

本轮只形成可实施的覆盖/同步合同；是否进入完整CPU由主会话决定。后续若获准，必须先动态root布局和同源数学审计、完整CPU真实allocator与事件，再考虑整链数值/保护/代际，不能直接把独立pair探针称为完整实现。

## 6. 经主会话修正：single分给下一组两核

最终候选把single owner改为`(3*g+3+s)%24`，pair仍`3*g+p`。这来自8个模型组和24AIC结构，不是B/S白名单。每组五owner互异，16个single全局owner互异；编号模3为0/1的16核各3个输出，其余8核各2个。`check_rotated_owners.py` 的T1..96覆盖/等待/DAG模型通过。

| g | 新pair owner | 新single owner | 量化等待并集 |
|---|---|---|---|
|0|0,1,2|3,4|0,1,2,3,4|
|1|3,4,5|6,7|3,4,5,6,7|
|2|6,7,8|9,10|6,7,8,9,10|
|3|9,10,11|12,13|9,10,11,12,13|
|4|12,13,14|15,16|12,13,14,15,16|
|5|15,16,17|18,19|15,16,17,18,19|
|6|18,19,20|21,22|18,19,20,21,22|
|7|21,22,23|0,1|0,1,21,22,23|

现在g0不再有任何同组pair→single的串行边；但对g1..7，pair的前两名owner会先做上一组single，然后才做本组pair；g7的single owner0/1先做过g0 pair。这只是换了跨组前序工作，不能断言所有组都提前。`mapping_comparison.json` 将两方案每个group就绪条件写成“各owner累计P(g,p)/S(g,s)之和的最大值”，没有假设pair耗时等于两倍single，也没有伪造微秒数。

最终接线保持三pair连续区间和single循环区间两次短Scalar信号遍历，合计五核，之后仅一次原业务fence/cacheinvalid/fence。所有OA仍先于本核OB等待，因此没有引入quant→未来OA的反向边。新候选在 `../oa_pair_full_candidate_20261018/`，冻结66882634全部33源，只改四文件；已获准且正在执行唯一完整CPU，禁止NPU。该目录结果为准，本文有限模型不是后端验证。
