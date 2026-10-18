# 动态 group wave 边界：源码准备与整数检查

2026-10-18。源码位于`source_candidate/wave_boundaries.py`；这是待接入的高层device Scalar helper，**没有挂schedule，没有PyPTO解析/lowering或NPU测量**。没有改正式仓、依赖、现有两L1候选或冻结产物。

接口：`next_wave_boundary(offsets, lengths, tokens, first_group) -> (next_group, query_end)`。请求数B只取`lengths.shape[0]`；groups读`offsets[B]`；基址`pad16(B+1)`后的四项描述符沿用正式`indexer_group_stages.py`的`(request, first_query, live, max_visible)`。边界只读device first/live，不猜固定S，不用B*S，不修改描述符或全局group编号。

从`first_group+8`开始，用Scalar while扫描第一个`live>0 && first_query%8==0`的下一group；命中即停止，没有命中就返回`(groups,T)`。调用者从`(first_group,query_begin)=(0,0)`开始，每轮沿用前次query_end作为下次begin。首段覆盖前dummy，末段覆盖后dummy，组间gap归入前一段。除最后一段外，每段至少8个真实非空group；8是现有M8 plan的行块约束，不是某个batch或S的白名单。当前全局Score owner必须继续按原groups、全局group/leaf/item计算，不得把wave内的数量传给旧owner公式重新编号。

合法描述符的first严格前进、live>=1且范围单调不重叠、总跨度位于[0,T)。因此groups<=T。若W>0，前W−1段各至少8组、末段至少1组，故`8*(W−1)+1<=groups`，得到`W<=ceil(groups/8)<=ceil(T/8)`。非末query_end是下一group的first且整除8，前面group已经全部包含；因此任何实际M8 plan行块不会横跨两段。这里只保证边界，不代替前波TopK根、merge和plan发布/等待。

`groups==0`返回`(0,T)`仅作为空输入结果；调用者必须走原不交错路径覆盖dummy/gap，不把它计为一个wave。非法、零live、重叠或越界描述符不是本helper支持的输入；不能用此公式给非法生产者输出兜底。

与同步代理对齐：保留60个固定phase，新增每wave ROOT/TOPK/PLAN三个phase，Host保守容量为`60+3*ceil(T/8)`，总signal words为该值乘`72*128`。实际wave数、group/query边界始终由device计算，不新增Host动态边界字段。该布局是并行设计合同，尚未在本目录实现信号扩容。

`check_boundaries.py`只使用标准库，提取同一helper AST去掉注解/装饰器，用受检查的dim/read/cast整数替身执行，再与独立列表过滤oracle比较。1,235个场景全部通过：S1/S5/S6各含G1/G2/G6、前后dummy和中间inactive gap；mixed/odd/零长度请求；T边界至4096；23个空groups回退；另4个非法描述符被合同检查拒绝。检查完整query/group覆盖、严格前进、无重叠、M8不跨段、波数上界、原metadata不写入和容量尾部不读取。4个非法用例属于CPU合同验证，未声称device helper完整验证非法输入。

复现：

```bash
python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_attention_wave_boundaries_20261018/check_boundaries.py
```

结果保存在`boundary_checks.json`，只保留少量代表边界清单，其余计入数量/合同汇总；没有hash、性能模型或性能结论。下一步等待统一调度接口，不自行增加编译/测量。
