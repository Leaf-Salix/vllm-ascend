# Wave word0 写者检查的最小设计（未接runner）

当前runner检查所有新旧padding、原phase0代际及旧输出phase owner；尚未核对新增ROOT/TOPK/PLAN的实际word0写者。以下只依据当前源码设计，没有新增metadata回读、CPU模型、编译或设备调用。

| phase | 应发布的AIV core集合 | 源码依据 |
| --- | --- | --- |
| ROOT(w) | 对本wave每个真实有效leaf，保留原全局item owner；`item=group*leaves+leaf`，groups<24时为`leaf*groups+group`。对lane=0/1，仅live>lane*3时加入`24+2*(item%24)+lane`；对一个core取并集，每波发布一次。 | `interleave_control.next_score_leaf` 的有效叶条件；`interleave_vector_wave` 叶尾`leaf_wrote=live>lane*3`，波内取max；`vector_schedule.py:875`仅root_wrote>0发布。 |
| TOPK(w) | `{24+q%48 : query_begin<=q<query_end}`，包含dummy/gap行，因为原range publisher仍写-1/-inf。 | `vector_schedule.py:885`起first_topk条件。 |
| PLAN(w) | `{24+(q//8)%48 : q为该wave覆盖的全局M8块首行}`；内部wave边界8对齐，最后一块允许尾行。 | `vector_schedule.py:897`起begin_plan_tile与first_plan_tile条件。 |

三者均无AIC writer；core0..23的新增word0应始终为0。实际wave边界必须从本次真实group descriptor重建，不能按B×S或统一最大history猜测。ROOT需要完整的有效leaf/visible/live信息，不可把“waiters会消费哪些根”当作所有真实writers；尾group的第二lane可能没有根写回。

完整离线checker需要每次已同步检查点保存本次T、group_plan、有效group描述符，以及query边界所需metadata的快照，或已有可信的同等文本证据。它应在设备计算和性能窗之外收集，而不是在输入准备/每wave过程中插入CPU同步。当前runner未保存足够数据，所以不能伪称已有严格ROOT writer验收。

对首次清零workspace，各phase期望为writers写本次generation、其他槽0。对metadata重放及后续代际，未使用wave槽不会被自动清零，必须从前一次快照递推：本次writers设为generation，其余保持旧值。不能要求所有未使用槽为0，也不能要求所有预留wave槽都等于当前generation。原60phase语义检查继续独立保留。

建议后续授权后，在现有launch回读点增加小型快照并用标准库离线核对；当前不改runner执行路径。即使writer集合通过，也只能核对发布覆盖/代际和越界写，不能独立证明release/acquire、FFTS和片上异步复用时序正确。
