# Score step / Attention tick 交错：有界资源审查

2026-10-18。只读正式668源码快照和同次Score诊断的已生成PTO；未编译、上卡或修改源码。**容量没有证明这一方向根本不可行，但原helper不能直接交错：必须处理Score Left重装与KV槽临时复用。**以下KiB均为1024B，讨论长G6/N64/score1024、Attention N128；不是性能预测。精确证据路径/行号见`evidence.json`。

| 空间 | 必须跨边界保存 | 可在完整step/tick边界释放或复用 |
| --- | --- | --- |
| AIC Left，64KiB | Score Query INT8[384,128]=48KiB、Coeff FP16[16,384]=12KiB；当前整leaf重用。 | Attention QK/PV实际各需16KiB，原地址0重叠Query；60+16>64。Right/Acc只在各自计算段活跃，可以串行复用，但必须保留原MTE1/M/FIX完成依赖。 |
| AIC Mat，512KiB | Attention Q BF16[64,512]=64KiB；KV三槽各128KiB，QK后留到PV(tick−2)。 | Score step内两FP16 panel各48KiB、key pool16KiB，共112KiB；原Query/Coeff Mat在移Left后本可释放。Attention PV临时16KiB。三个KV物理槽独立保留时，448+112=560KiB已超限。 |
| 每个AIV Vec，当前规划预算184KiB | Score roots12KiB+segments24KiB；Attention oi两半共64KiB、m/l共256B、逆RoPE swap4KiB：合104.25KiB。 | KV半块64KiB、softmax/PV/最终发布临时、Score scale/排序临时不必跨完成边界保留。104.25+64=168.25KiB，尚无名义硬超限；现成Attention所有Vec地址上界132.75KiB，加独立Score36KiB为168.75KiB。它是已有布局加驻留状态的容量检查，**不是新planner已通过**。Score现成roots在0、segments在64KiB，与Attention oi/KV直接冲突，需重排。 |
| GM | 两套既有FIFO及已发布根/plan须保留，各自控制槽生命周期。 | 不需要因切换就复制所有FIFO；不能用phase通知代替逐槽消息，也不能把异步store发出当作可覆盖。 |

**最小有容量依据的边界**：Score完整1024-step的最后WS/FIX/READY后切换；Attention完整tick的QK和PV(tick−2)后切换，旧KV槽须真正完成读取。tick r末仍需保留r与r−1的KV，r−2槽才可能借给下一Score step的112KiB临时；mask空块也必须按原事件谓词管理。不能凭Scalar代码走出循环认定MTE1/M已经完成。独立协议审查已核对同控668 C++：tick尾18139–18141的MTE1→MTE2、M→MTE1、FIX→M通知由下一tick17664–17666等待；借槽仍须这些实际复用依赖，且不能拼接两helper都使用EVENT_ID0的生成代码。

Score不必每step重读GM：可把Coeff12KiB继续留在独立Left，Query48KiB保留Mat，每次返回Score时重新TMOV Query。这样Attention段Mat=64+384+16+48=512KiB恰满；Score段在借出已释放第三槽后，逻辑活跃量=64+256+48+112=480KiB。代价是每次切回48KiB MTE1重装、零Mat裕量和显式槽别名管理；不是免费保留。若Query与Coeff都保留Mat，Attention峰值变524KiB，仍需额外重用；从GM重载60KiB是另一有真实新增读取的退路，不据此预测收益。当前源码只在leaf加载一次，改变加载频度必须单独计成本。

**现有表达能力的边界**：PyPTO已有`pl.MemRef(slots=N)`及运行时槽索引、`tile.slice/reinterpret_view`；但动态槽共活检查明确交给作者，跨BF16/FP16/INT8的NZ布局重用未被本审查证明。`tile.alloc`是内部打印IR stub；`create`无地址参数。现成测试还注明PyPTO planner level3动态槽退化为保守alias，PTOAS多槽要求统一shape。后续只读已发现boxed `reinterpret_view`明确受限，详见`API_GAP.md`；这不等同不同dtype物理MemRef绝对不能共享。不能把“API存在”当作空KV槽可以安全拆成两个48KiB+16KiB的证据。下一关应先证明这一个局部存储合同，不能直接搬两份大helper期待自动复用；当前inline Tile传参限制也不支持凭空拆成持Tile参数的step接口。

三槽降两槽不是本轮结论：若只缩AIC L1，原先先QK(r)后PV(r−2)会覆盖同一槽，至少需改为先消费旧PV并重新证明事件/重叠；若把所有GM ring一并改两槽，AIV先写mi/li(r−1)再读mi/li(r−3)还会撞槽。不能仅修改常量省128KiB。

历史排重：`PAR/wave_audit/README.md`只覆盖三角色排空后整helper切换，明确没有否决本次细粒度忙闲互补；`PAR/attention_aiv_d253_audit/README.md`及多消费者/direct-L1记录是更换KV生产者/传输路径，不是本方案。旧`attention_local_ring_evidence`的768B状态却导致整体255KB规划失败，说明名义容量可行仍需真实生命周期验证。这里只给资源前提；协议无环由独立审查负责，尚无新实现/后端/精度/性能结论。
