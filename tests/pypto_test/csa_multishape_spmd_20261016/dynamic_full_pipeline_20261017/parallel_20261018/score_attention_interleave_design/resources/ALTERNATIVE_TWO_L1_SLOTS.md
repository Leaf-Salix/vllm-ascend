# 替代路径：仅 AIC L1 KV 两槽，GM 三槽不变

2026-10-18，继续只读同目录记录的formal668源码/现成分配。**没有发现遗漏的必需持久Mat或Left导致名义容量超限；这条路径避免了异dtype KV槽别名，现有高层算子可描述所需访存。尚未经过新lowering、设备正确性或性能验证，不能据容量成立采用。**

## 状态与空间

| 项目 | 同核驻留/工作量 | 生命周期与限制 |
| --- | --- | --- |
| Attention Q | Mat64KiB | query内保持，首块更新；PV不读Q，先执行旧PV再更新下一query的Q不损失旧PV输入。 |
| Attention KV | Mat2×128=256KiB | 一个同dtype BF16[256,512] Tile；QK后保存到两拍后的PV。 |
| Score Query | Mat48KiB | 本leaf保留原INT8[384,128]（含现有安全padding）；切回每个Score step时重新TMOV到Left，不能让现有Score临时覆盖它。 |
| Score Coeff | Left12KiB | 本leaf保留FP16[16,384]；初始化用的Mat12KiB是暂态，不是另一份必须跨step保存的内容。 |
| Score step临时 | Mat112KiB | 两份FP16[384,64]各48KiB、INT8 key pool16KiB；panel内预取与前块归约顺序保持，完整step后不必留到Attention。 |
| Attention PV临时 | Mat16KiB | BF16[64,128]概率块；无须跨完整tick保留。 |

Mat的保守并列和为64+256+48+112+16=**496KiB**。把leaf初始化的Coeff Mat12KiB也独立计入，仍为**508KiB**；它与其他暂态实际可分时，不能据这笔总和强制保留所有对象。现有Score的`buf_panel_keys`8KiB在PREFETCH=1路径没有独立有效load/消费，实际分配也与后续48KiB panel地址共用，不是遗漏的跨步状态；所有真正prefetch key在16KiB pool里。

Left不能把不同阶段机械相加：Score需Query48+Coeff12=**60KiB**；Attention需局部16+仍驻留Coeff12=**28KiB**。两段的Query/Attention临时可分时重用，Coeff必须被联合生命周期保护。Right与Acc均无跨完整step/tick必须保留的数学状态，仍需原MTE1/M/FIX完成依赖才能复用；不要靠Scalar语句结束判完成。AIV的roots/segments与m/l/oi/swap仍沿主报告的104.25KiB必要状态分析，缩AIC L1不减少AIV UB需求。

## 源级表达与两个不同的槽号

沿`attention_stream_stages.py:136–200`现有`create_tile/gather_row/transpose_view/tile.slice/extract`即可写同dtype两槽，无需boxed `reinterpret_view`、多dtype MemRef别名或固定物理地址。必须分开两个常量：**GM槽始终work%3，L1槽为work%2**。每tick r先按原mask处理PV(r−2)，从L1的(r−2)%2读取；完成该槽的MTE1读取后，再把QK(r)的新KV写入r%2。r<2只做QK，末尾两拍只排空PV；无效块仍在两侧按同一mask跳过事件/访存。所有GM score/probability/PV/mi/li仍三槽，AIV逻辑不因这一改变改成两槽。

复用关键是实际MTE1读完成，不要求为了覆盖L1而笼统等待所有远端消费。现有C++tick尾/下tick的MTE1→MTE2复用依赖给出需要保留的类型；新顺序须由联合lowering在最后PV extract与下一QK gather之间生成对应等待。同步代理已独立确认该PV先行顺序没有新增跨核依赖环，此处不重复协议模型。

“高层操作能描述”仍不是“整个状态机已可编译”：需要确认Score Query Mat跨步保持、Coeff Left未被Attention自动分配覆盖、Mat回边未引入不支持的Mat→Mat搬运，以及完整函数的生命周期/对齐/临时区实际峰值。当前inline不能直接以Tile参数拆成两个step helper，应在同incore控制流中保留状态或使用实际支持的接口，不能假设Tile传参已经可用。

## 明确增加或丢失的工作

每次返回Score增加48KiB Query Mat→Left重装；从下一leaf换入新Query/Coeff仍有原GM读取，不能算成零搬运。把PV提前到QK前会推迟本轮SCORE_READY，减少原先QK领先发射、随后PV的重叠机会；新顺序的收益需与两流水互补一起实测，不能从名义空间或各阶段时钟相减得到。496/508KiB只留16/4KiB名义裕量，真实allocator可能因对齐、跨回边活跃或局部事件产生额外约束。

本轮没有修改源码、没有编译、没有上卡；结论仅是值得保留的资源/表达替代路径。
