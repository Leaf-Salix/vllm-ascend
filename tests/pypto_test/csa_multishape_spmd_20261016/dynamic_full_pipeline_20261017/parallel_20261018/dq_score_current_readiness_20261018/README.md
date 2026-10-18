# 主Query DQ与Score重叠：协同前段后出现的新依据

2026-10-18，只读复核既有证据；尚未写候选、不声称收益。

旧 `dq_score_interleave_readonly/README.md` 根据当时的执行次序降低本方向
优先级：DQ大部分发生于MAIN_PROJECT尚未结束的期间，移走只会更早等MAIN。
之后已采用COOPERATIVE_PREFIX，主压缩投影和AIV消费都前移。该旧判断不宜
继续直接外推到当前入口。

使用已完成的 `attention_dependency_evidence/h131072_b16_phase_alignment_v1`
三轮现有端点，未新跑探针。单位μs，全部是插桩观察：

| 轮 | MAIN_OUTPUT最晚 | IDX_SCALE最晚 | QUERY最晚 | QUERY前同核区间中位 |
|---|---:|---:|---:|---:|
| 0 | 244.20 | 277.14 | 300.22 | 20.87 |
| 1 | 237.98 | 272.48 | 295.24 | 20.97 |
| 2 | 240.08 | 275.16 | 299.40 | 21.03 |

当前MAIN_OUTPUT已在DQ之前完成，AIV的DQ之后直接进入Score；原“移走DQ只是
更早等主压缩”的抵消因素已经改变。区间仍含全QB获取/缓存维护和插桩，
不能把整段差值直接称为可省时间。Score的逐步空隙长度尚未实测。

可继续评估的单一设计：仅在现有long G6/balanced区间，将原owner的DQ按
最多8个head切段，只在Score偶数步（尚未执行2048排序）完成ACK/本步scale
写入后插一段，减少与奇数排序争用。相比旧4-head设计，8-head减少重复
RoPE/scale准备；是否能藏入AIC供数时间仍需同轮完整L1。

原head分组选择和owner必须保留：dq_work=worker+k*48，head_offset在本任务
内推进，完成原head_tile才前进work。尾行仍4head并沿用recip(sqrt)数学；
完整八行用原逐head stage2。应提取可复用的数学helper，避免复制整份DQ。
所有DQ输入在Score前取得，不在已持有Score槽时新增前驱等待；零Score工作
的AIV以及剩余DQ都在尾部补齐，然后发布原QUERY/SCORE_VECTOR。保持全部
Score READY/ACK/drain及后继QUERY获取，不用SCORE_CUBE整段完成作前驱。

持久Score根12KiB+segment24KiB与DQ临时的实际UB共存必须重新编译核对。
新局部统计环位于后续Attention，不与Score/DQ同时存活，但共享完整函数的
内存规划可能改变，不能只把局部大小相加就宣称可装下。短G2及S1保留原路径。
下一步以最新正式快照为参照做源级候选和CPU检查，不重测旧只读方案。
