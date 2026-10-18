# AIC Score/Attention 单 wave executor 源码交付

本目录记录共享候选包新增的 `interleave_cube_wave.py`。源码已完成并冻结；只做标准库 AST 和协议事件 DAG 检查，未导入 PyPTO、未 lowering、未 CCEC、未上卡。正式包和已有冻结候选未修改。本次没有计算 hash。

## 文件与调用合同

共享源码：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/source_candidate/interleave_cube_wave.py`。

`interleave_cube_wave_g6` 的 Tensor 参数顺序为 Score 的 query/coeff/key0/key64/Native cache/page table/positions/lengths/bounds/offsets/plan/transfer，随后 Attention 的 flattened q_heads/mask/KV ring/score ring/prob ring/PV ring/FFTS。Scalar 顺序固定为 `worker, score_group_begin, score_group_end, attention_query_begin, attention_query_end`；constexpr 为 `CORE_GROUPS=24, SCORE_READY_EVENT=0, SCORE_CONSUMED_EVENT=1, ATTN_EVENT_BASE=2`。返回 `indexer_transfer, score_transfer, pv_transfer`。

只接受既有 G6/N64/1024/balanced/roots1/prefetch1/toL0 数学策略。B/T/Q/L/position/page/leaf 循环仍为设备动态值。外层须先取得 Score 输入和前波 Attention 的真实 owner 发布，使用同一 FFTS 域，并按各 wave 独立 ROOT/TopK/PLAN 发布协议执行。当前共享 schedule 只在实际存在内部 wave 边界时调用交错分支；首次 Attention 的原 QUERY/cache/rope acquire 不能删。

## 源码和边界证据

- `score_cube_leaf_init_g6` 保留原有效行 load、set_validshape(384,128)、按首有效 query 补齐无效 lane、Coeff Mat load 和 Left TMOV。五个子树只把 `query_group_size` 换为 6 后与 formal668 原 AST 一致。
- 原 Query Mat→Left 一次 TMOV 从叶初始化移到每个完整 Score step 前。Query Mat[384,128] INT8 和 Coeff Left[16,384] FP16 跨轮保存；每步重新恢复 Query Left，防止它被前一 Attention tick 的 Left 使用覆盖。是否生成正确的地址/事件仍须联合 lowering。
- 显式循环携带四份 Tile（上述两份及 Attention Q/KV Mat）和七个 Scalar（游标、step、steps、batch、cache_len、logical_begin、lane_span）。六处分支 phi 在各分支末端返回同名、同数量的值。
- `score_wave_step_count` 与 `next_score_leaf` 读取同一本次不可变 plan。每个 wave 重新从物理 worker 的全局 item 条带开始，命中后继续使用返回的 next_item；不把 wave 内编号当 owner。
- 创建的 Score 占位 Tile 只携带句柄：仅 `found>0` 初始化，且仅 `steps_ready>0` 读取其内容。Attention 占位 Q/KV 仍由原 tick 的 query/block/mask 控制使用。
- Attention 的首 query 是当前半开区间内满足 `q%24==worker` 的首项；GM ring 仍按物理 qk_core。实际每核 query_count 决定 5 个块/条 query 与末两拍，不能用 qk_core 代替 query_first。
- 空 Score 区间不发 Score READY/ACK；空 Attention 区间只走两拍无业务读取控制。所有 Attention mask 保持原 helper 谓词；末两拍和叶末 ACK drain 均保留。
- 叶尾 drain 与原 AST 仅变量改名后相同，每叶排空 `min(steps,2)`。它只说明 Score FIFO 已读，不说明 AIV 根排序/写回或 TopK/plan 完成。executor 本身不读写 phase，不增加业务 fence/cache 操作。

`source_audit.json` 给出五个 init 子树、六个 helper 实参/形参、分支 phi、carry 和 drain 对照。共享候选根的 `cube_executor_audit.json` 是同一审计摘要，供主流程直接读取。源文件语法 PASS 不能代替 Tile JIT/后端通过。

## 叶尾 drain 的无环范围

AIC 本地轮 r 先产生 Score，再做 Attention 的 PV(r−2)→QK(r)；双 AIV 先完成本轮 Attention，再读本轮 Score FIFO、发 ACK、继续原数学。AIV 本轮 Attention 等待的是 QK(r−1) 和 PV(r−3)，二者均由 AIC 更早的 tick r−1 产生；早 KV 分支也先等原 QK(r−1) 再发当前 KV。因此 AIC 当前叶尾 drain 等本轮 ACK，不会要求 AIV 等待尚未执行的 AIC Attention(r)。不同核的本地轮数不用相等。

`check_event_dag.py` 按实际先后列出单个 AIC 和两个 AIV 的 READY/ACK/KV/SCORE/PROB/PV 节点，并保留 ACK 后的 Score 数学/叶根写回及无效末块输出重置。覆盖空 Score、单叶 1..8 步、七种跨叶两槽/重置边界，0/5/10 Attention 块的所有有效 mask，以及早/晚 KV 两分支；共 33,824 例。所有 wait 均有对应 emit，全部 DAG 无环。结果在 `event_dag_result.json`。

这是同组事件偏序检查，不是硬件模拟。它没有验证完整跨 wave metadata、外层 phase 发布、CCEC 局部事件、FIFO 机器实现、物理地址复用或异步 DMA 完成。Mat 双槽必须保留 PV 最后 MTE1 读取到下一 QK MTE2 覆盖的依赖；源码先后不能代替这一生成物检查。

## 待联合 lowering 核对

1. full-valid Mat/Left Tile 的 helper 入参、分支 phi、循环回边是否保持类型和原地址；初始占位 Tile 不应引入永久额外槽。
2. Query 48 KiB Mat、Coeff 12 KiB Left、Attention 两槽以及 Score 瞬态 Mat 的实际峰值；不能把名义容量当 allocator 通过。
3. 每步 Query TMOV 的 Left 复用，以及 Attention PV→QK 的两槽覆盖事件；已有业务 FFTS/原数学 helper 不应被前端改变。
4. 三份返回 Tensor 仍为原工作区别名，单 MIX、动态 ABI/轴以及完整精度需由联合验证确认。额外计数遍历和每步 48 KiB TMOV 有开销，不预记收益。

复现本次标准库检查：

```bash
python '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/score_attention_cube_wave_source_20261018/audit_source.py'
python '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/score_attention_cube_wave_source_20261018/check_event_dag.py'
```

最小归档：README.md、audit_source.py、source_audit.json、audit_stdout.txt、audit_stderr.txt、check_event_dag.py、event_dag_result.json、event_dag_stdout.txt、event_dag_stderr.txt；实际新源码保留在共享候选包，只是一份约 15 KiB 的源码文件，无大生成物复制。

