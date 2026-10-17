# 压缩 Attention gather 重复 guard 删除：独立只读审阅

审阅日期：2026-10-18。结论：在当前完整单 SPMD 的输入、工作区与顺序调用契约内，未找到本次删除造成非法地址或数学变化的反例。这里是源码证明与既有证据核对，不是本候选设备 PASS 或性能结果。本次未编译、未运行 CPU/NPU 测试、未修改任何实现、未计算校验散列。

审阅目录：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_attention_guard_elision_20261018`。下文源码行号均相对其 `source_candidate`。逐文件直接文本比较确认 `source_before` 与候选各有相同路径的 33 个 Python 文件，只有 `attention_stream_stages.py` 改动：压缩分支删除表列、物理页两层 guard；非负索引检查、地址公式、采样顺序、SWA、mask、事件及数学体未变。

## 核心不变量与边界

`attention_plan_stages.py:204` 先得到经过原 FP32 算式转换的最终 INT32 `c_page_indices`，页检查针对这个地址值，不是未经转换的原始 Top-K。`288–290` 最终输出使用相同的 `(mask * (raw + 1)) - 1` 算式；整数输入及 clamp 使 mask 为 0/1。页检查通过时 mask 保留原值，输出等于被检查的索引；不通过则写 -1。INT32 输入转 FP32 仍有限，因而不存在 `0 * inf` 破坏失效值的情况。消费者保留 `ridx >= 0`（`attention_stream_stages.py:317–330`），只可能读取已验证地址。

| 边界 | 源码证据与结论 |
|---|---|
| FP32 精确范围内 | `attention_plan_stages.py:104–131` 只在 `L <= 2^24`、`0 < physical_pages <= 2^24` 且表列足够时启用整前缀验证。活动 query 满足 `0 <= position < L`，故压缩上界不超过 `L//4`；`226–236` 另验最终索引范围，所有可达列都在已检查前缀。 |
| FP32 精确范围外 | 长度或物理页数超界使快路径返回 0；`237–246` 直接读取最终 INT32 索引，检查非负、上界、表列及 INT32 页号。原 FP32 上界可能舍入，但不会绕过地址检查。未假设大整数数学等于精确整数除法。 |
| 超长/短表、空洞、负页、越界页 | 超长表只需验证可达前缀；短表在快路径加载前被拒绝，慢路径检查列后才读页。负页及超过 cache 页数的页被屏蔽。页号超出 FP32 精确范围时，若物理页数仍不超过 `2^24`，舍入也不能把非法页号移入合法区间；物理容量更大则直接慢路径。1024 列检查尾块使用 `valid_shape=[1, live]`，归约也限制 live（`110–129`）。 |
| inactive、mixed Q、奇数尾行 | `request_geometry.py:50–60,77–92` 要求合法正长度请求与唯一 owner；`attention_plan_stages.py:168–175` 再验本次 causal position。失活行 owner=-1，页有效 mask 保持 0，索引写 -1、块标志为 0。实际行以 `T` 和 `rows=min(8,T-first)` 控制（`153–156,290,303–305`），未使用容量尾行。请求改变时重新检查页前缀（`210–225`），没有跨调用缓存。 |
| `CACHE_QUERY_METADATA=True` | plan 从已通过活动/causal 检查的 owner 写 mask 第 5/6 列；位置从原 INT64 无损转 INT32，而非从 FP32 回读。失活行显式写 (-1,0)（`180–190`）。消费者仅在当前已发布的 mask 中读这些值（`327–333`）。关闭分支重做同一几何判断（`334–342`）。 |

## 真实调用与发布链

候选源中该 gather helper 只有 `vector_schedule.py:891` 一处业务调用。同一函数在 `865–879` 调用 plan，随后 `891–917` 调用消费者：两处均传 `positions`、`idx_bounds`、`idx_lengths`、`compressed_table`、同一份 indices/bias/mask，且传同一 `ATTENTION_QUERY_METADATA`。plan 的压缩物理页数直接来自 `dim(main_cache,0)//32`（`876`），gather 的 cache 参数正是 `main_cache`（`893`）。Native 绑定分别为 `cmp_block_table`、`cmp_kv.view(-1,512)`（`binding.py:102,145`），没有另一个容量来源。

plan owner 为八行 tile 按 48 个 AIV 轮转（`attention_plan_stages.py:155`），实际 signal core 为 `24+worker`（`vector_schedule.py:367–369`）。Top-K 先发布并等待真实 owner（`863–864`）；plan 完成后发布 `PHASE_ATTN_PLAN`（`881`），消费前按相同八行/48-worker 分工等待所有实际 plan owner（`885`）。`phase_sync.py:94–98` 和 `signals.py:42–43` 在 notify 前维护缓存并排空本核流水，`phase_sync.py:140–154` 在等待全部 owner 后执行 fence/cacheinvalid/fence。AIC 同样等待该 plan（`cube_schedule.py:379`），原/压缩 cache 写回等待仍在（`vector_schedule.py:886–887`）。删除地址 guard 没有改变任何一侧的事件谓词。

这个结论依赖现有完整入口契约：本次调用的 Q/L/positions/页表保持只读；缓冲不发生未声明别名覆盖；工作区按单图/单流顺序使用、代际不过期（`entry.py:43–52`）。独立给 helper 注入外部或陈旧 indices，或在 plan 后并发改表/容量，不能使用此结论。候选注释已明确该限制；当前调用链未发现此类路径。

## 既有证据与一个文档问题

MASTER 为主仓 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`：

- `15448–15471`：原 plan 快路径的 32 个 CPU 边界用例，以及 `task_20261017_191833_66770729044` 的三组真实页表 probe，覆盖 T13/mixed/dummy、空洞、越界页和短表。原产物位于 `dynamic_full_pipeline_20261017/perf_h131072_b16_plan_fast_v1`。
- `15985–16020`：metadata 开关的几何边界和真实重放；任务 `task_20261017_225635_12278994621`、`task_20261017_225636_122852430692`。不能把这些旧测试直接标成新 guard 候选的设备通过。
- `16260–16283`：物理行 plan 候选的整型/页表边界地址 probe。该候选改变了 plan，不用它代替本次逻辑索引路径的证明。

非阻塞文档问题：未改动的 `attention_plan_stages.py:321–322` 仍称页表列/物理页保护由实际 gather 逐地址执行。若采纳候选，应更新为“压缩分支复用同次 plan 的检查，SWA 保留现场检查”，避免后续读者误判安全责任。审阅中没有修改该文件。
