# Attention 压缩候选：复用 plan 的地址合法性结论

本候选冻结正式提交 `e58169ecf381a0228774070a25838feaf3d3940c` 的全部 33 个
Python 文件，位于 `source_before/`；`source_candidate/` 只有
`attention_stream_stages.py` 不同。没有修改正式仓库、依赖或已有候选。

目标是删除压缩 gather 热循环内两层已由同次 plan 保证为真的条件：
`column < table_columns` 和 `0 <= page < cache_pages`。保留 `q_active`、
`ridx >= 0`、原逻辑索引含义、原页表 Scalar 读取、原行顺序、零 KV、SWA、
所有数学和事件。没有把索引改成物理行，没有新工作区、额外 plan 计算或缓存。

## 为什么这两个条件冗余

这个结论只适用于本完整入口已有的合同，不能把任意外部 Top-K 直接当成 plan。

1. `vector_schedule.py:865–916` 的 plan 和 consumer 使用相同的
   `idx_bounds / idx_lengths / positions / compressed_table`，以及相同
   `main_cache`；plan 的 `compressed_pages` 实参严格等于
   `dim(main_cache,0)//32`。两份 Q/L、页表和位置在本次调用期间只读，不允许
   异步改写或与业务输出非法别名。Native 可以在已完成调用之间更新页表内容。
2. plan 每轮覆盖实际 T 的全部 512 个 indices。非活动、无合法 owner、dummy、
   陈旧 causal position 都不会设置页合法位，因此输出为 -1。consumer 保留
   原 geometry 检查与 `ridx>=0`，负索引留在本轮初始化的零 KV 中。
3. plan 先按原 FP32 数学产生 `c_page_indices`。页检查针对这个最终 INT32
   地址，不能改查原始 Top-K。`c_mask` 和页合法位只可能是 0 或 1；只有页合法位
   为 1 才能得到非负输出，此时最终写出的索引与已检查的 `c_page_indices` 相同。
   数学涉及的有限 INT32 值不会引入 NaN。
4. 快速路径只有 `L<=2^24`、`0<physical_pages<=2^24` 且页表足够长时才可返回 1。
   它验证前 `ceil(floor(L/4)/32)` 列全部满足物理页范围；plan 随后仍验证最终
   `0<=ridx<upper`。此区间位置及 `position+1` 能由 FP32 精确表达，
   `upper=floor((position+1)/4)<=floor(L/4)`，故 `ridx//32` 必在已验证前缀。
   前缀长度为 0 时没有非负输出，不存在未验证列被消费的例外。
5. 页表短、存在负页/越界页、L 或物理页数超出上述 FP32 快速范围时，保留原
   INT32/INDEX 逐候选回退。回退直接检查 `c_page_indices` 的列和物理页；
   极大值可能经历原数学舍入，但最终被消费的地址就是这个已检查的值。
   本候选不宣称修改了原大值数学语义，只复用已完成的地址安全判断。
6. consumer 前仍等本代全部真实 plan owner：`publish_phase` 写回并发布，
   `wait_row_owners` 等待后执行 fence/cacheinvalid/fence。原 cache 写回与
   RoPE 阶段等待同样保留。不能用时间间隔、Scalar cache 偶然命中或阶段 phase
   替代原逐槽 FFTS；本候选完全没有改变这些协议。

因此合法调用中，非负 indices 已蕴含被删除的两个条件。若输入页表在 plan 与
consumer 之间变化、cache 容量或请求映射不一致、或消费者接入过期 indices，
该证明不成立；这些不是原完整入口允许的调用方式，也不能把本候选包装为独立
接受任意 indices 的安全 gather API。

## 历史排重与范围

已只读搜索统一日志及 `csa_multishape_spmd_20261016` 的 Markdown 历史，未发现
同一“保留逻辑 indices 和原页读取，仅删除消费者两层检查”的独立尝试。
已有相关方案如下，不能混为本次结果：

- `qk_address_cache_v1`：本核缓存地址，额外跨 tick Tile/工作量。
- `query_metadata_v1`：缓存请求 owner/position，减少 Q/L 反查；本候选保持关闭。
- `parallel_20261018/physical_plan_evidence`：改内部 indices 为物理行并增加 plan
  工作；本候选不改 plan 及其 GM 输出。
- 原有 plan 快速页前缀检查：已经采用；这里复用其结论，不重复实现它。

`attention_plan_stages.py` 保持冻结原文，包含其当时对 consumer 再检查的说明；
候选的新调用合同已写在被修改的 stream 模块总览及采样节点注释中。若最终采用，
可再同步这处历史 docstring，当前不为注释扩大执行源码差分。

## 源级检查与复现

`make_candidate.py` 从指定 Git 提交读取 33 文件，并要求与当前正式源码逐字一致。
AST 只允许展平上述两个 If；忽略 docstring 后必须与候选整个模块相同。
已通过：其余 32 文件逐字相同，只有两个 `pl.tensor.dim` 调用减少，没有新增调用；
page read、gather、零初始化、数学、有效范围和同步均未改变。
具体冻结时间和结果见 `source_audit.json`，完整源码差异见 `candidate.patch`。

一次完整 long 策略 CPU 符号编译使用 OA128/OB96、HC row dependencies on、
HC consumer reduce off、Attention query metadata off。没有传入 B16/T96 常量，
所有输入轴仍通过动态签名生成。命令如下：

```bash
E=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_attention_guard_elision_20261018
D=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
python "$E/make_candidate.py"
bash "$D/reproduce_cpu.sh" "$E/cpu_v1" --strategy long \
  --attention-query-metadata off --spmd-source "$E/source_candidate" \
  --controls-json "$E/controls.json" > "$E/cpu_v1.log" 2>&1
```

目录拒绝覆盖。CPU 状态与生成审计在完成后追加；目前不预记 PASS，不提交设备任务。
CPU 或数学合同证明不能代替真实 A3 精度、页表交换、保护区、图重放和完整 L1 对照。

## CPU 完成及生成物审计

唯一一轮 `cpu_v1/result.json` 已完成 PASS：58 个动态轴、130 个根入参、
single MIX 和 sync_start 保持。没有给编译器传入样例 shape，没有追加第二次编译。
源冻结在 UTC `2026-10-17T19:35:39.997099+00:00`，完成后未再改动。

`audit_generated.py` 对照已有
`parallel_20261018/hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat`
的 hook 前原始 PTO/C++。先确认该任务所归档的 33 个源文件与本次 e581
`source_before` 逐字一致，随后比较现成产物，不重编基线。

- Attention AIV 的 PTO 少 `arith.cmpi` 3 处、`arith.andi` 1 处、
  `arith.divsi` 1 处，正是列检查、物理页上下界、合并条件及页容量表达式。
  这是生成表达式数量，不是对最终 CCEC 动态指令数或时延的测量。
- 整个 AIV C++ 的 If 由 462 降为 460；AIC 仍为 167。
  Tile 数学、TLOAD/TSTORE 和本地/跨核同步调用计数全部保持。
  少两处 `pto.alloc_tile` / C++ `TASSIGN` 是删除分支后不再需要的 Tile
  别名声明，不是额外业务 copy。未出现新增搬运或数学运算。
- 原同步与数学执行顺序由完整源级 AST 比较支撑；生成调用计数另作交叉检查。
  未把单纯计数相同当作动态协议等价的唯一证据。

完整函数片上地址上界如下，两侧相同，单位为字节：

| 资源 | e581 对应已有产物 | 候选 |
| --- | ---: | ---: |
| AIV Vec | 163872 | 163872 |
| AIC Mat | 475136 | 475136 |
| AIC Left | 65536 | 65536 |
| AIC Right | 65536 | 65536 |
| AIC Acc | 131072 | 131072 |
| AIC Vec | 16384 | 16384 |

AIV Vec 上界同时经分配 pass 的 MemRef 结果核对。这里不声称所有临时量地址
逐一相同，也不将“空间未增加”换算为性能收益。详细计数、来源及局限见
`generated_audit.json`；局部实际 PTO 摘录为 `baseline_compressed_gather.pto.txt`
和 `candidate_compressed_gather.pto.txt`。再次运行审计脚本仅分析已有文本。

## 根代理提交的设备验证

源码保持冻结。根代理已使用同一 `source_before` 参考提交两例完整同进程矩阵：

- 128K B16：`task_20261018_033720_281285324271`，输出
  `parallel_20261018/attention_guard_elision_h131072_b16_v1`。
- 8K B32：`task_20261018_033720_281290823741`，输出
  `parallel_20261018/attention_guard_elision_h8192_b32_v1`。

本子任务没有提交设备或增加测试。这一节只记录已提交的任务，不预记其精度或
性能结果；是否采用由上述完整结果决定。候选不默认合入正式，不提交或 push。

## 设备结果：完整检查通过，但无完整 L1 收益

根代理已收取上述两个同进程矩阵，完整检查均 PASS。完整 L1 中位数如下：

| 档位 | 同轮参考 | guard-elision 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 128K B16 | 874.12 μs | 874.28 μs | 慢 0.16 μs，约 0.018% |
| 8K B32 | 995.36 μs | 996.02 μs | 慢 0.66 μs，约 0.066% |

因此本候选**不合入正式**。删除两个重复地址 guard 在已有两档中没有形成完整
CSA 性能收益，不据 Scalar 条件数量减少推导收益，不扩档或继续组合。保留全部
冻结源码、CPU 证据、任务 ID 与本次结论；本次收口只追加 README，未修改 runtime、
source_before/source_candidate 或已有 CPU 产物。主仓归档由根代理负责。
