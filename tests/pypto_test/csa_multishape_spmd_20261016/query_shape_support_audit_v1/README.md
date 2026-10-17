# CSA 的 query 数泛化：当前限制、可复用部分与最小安全路径

结论：**当前整层 PTO CSA 的请求语义确实限定 uniform S6，限制同时存在于入口、数学核与冻结 IR，不能只解除 fixture/入口检查。**
这应作为当前实现边界，而不是把最终目标缩减为 S6。HC_pre 提供的是依据工作量选择策略的设计思路，不能照搬其具体阈值。
本文把历史/可见 KV 长度记作 L，把本轮每请求 query 数记作 S；二者不是同一个 seqlen。
S6 的各个 B/L 是否已验证仍需分别看现有记录，本审计不扩大其设备通过范围。

只读取本 worktree 的 production、冻结逻辑 IR 与测试代码；新增文件都在本目录。
没有导入 PyPTO、没有修改 S 常量、没有重新 lowering、没有执行设备、没有改运行中任务的 helper。
`audit_query_shapes.py` 可重现 AST 检索及纯整数反例；结果在 `source_inventory.json`、`counterexamples.json`，日志在 `audit.log`。

## 1. 入口与 fixture 限制

| 位置 | 当前合同 |
|---|---|
| `config.py:193`～`:198` | DECODE_BATCH=64，DSPARK_SPEC_TOKENS=5，DECODE_SEQ=6，容量 DECODE_TOKENS=384 |
| `service_config.py:8`、`:27`～`:30`、`:75` | QUERY_TOKENS=6；图重放要求 uniform_decode 且实际 T=B×6；要求 dspark 的 speculative tokens=5 |
| `service.py:33`～`:53` | 整层服务要求 T 整除6、每请求六行，metadata.max_seqlen_q=6；补位请求另按 seq_lens=0 屏蔽 |
| `native_adapter.py:250`～`:269` | T=B×6，实际 token 数也整除6；绑定的是原 Native Tensor，不为其他 S 重排 |
| `N/shape_specialization.py`、`N/range_policy.py` | 当前入口要求 S6、T≤384；尚未提供其他 query 数数学体 |
| `tests/pypto_test/dsv4_csa_single_layer.py:95`～`:99`、`:143` | fixture 直接构造 T=B×6、bounds=b×6、length=H+6、positions=H+arange(6)、max_query_len=6 |
| `N/device_compare.py:490`、`:657`～`:667` | 测试也按 T=B×6、L=H+6 绑定 |

这些限制需要随泛化同步更新，但它们不是唯一阻碍。

## 2. 数学 kernel 中真实的 uniform-S 依赖

以下路径均相对 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa/`。

| 模块/源码位置 | 依赖 S 的行为 | 对泛化的影响 |
|---|---|---|
| `decode_indexer.py:407`、`:463`、`:569`、`:691`、`:711`、`:977`、`:1199`、`:1503`～`:1514` | query//S 找请求；query_count//S 扫描请求 | 改 query 数必须同步改变所有生产者、消费者及 Top-K merge；mixed query 不能用固定除数 |
| `decode_indexer.py:589`～`:633`、`:679`～`:703` | 每组 G 个 query 共用 Key，coefficients/通信区 shape 依赖 G，循环按 T//G | 组不能跨请求；尾组必须有明确定义，不能简单用 floor 丢弃 |
| `decode_indexer.py:967`～`:1028`、`:1180` | balance_leaves 长档每个 AIV 负责 G//2 个完整 query | **奇数 G 丢最后一条 query**；S5 整请求分组会只处理4条 |
| `decode_indexer.py:417`～`:429`、`:463`～`:470`、`:1625`～`:1642` | merge 依据长短档/分组策略决定 leaf 划分和每 query 一根或两根 Top-K root | 改分组必须同时更新 merge，不能只改 score 端 G 或关闭 balance_leaves |
| `decode_sparse_attn_csa.py:266`、`:300`、`:487` | 有效块规划、bias、QK/PV 均用 token//S 找页表和 seq_lens | uniform 其他 S 需新参数/特化；mixed 需要边界映射 |
| `decode_sparse_attn_csa.py:136`、`:354`～`:368` | RoPE tile 行数等于 S；原 S6 无请求尾块 | 需重新生成 tile/尾块；它是可调整的局部性能策略，不应反过来限定整个接口 |
| `compact_metadata.py:26`～`:33` | offsets 生产者已经读取每请求 begin/end | 这一部分本来支持不同 query 长度的边界计算，可复用 |
| `compact_metadata.py:52` | compact RoPE 消费者仍用 token//DECODE_SEQ | 生产者能处理 ragged，不代表消费者能处理；必须改请求定位 |
| `decode_compressor_ratio4.py:176`～`:197`、`:322`～`:336` | pool/commit 用 s_dim=T//B、token=request*s_dim+local | 可以表达其他 **uniform** S，不能表达 mixed S |
| `decode_compressor_ratio4.py:405` | cache 写入用 token//S | 需随请求映射一起改 |
| `decode_indexer_compressor.py:183`～`:196`、`:270` | 同样由 T//B 得到 uniform 的 pool/commit 范围 | mixed S 必须改为 begin/end |
| `decode_indexer_compressor.py:112`、`:293`～`:301`、`:475`～`:489`、`:518`～`:531` | 每请求 boundary 行容量 ceil(S/4)，first_pos=request*S，local_token<S | 支持其他 S 需容量/地址/有效行一起变化，mixed 则需逐请求真实边界 |

纯 token 行的 HC_pre、HC_post、Q/QR/KV 投影、RoPE 算术、O 投影多数按 T 分块，
数学本身不需要知道每请求有几条 query。例：`hc_pre.py:82`、`qkv_proj_rope.py:397`、`:531`、`:729`。
这些部分可作为泛化的复用主体，但仍要核对尾块与 workspace 容量，不能直接宣称所有 T 已验证。

## 3. 不能直接改 S 的具体反例

反例只代入当前源码的地址/分配公式，不执行数学或设备：

1. **保留冻结 //6，输入 uniform [5,5]。**
   query index5 已属于请求1，但 //6 得到请求0，使用错误长度和页表。
2. **全局把 S 改5，却保留 pair 分组。**
   连续 pair [4,5] 横跨两个请求；源码只从该组首 query 求一次 batch_idx，
   因此共用 Key/页表不成立。T 为奇数时 T//2 还会遗漏最后一行。
3. **改成整个 S5 请求为一组，却保留长档 balanced 双 AIV 分工。**
   两个 AIV 各处理 floor(5/2)=2 行，覆盖0～3，遗漏第4行。
   G1 在同一分支会覆盖零行，所以“先退成单 query”也不能保留此分支不动。
4. **mixed [2,4,6]，使用 pool 的 s_dim=T//B=4。**
   token2、3、6、7 被分给错误请求；与是否有 device query_start_loc 无关，消费者必须真正使用它。
   同为 T6 的 [2,2,2] 与 [4,2] 也不能靠 T 单独描述请求语义。
5. **仅令 config 的 S=5、B容量仍64。**
   DECODE_TOKENS=320，O 投影现有容量按16行对齐仍为320，
   不满足 `decode_o_proj.py:168` 的 128-row 整除断言。
   S1 对应64也不满足；不能假设改全局常量后模块一定能加载。

容量、数学策略、query 语义必须分开：token 容量不应被简单绑成 B上限×固定 S，
而应根据各阶段对齐/最大工作量配置并核对共用布局。

## 4. 为什么必须从符号源码重新生成

冻结文件 `F/references/symbolic_ir/10_after_OutlineIncoreScopes.py` 已将 S6 编译进来：

- 第520/545行：attention 请求定位 `//6`。
- 第746～750行：indexer cache 写回 `request*6`、`local_token<6`。
- 第942～944行：coefficient 分组 `//6`、`*6`。
- 第1614、1673～1702行：请求扫描/组循环/请求定位固定6，Cube query tile 为 `[384,128]`。
- 第2877～2907行：双 AIV 分组已经变为每 lane3条 query。
- 第13659/13689/13758行：compressor 仍保留 T//B，但其他消费者并未同步变动态。

`F/generated/core_bodies_ir.py` 同样保留 `//6`，第6281～6297行是 `[6,64]` RoPE tile，
第6405行是 QK/PV 的 `qk_t//6`。
AST 机械检索记录了逻辑 IR 80 处、核体86处以 literal6 为直接操作数的算术。
这只是定位清单，**不是80/86个独立bug**；某些6是分片倍数或其他调度参数，不能批量替换。

新的 query 策略会改变类型 shape、constexpr 展开、AIV分工、通信区域、Top-K root布局及事件路径。
必须在独立源码/配置上下文从数学源码重新 symbolic lowering，重新提取 logical IR 与 core catalog，
再生成 SPMD 的依赖和调度。仅改 `build_variant` 的 S6 guard、描述符或生成文件中的数字均不成立。
现有 S6 静态归档与设备证据应完整保留；新目录独立生成，不能让其他会话的正在运行任务加载变化中的数学模块。

## 5. 可复用的 Native 边界 metadata

Native builder 本来保留完整 TND 请求边界：
`vllm_ascend/attention/dsa_v1.py:950` 获取 query_start_loc，`:974`～`:980`
按相邻边界计算每请求 query 数、max_seqlen_q 和 start_pos，`:1156`～`:1163`
把边界装入 decode metadata。
本 PTO adapter 已在 `native_adapter.py:318`～`:320` 传入 compressed/indexer 的原 device query_start_loc，
但目前主要用于 compact offsets，没有覆盖 dense query→request 的所有消费者。

因此 mixed query 的必要信息不必重新在 Host 造一套。可以在确认两组边界都对应同一原始 token 顺序后，
复用其中一份已有 device bounds，传到实际需要的 Indexer、attention 和 compressor 消费者。
这说明数据来源可复用；不等于当前 PTO 核或整条 Native 路径已经由本次验证支持 mixed query。

## 6. 最小安全泛化路径

1. **把接口语义与性能策略分开。** 显式描述 token 总数 T、请求数 B、uniform S 或 device query bounds、
   KV长度/容量；调度策略按工作量选择。七个 S6 点保留为回归点，不作为支持白名单。
2. **先建立 uniform 的多 S 数学模板。** 使用明确的 query constexpr/工厂配置传播到相关模块，
   不修改共享全局 `DECODE_SEQ`；对 S5 等奇数同时解决组尾、AIV不等宽query分配及输出root合并合同。
   分组 G 必须保证每组属于同一请求，未整除部分有有效mask或专门尾组。
   并行 lane 可以不等宽；关键是每条有效 query 恰好处理一次、数据同步双方协议一致。
3. **独立配置容量。** 核对 O投影128行对齐、Indexer coefficient组容量、通信缓冲、boundary压缩行容量；
   不能让 S变化隐式打破固定容量断言或 stage 间共享layout。
4. **重新 symbolic lowering 后重建单SPMD。** 复用 tokenwise 数学、缓存可见性协议、源码DAG构建和区间入口的框架；
   新数学tile/成员所有权与等待细化要重证，S6/B16 的精细依赖或实测成本不得冒用。
5. **mixed query 是另一项实际语义工作。** 把所有 `token//S`、`request*S`、`T//B` 改为bounds派生请求区间；
   Indexer按请求内部划组，compressor按真实begin/end遍历，padding/空请求也使用明确边界。
   可让请求级worker天然知道请求号，或在融合SPMD内部准备映射；不预设必须再增加外部适配kernel。
6. **验证跟随语义。** 先用最小CPU所有权/尾组反例、重新lowering，再做必要的单卡正确性；
   uniform奇数 S、不同请求长度分布、压缩边界与padding是不同检查维度。
   性能再按新 T/query 工作量选点；不能在尚无新数学体时把入口放宽后标“动态shape完成”。

本轮只交付上述边界审计和可复现反例，不实施新 S，不扩大性能或正确性结论。
