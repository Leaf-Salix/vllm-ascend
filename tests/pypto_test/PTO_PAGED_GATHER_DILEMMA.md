# PTO 表达不了「分页交错布局下的整块 GM→L1 搬运」

面向 PyPTO 团队的问题陈述。全部结论均有实测依据，路径与报错原文见第五、六节。

## 一、一句话

Native 的分页 KV cache 把**一个 block 的多个分量交错摆在同一页内**，于是任何单个分量在
block 之间都不连续。PTO 的 `pl.gather_row` 要求「源偏移按源视图的行计」且
「目的块的列宽不得超过源视图的行宽」，两条合起来等价于要求
**该分量的页跨度必须是它自身行宽的整数倍**。我们的 indexer key 行宽 128 字节、
页跨度 4160 字节（`4160 mod 128 = 64`），不满足，于是无法把键直接搬进 L1，
只能先落 UB 再 `aic_gather` 跨核回传。该 kernel 因此比 Native 慢 **2.2 倍**，
是我们与上游参考实现之间剩余性能差距的 **88%**。

**补充（第九节）：存在一条不改 PTO 的便宜解**——页跨度不整除只是因为
indexer 用了全局 block_size=32；它自己的后端声明的是 128，而框架已有按后端声明
取块大小的机制，只是 indexer 的分配分支没走。B=64 或 128 时页跨度即整除 128。

硬件层面这件事是做得到的：MTE 的源与目的是两个独立描述符，可以「读 4096 个连续字节、
写成 L1 的 32×128 分形块」；而我们每页的 32 行键**本来就是 4096 个连续字节**，
连跨度都不需要。卡住的纯粹是 PTO 这层 API 的形状校验。

## 二、场景

DSV4 Flash 的 CSA decode 融合算子（vLLM-Ascend 集成），配置 b=16 / S=6 / TP1 / EP-DP16 / seqlen 8k。

`indexer_score_topk_leaf` 这个 MIX 任务要为每个 query 从 paged indexer key cache 里
取 384 个候选 token 的 INT8 键（每个 128 维），交给 Cube 做
`[64,128] × [128,384] -> [64,384]` 的 INT8 矩阵乘，再在 Vector 侧做反量化、加权与规约。

单次耗时对照（同配置、同一份 cache）：

| 实现 | 单次 | 说明 |
| --- | --- | --- |
| Native `VllmQuantLightningIndexer` | **60.9 µs** | 3,836.9 µs / 63 次，手写 CCE |
| 上游 pypto-lib `indexer_score_topk_leaf` | ~64 µs | 独立 benchmark，自己分配非交错张量 |
| 我们（PTO + Native 的 cache） | **~134 µs** | 键绕 UB + `aic_gather` 跨核 |

## 三、Native 的存储布局（实测，非推断）

在真实运行中抓取的视图元数据（`tests/pypto_test/offline_pd/observer.py` 的 `argdump`）：

```
indexer_key  : shape [18879, 32, 1, 128]  strides [4160, 128, 128, 1]
               storage_offset 1638400  dtype int8    is_contiguous False
indexer_scale: shape [18879, 32, 1,   1]  strides [2080,   1,   1, 1]
               storage_offset  821248  dtype float16 is_contiguous False
```

两者同一块 storage；key 的字节偏移 1638400，scale 的字节偏移 `821248 × 2 = 1642496`，
差值正好 4096。即每页布局为：

```
page p 的字节区间 [p*4160, (p+1)*4160)
  ├── [0, 4096)     32 个 token 的 INT8 键，每 token 128 字节，连续
  └── [4096, 4160)  32 个 token 的 FP16 反量化系数
```

**为什么这样摆。** 造出该布局的是 vLLM-Ascend 的通用重整代码
（`vllm_ascend/worker/model_runner_v1.py`，`_reshape_kv_cache_tensors` 的辅助段）：

```python
num_element_per_page = page_size_bytes // dtype_size
target_stride = (num_element_per_page, *stride[1:])   # 用页跨度，不是自然跨度
tensor = torch.as_strided(raw_tensor.view(dtype), shape, target_stride,
                          storage_offset_bytes // dtype_size)
storage_offset_bytes += stride[0] * dtype_size        # 下一个分量紧跟在页内
```

它把一个 block 的**全部分量**摆进同一个 `page_size_bytes` 的页，使
「一个 block id → 一段连续内存」成立。分页 KV 的 block 分配、拷贝、换出、
prefix caching、KV transfer 注册都因此只需操作单段连续区间；
同仓库 `_allocate_sparse_c8_indexer_tensors` 的注释还指出，
两个视图能被 `register_buffer` 合并成一段注册内存，减少 HCCL / Mooncake 的注册数量。

**这是合理的设计，不是疏忽。** 代价仅仅是：没有任何单个分量在 block 之间连续。
Native 不受影响（手写 CCE 自行计算地址并下发 MTE），
上游 pypto-lib 的独立 benchmark 也不受影响（它自己分配非交错张量）。
只有「要用 PTO 的张量视图去读这块 cache」的集成路径被卡住。

## 四、上游 pypto-lib 的写法，以及它成立的前提

```python
kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM])
...
kv_i8 = pl.create_l1([SCORE_TILE, IDX_HEAD_DIM], pl.INT8)
for page in pl.unroll(SCORE_TILE // BLOCK_SIZE):          # 12 页
    physical_row = physical_block * BLOCK_SIZE
    kv_i8 = pl.gather_row(kv_i8, kv_cache_i8_flat,
                          [page_begin, 0], [physical_row, 0],
                          [BLOCK_SIZE, IDX_HEAD_DIM])      # [32, 128]
score_i32 = pl.matmul(query_vector, kv_i8, out_dtype=pl.INT32, b_trans=True)
```

12 次整页 DMA 直接 GM→L1，Cube 就地读，**Vector 全程不碰键**。

它成立的唯一前提是：上游的 `idx_kv_cache` 是
`[blocks, BLOCK_SIZE, 1, IDX_HEAD_DIM]` 的**独立连续分配**，
因此 `reshape` 成 `[blocks*32, 128]` 是合法的零拷贝视图。

我们的 cache 是交错的，这个视图不存在——元素数对不上，键行跨页也不连续。

## 五、我们逐一实测过的七条路径与确切报错

| # | 尝试 | 拒绝方（阶段） | 原文 |
| --- | --- | --- | --- |
| 1 | `create_l1([SCORE_TILE//BLOCK_SIZE, INDEXER_KEY_BYTES])` 即 `[12, 4096]`，按页搬字节后 reshape 成 `[384,128]` | codegen | `a Mat tile of physical shape [12, 4096] and dtype int8 must be a whole number of 16x32 fractal boxes, but its row extent 12 is not a multiple of 16` |
| 2 | 行数补到 16：`create_l1([16, 4096])` 再 reshape `[512,128]` 取前 384 行 | ptoas | `'pto.subview' op boxed layout subview offsets must be multiples of inner shape`。**语义上也不成立**：L1 按 16×32 装箱，两种形状的字节排布不同，reshape 不是零拷贝视图 |
| 3 | `create_l1([IDX_HEAD_DIM, SCORE_TILE], transpose=True)` 逐行 DN2ZN 装填 | ptoas | 同上。INT8 的 ZN 分形内形状是 32，落点列号须为 32 的倍数，而逐行装填的列号是 0..31 |
| 4 | kernel 内 `pl.slice` 出键区再 `pl.reshape` 成 `[blocks, 32, 128]` 三维分页视图 | **运行期** | AICPU 异常 `507018`。用「视图建好但不使用」的变体隔离，确认崩在视图构造本身。根因：`pl.reshape` 文档要求视图是「缓冲的连续前缀」，而我们切出的键区只是**每行**的前缀 |
| 5 | 主机侧构造三维分页视图（`key.squeeze(2)`，strides `(4160,128,1)`）作为根入参传入 | 运行期 | `Parameter 'idx_key_pages' requires a contiguous strided tensor`。PTO 的根入参必须完全连续 |
| 6 | `pl.paged_gather(src, indices, block_table, block_size=1, size=4096, ...)` | — | 文档写明返回形状为 `[max_indices, size]`，即 `[12, 4096]`，与 ①② 同一堵墙 |
| 7 | 解耦寻址：源换成一维连续视图、`src_offset` 直接给字节地址 `phys * page_bytes` | codegen | `tile.gather_row offsets and shapes must have at least 2 elements`。改用 `[blocks*65, 64]` 的 64 字节行视图（`4160 = 65×64`，第 p 页起始行正好 `p*65`）后，ptoas 报 `'pto.partition_view' op size at dim 1 (128) exceeds static source dim (64)` |

第 4、5 条尤其值得 PTO 注意：**lowering 阶段全部放行，问题直到 codegen 或运行期才暴露**。
第 4 条更是编译期毫无提示、设备上直接 AICPU 异常。

## 六、归纳：三条约束构成封闭证明

从上述报错可以提炼出 `pl.gather_row` 的三条约束：

1. `shapes` **同时**约束「源读多少」和「目的写多少」，两侧被绑死；
2. `src_offset` 至少两个元素，**不能**用扁平字节地址寻址；
3. `shapes[1] ≤ src.shape[1]`，**不能**跨源视图的行读。

由 (1)(3)：目的要 `[BLOCK_SIZE, IDX_HEAD_DIM] = [32, 128]` 的 L1 块
⟹ 源视图的行宽必须 ≥ 128；由 (2) 与「源偏移按行计」：页起始必须落在整行边界
⟹ 页跨度必须是源视图行宽的整数倍 ⟹ **页跨度必须是 128 的整数倍**。

我们的页跨度是 4160，`4160 mod 128 = 64`。**无解。**

这不是「还没想到别的写法」，而是在现有 API 的形状与寻址规则下不存在解。

## 七、硬件本来做得到

Ascend 的 MTE 搬运指令，源与目的是两个独立描述符：
源给「起始地址 + 每次传多少字节 + 传几次 + 次间跳多少」，
目的给「片上落成什么形状」。因此硬件可以直接表达
「从 GM 读 4096 个**连续**字节，写成 L1 里的 32×128 分形块」。

对本场景而言，**连跨度都不需要**：一页内的 32 行键就是 4096 个连续字节，
一条指令即可；页与页之间由 block table 给出各自的基地址，本就是 12 次独立搬运。

所以 Native 的手写 CCE 能用同一份 cache 跑到 60.9 µs；
差距不在硬件能力，而在 PTO 这层把源形状与目的形状绑在了一起。

## 八、给 PTO 的可能方向（按侵入性从低到高）

1. **放宽 `gather_row` 的源寻址**：允许 `src_offset` 为一维（扁平元素/字节偏移），
   `shapes` 仅约束目的块形状。本场景只需这一条即可解开——
   源是完全连续的 `[blocks, page_bytes]` 张量，扁平偏移 `phys * page_bytes` 合法且精确。
   路径 7 表明这个写法能通过 lowering，只是 codegen 要求 `≥2` 个元素。

2. **放宽 `shapes[1] ≤ src.shape[1]`**：当源视图在内存上连续时，允许跨行读。
   `[blocks*65, 64]` 视图配 `shapes=[32,128]` 在字节意义上是良定义的。

3. **增加分页张量视图类型**：让 `pl.Tensor` 能描述
   `(page_count, page_stride_bytes, payload_offset, payload_shape)`，
   即「按页跨度定位、页内按自然形状读」。这正是 Native 的 `PA_BSND` 语义，
   也是分页 KV 场景的通用需求，不止本算子。

4. **放宽根入参的连续性要求**：至少支持「最后一维连续、其余维带 strides」。
   路径 5 的 `key.squeeze(2)` 恰好属于这一类，且 PTO 内部的
   `make_tensor_view` 本来就带 strides 参数。

5. **把校验前移**：路径 4、5 在 lowering 阶段完全放行，到 codegen 或运行期才失败；
   路径 4 在设备上表现为 AICPU 异常 `507018`，没有任何可定位信息。
   即使不放宽能力，也建议把「reshape 必须是缓冲的连续前缀」「根入参必须连续」
   这两条在 `lower()` 阶段就报出来。

## 九、最便宜的一条：让页跨度整除，PTO 一行不用改

第六节的封闭证明里，「无解」只取决于一件事：**页跨度不是分量行宽的整数倍**。

```
page_size_bytes = block_size × (head_size × 1 + scale_dim × 2) = 130 × block_size
需要 130·B ≡ 0 (mod 128)  ⟺  2B ≡ 0 (mod 128)  ⟺  B ≡ 0 (mod 64)

B = 32（当前实际值）  → 4160    mod 128 = 64   ✗
B = 64               → 8320  = 65 × 128       ✓
B = 128              → 16640 = 130 × 128      ✓
```

B 取 64 或 128 时，`[blocks × (page_bytes/128), 128]` 就是合法的零拷贝视图：
第 p 页的键正好落在第 `p × page_rows` 起的连续若干行上，scale 在其后。
第四节里上游那段代码可以**原样照搬**，PTO 侧一行不用改。

更关键的是：**框架里这条路已经存在，只是 indexer 分支没走。**
`vllm_ascend/worker/model_runner_v1.py:4204` 的通用 `AttentionSpec` 分支里有

```python
if hasattr(attn_backend, "get_supported_kernel_block_sizes") and self.use_hybrid_blocks:
    block_size = attn_backend.get_supported_kernel_block_sizes()[0]
    block_size_chunk = current_kv_cache_spec.block_size // block_size
    kv_cache_shape = attn_backend.get_kv_cache_shape(num_blocks * block_size_chunk, block_size, ...)
```

而 `AscendSFAIndexerCache.get_supported_kernel_block_sizes()` 返回的正是 **`[128]`**。
但 `AscendSFAIndexerCacheSpec` 的分配分支（同文件 4082 起）没有这段，
直接用了全局的 `current_kv_cache_spec.block_size`（=32），于是页跨度成了 4160。

**⚠ 2026-09-25 实测更正：block size 这条走不通。**
vLLM 在 `v1/core/kv_cache_utils.py:1456` 用 `block_size=uniform_block_size`
**统一所有 KV cache group 的块大小**，改单个 spec 的 `block_size` 会被覆盖
（实测：设成 64 后 key 每页仍是 32 token）。要走这条得改 vLLM 的统一块大小机制，
代价远超收益。

**改为建议这条：用 `page_size_padded` 把页跨度对齐到键行宽。**
vLLM 的 `KVCacheSpec` 基类本来就有这个字段，`page_size_bytes` 会优先返回它：

```python
page_size_padded: int | None = None
...
if self.page_size_padded is not None:
    assert self.page_size_padded >= real_page_size
    return self.page_size_padded
```

而 `vllm_ascend/core/kv_cache_interface.py` 的 `AscendSFAIndexerCacheSpec`
**覆盖了 `page_size_bytes` 并直接返回 `real_page_size_bytes`**，把基类这个机制屏蔽了。
若让它像基类一样尊重 `page_size_padded`，把页从 4160 填到 **4224 = 33×128**：

- 块大小不变、块表不变、调度与 block id 共享不受影响；
- 键仍在页内偏移 0 起的 4096 连续字节，scale 在其后，只是每页多 64 字节空洞；
- 显存代价 **+1.54%**，且只在 indexer 这一块 cache 上；
- 之后 `[blocks*33, 128]` 即合法连续视图，第 p 页的键正好是第 `p*33` 起的 32 行，
  第四节上游那段整页直搬 L1 的代码可以原样用。

这正是「只加一点描述信息 + 多占一点内存」的形态，不改语义、不改 PTO。

实测提示：本轮试做时改了 `AscendSFAIndexerCacheSpec.page_size_bytes` 与其
`merge()`（后者原本不透传 `page_size_padded`），但分配层拿到的跨度仍是 4160。
框架里存在两条 KV cache 重整路径，实际生效的是
`model_runner_v1.py` 中用 `torch.as_strided(..., (page_size_bytes // dtype_size, *stride[1:]), ...)`
的通用那条；要落地需确认 `page_size_padded` 能贯穿到该处。这是留给框架侧的收尾。

（原文保留：让 indexer 分支尊重后端声明的 kernel block size。）
它不改 PTO、不改 cache spec 结构、不拆分配、不增加 KV transfer 的注册段数，
用的是框架自己已有且已在别处生效的机制。

需要连带核对的：indexer block table 的列宽与 `IDX_MAX_BLOCKS` 推导、
PTO 侧 `config.py` 里 indexer 路径用的 `BLOCK_SIZE` 常量（当前与主 KV 共用 32）、
以及 Native `npu_vllm_quant_lightning_indexer` 在 block 128 下的行为。

## 十、退一步：改存储契约（代价更大，不优先）

把 indexer 的 scale 提成**独立的 cache spec**，使 key 的页内只有 key（4096 字节/页），
key 即跨 block 连续，`[blocks*32, 128]` 视图成立。vLLM 本来就支持多个 cache group
共享 block id（`AscendSFAIndexerCacheSpec` 的注释即写明它与 MLA cache 在同一
UniformType group 内共享 block id）。

代价：

- 显存记账（`page_size_bytes`）与分组结构要改；
- KV transfer 的注册从 1 段变 2 段，正是上述注释想避免的；
- 所有假设 key/scale 相邻的代码要复核，包括 Native 的 `indexer_quant_scatter` 写入路径。

即：为了让**一个** PTO 算子能用上整页直搬，要改动框架侧的分页存储契约，
并让 Native 路径一起承担代价。这个取舍是本文档希望 PTO 一方一并评估的。**但这条应排在第九节之后**——
若 block size 一条即可解，就不必动存储契约。

## 十一、复现与证据

- 存储布局实测：`offline_pd/run.py argdump`，产物 `csa_args_meta.json` 的 `indexer_storage` 字段；
- 单算子单卡回放与泳道：`tests/pypto_test/dsv4_csa_single_card_bench.py --swimlane 4`，
  一轮约 40 秒、占 1 张卡，逐任务 kernel 时长可与上游泳道直接对照；
- Native 逐算子耗时：`results/release_csa_perf_8k_20260924/profile_comparison.json`；
- 七条路径的完整结论与清单口径：`DSV4_FLASH_CSA_TASK_CHECKLIST.md` 的 T2.14。

## 附：该项在整体中的位置

本算子之外，PTO 版本相对上游参考实现已经追平或反超：
`qk_pv`（136.3 vs 140.2 µs/次，更快）、`weights_proj`、`qr_proj_matmul`、
`idx_qr_dequant_rope`、`compress_state_commit`、`qr_hadamard_matmul`、
`rope_cs`、`csa_cache_writeback`。

整体 kernel 合计已由 45,497 µs（1.70×）降至 33,287 µs（1.24×，上游 26,822 µs）。
剩余 +6,465 µs 中，`indexer_score_topk_leaf` 一项占 **88%**。
关键路径分析显示依赖决定的静态下界为 0.67 ms，**低于上游 0.728 ms 的窗口跨度**，
即只要这一项解开，追平乃至反超是可达的。
