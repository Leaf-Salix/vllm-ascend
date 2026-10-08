# TND 效率单因素 A/B（2026-10-08）

## 源码与实验边界

正式分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，代码基线 `ad8b69520`。
BSH 固定 `8e83adda1f5959da0d407f2d27d56f52a16466e5`。候选为独立副本，正式 kernel 未修改。
CANN9.2.0-beta.2、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1；
PyPTO3e87a843、Simplera54c0509、PTOAS0.66，沿用已构建依赖。
真实第4层权重、seed1024、合成 hidden/history，包含 HC pre + norm + CSA + HC post。
不是整模型 TPOT/吞吐，也未测 DP/EP16、EPLB 或真实 DSpark 接收率。

## 四个独立候选

| 候选 | 唯一改变 | 保留部分 |
| --- | --- | --- |
| counts | merge 读取建组任务已产出的两类组数 | 主控分派、Score/TopK算术及舍入 |
| metadata | 原建组和无效TopK初始化整体并入 offsets 任务 | 原merge计数与各请求边界 |
| qr_capacity | QR accumulator 从448恢复384行，减少2MiB | query尾部432行容量及实际矩阵行数 |
| rope | 删除 offsets→RoPE sign 的手动等待边 | 两条真实RAW、根分配及全部算术 |

第一轮为前三候选；第二轮改变基础计时排列、复测前三候选，并增加rope独立实验。
不把候选组合，不将几个因素一起改后归因给其中一个。

## 门禁与测量

- 同case同进程同卡串行，各arm独立cache、图owner及op注册。
- 生产opaque边界，named npugraph_ex；Native static+SuperKernel，PTO沿参考整体编排。
- 候选对control输出、TopK和六类cache/state必须完整逐位一致后收性能。
- compiled/raw、连续2次、100次后和恢复初态重放均一致；保护区与有效写区finite检查。
- 确定性level1/HCCL=true/PTO atomic_add=0；禁止PTO回退，BSH必须拒绝变长。
- 循环移位+反序平衡位置和先后方向；所有样本保留，不删除尾样本。
- 100次连续重放的轮均值，与恢复初态后单次重放分开报告。
- 每轮内部对拍同卡；跨任务只比较各自同场相对值，不拼接绝对基线。

## 完成状态

| case | 完成 | 物理卡 | tokens | 轮数/单次样本数 |
| --- | --- | --- | --- | --- |
| confirm-h131072-b16 | True | 0 | 96 | 42/140 |
| confirm-h131072-b4 | True | 0 | 24 | 42/140 |
| confirm-h8192-b24 | True | 0 | 144 | 42/140 |
| ragged-confirm-h131072-b16 | True | 0 | 60 | 36/120 |
| ragged-confirm-h8192-b4 | True | 0 | 11 | 36/120 |
| ragged-screen-h131072-b16 | True | 0 | 60 | 30/100 |
| ragged-screen-h8192-b4 | True | 0 | 11 | 30/100 |
| screen-h131072-b16 | True | 0 | 96 | 36/120 |
| screen-h131072-b4 | True | 0 | 24 | 36/120 |
| screen-h131072-b40 | True | 0 | 240 | 36/120 |
| screen-h8192-b16 | True | 0 | 96 | 36/120 |
| screen-h8192-b24 | True | 0 | 144 | 36/120 |

## 配对连续重放结果

单位为相对control的延迟变化百分比；负数表示加速，统计为配对差中位数。
不是单次重放结果。没有BSH变长行，不以Native回退冒充BSH性能。

| case | counts | metadata | qr_capacity | rope |
| --- | --- | --- | --- | --- |
| confirm-h131072-b16 | +0.014% | -0.483% | -0.966% | -0.875% |
| confirm-h131072-b4 | +0.672% | +0.201% | +0.185% | -0.609% |
| confirm-h8192-b24 | -0.573% | -0.408% | -0.678% | -1.221% |
| ragged-confirm-h131072-b16 | +0.095% | -0.027% | -0.173% | +0.754% |
| ragged-confirm-h8192-b4 | -2.039% | -3.644% | -1.267% | -1.296% |
| ragged-screen-h131072-b16 | +0.223% | +0.360% | -0.454% | — |
| ragged-screen-h8192-b4 | +1.401% | -2.148% | -0.332% | — |
| screen-h131072-b16 | +0.492% | +1.598% | -0.552% | — |
| screen-h131072-b4 | +1.340% | +0.191% | +1.485% | — |
| screen-h131072-b40 | +0.950% | +1.851% | +0.025% | — |
| screen-h8192-b16 | +0.971% | -0.398% | -0.072% | — |
| screen-h8192-b24 | -0.352% | -0.358% | -0.489% | — |

## 单次恢复初态均值

单位μs。这一表独立于上表，不能混用两种P95或提升率。

| case | Native | BSH | control | counts | metadata | qr_capacity | rope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confirm-h131072-b16 | 1146.331 | 970.231 | 982.397 | 982.134 | 982.495 | 971.346 | 976.486 |
| confirm-h131072-b4 | 757.784 | 654.830 | 658.250 | 669.193 | 672.940 | 667.986 | 658.447 |
| confirm-h8192-b24 | 957.610 | 941.357 | 958.347 | 956.578 | 953.097 | 955.242 | 947.840 |
| ragged-confirm-h131072-b16 | 953.540 | — | 924.235 | 922.314 | 921.164 | 923.618 | 930.749 |
| ragged-confirm-h8192-b4 | 418.843 | — | 480.785 | 480.869 | 474.160 | 474.907 | 479.176 |
| ragged-screen-h131072-b16 | 943.585 | — | 912.468 | 918.023 | 914.360 | 909.528 | — |
| ragged-screen-h8192-b4 | 427.120 | — | 459.098 | 461.924 | 449.968 | 455.113 | — |
| screen-h131072-b16 | 1141.485 | 977.355 | 983.275 | 989.853 | 999.610 | 980.035 | — |
| screen-h131072-b4 | 766.710 | 662.782 | 668.186 | 672.084 | 673.015 | 685.145 | — |
| screen-h131072-b40 | 1874.408 | 1858.057 | 1877.392 | 1898.003 | 1915.499 | 1882.961 | — |
| screen-h8192-b16 | 773.848 | 724.963 | 752.183 | 744.862 | 748.785 | 746.090 | — |
| screen-h8192-b24 | 921.551 | 928.560 | 945.566 | 942.200 | 937.986 | 936.773 | — |

## 对拍精度

逐位一致只针对候选/control，以及等长control/BSH；不等于对Native逐位一致。
继承的性能版对Native误差参见三方完整矩阵；本轮不得据候选相等宣称消除了这部分差异。

| case | 完整逐位一致的比较 |
| --- | --- |
| confirm-h131072-b16 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、rope_vs_control、control_vs_bsh |
| confirm-h131072-b4 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、rope_vs_control、control_vs_bsh |
| confirm-h8192-b24 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、rope_vs_control、control_vs_bsh |
| ragged-confirm-h131072-b16 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、rope_vs_control |
| ragged-confirm-h8192-b4 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、rope_vs_control |
| ragged-screen-h131072-b16 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control |
| ragged-screen-h8192-b4 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control |
| screen-h131072-b16 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、control_vs_bsh |
| screen-h131072-b4 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、control_vs_bsh |
| screen-h131072-b40 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、control_vs_bsh |
| screen-h8192-b16 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、control_vs_bsh |
| screen-h8192-b24 | counts_vs_control、metadata_vs_control、qr_capacity_vs_control、control_vs_bsh |

## 编译后证据与判断修正

不能把Python源代码的循环直接当成实际NPU开销：

- False merge（短历史）中的组数扫描已被DCE。counts只新增两个未在核正文使用的GM参数，没有减少实际循环。
- True merge（长历史）少两段组数扫描，改为两次GM标量读取；最大长度扫描保留。
- metadata确实少一个任务，但同时延长offsets→RoPE/QKV前置链；省任务数不自动等于降低延迟。
- QR容量减少没有减少矩阵行数，实际循环仍由bs决定，不应据此估计同比例计算收益。
- rope只删手动边；sign仅消费外部sin，metadata与sign各自真实RAW和根scope生命周期保留。
- 当前CANN trace只见整个PyPTO程序；未取得各内部task耗时，不能把端到端差异全部归某模块。

## 来源

第一轮task_20261008_144940_31773112290：7case，exit0。
第二轮task_20261008_155222_411381014439：状态以完成表和原始报告为准。
候选patch、逐文件SHA256、runner SHA、全部原始报告/profile保存在实验记录中。
全部计时样本见 [原始样本](DSV4_FLASH_CSA_TND_EFFICIENCY_AB_20261008_SAMPLES.md)。

## 决策：不能全局合入这些候选

两任务均exit0，全部12份报告complete；第一轮7档、第二轮5档，物理卡均为0。
未修改正式kernel，候选是可丢弃的独立实验副本。

| 候选 | 两轮验证支持的判断 | 全局采用 |
| --- | --- | --- |
| counts | 没有跨形状一致收益；短档原扫描已DCE；T11两轮连续结果还反号 | 否 |
| metadata | T11连续重放两轮分别快2.148%、3.644%；128K/S6结果不稳定，B40首轮慢1.851% | 否，可继续研究短变长场景 |
| qr_capacity | 128K/B16两轮快0.552%、0.966%；B4两轮慢1.485%、0.185%；B40首轮近零 | 否，可继续研究容量策略 |
| rope | 新候选第二轮三个S6连续快0.609%、0.875%、1.221%；T11快1.296%，T60慢0.754%且36轮全部慢 | 否，T60回退明确 |

rope只在第二轮新增，因此没有同形状跨任务独立复跑证据；不能把第一轮其他候选当作rope复跑。
第二轮单次均值，rope相对BSH在128K/B4、128K/B16、8K/B24仍慢0.552%、0.645%、0.689%。
QR在128K/B16单次只慢BSH0.115%，但不能将此单点推广到所有形状。
TND/control相对BSH的测量差距随口径与任务变化，必须使用同场基线，不能引用旧均值填补。

所有候选均维持当前control的八类状态逐位一致。T11长度[6,3,1,1]、T60长度
[1,2,3,4,5,6,6,6,1,2,3,4,5,6,4,2]，无外部S6扩展；BSH均拒绝。
当前性能版继承的Native数值差异没有改善，不能将“候选零新增误差”写成“Native精度已通过”。

本轮结论是：确认了部分局部收益和回退，尚未证明任何一个改动能全面抹平BSH效率差距。
后续应针对真实关键路径做测量；减少Python循环、任务数或分配字节都不能直接当成速度提升。
完整仓库format CI有既存历史归档问题；本轮仅报告修改文件的增量检查，不宣称完整CI通过。

## 候选可复现补丁

以下补丁均相对同一冻结control，不累积应用；这保证单因素定义。

### counts

```diff
--- control/decode_indexer.py
+++ counts/decode_indexer.py
@@ -472,12 +472,14 @@
     pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
     topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
     topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
+    request_group_count: pl.Tensor[[1], pl.INT32],
+    pair_group_count: pl.Tensor[[1], pl.INT32],
     multiway: pl.constexpr,
 ):
     """Merge query roots on one persistent worker per physical AIV."""
     worker = pl.tile.get_block_idx()
     query_count = pl.tensor.dim(position_ids, 0)
-    merge_request_groups = indexer_query_group_count(query_start_loc, kv_seq_lens, S)
+    merge_request_groups = pl.cast(pl.read(request_group_count, [0]), pl.INDEX)
     merge_use_request = pl.cast(merge_request_groups >= LONG_S6_MIN_REQUEST_GROUPS, pl.INDEX)
     merge_leaf_tiles = TOPK_CANDIDATES_PER_LEAF // BUFFERED_LONG_SCORE_TILE
     merge_extra_leaves = 0
@@ -485,7 +487,7 @@
         max_cache_count = 0
         for batch in pl.range(pl.tensor.dim(kv_seq_lens, 0)):
             max_cache_count = pl.max(max_cache_count, pl.read(kv_seq_lens, [batch]) // COMPRESS_RATIO)
-        query_groups = indexer_query_group_count(query_start_loc, kv_seq_lens, 2)
+        query_groups = pl.cast(pl.read(pair_group_count, [0]), pl.INDEX)
         if merge_use_request == 1:
             query_groups = merge_request_groups
         _merge_leaf_count, merge_leaf_tiles, merge_extra_leaves = indexer_long_leaf_plan(
@@ -1875,6 +1877,8 @@
                     pair_arena,
                     topk_scores,
                     topk_idxs,
+                    rgroup_count,
+                    pgroup_count,
                     False,
                 )
         else:
@@ -1889,6 +1893,8 @@
                     pair_arena,
                     topk_scores,
                     topk_idxs,
+                    rgroup_count,
+                    pgroup_count,
                     True,
                 )

```

### metadata

```diff
--- control/decode_csa.py
+++ metadata/decode_csa.py
@@ -18,6 +18,7 @@
     DECODE_BATCH,
     DECODE_SEQ,
     KV_ORI_BLOCK_NUM,
+    FP32_NEG_INF,
 )
 from .config import (
     FLASH as M,
@@ -26,7 +27,8 @@
     TP as TP_SIZE,
 )
 from .decode_compressor_ratio4 import compressor_ratio4
-from .decode_indexer import indexer
+from .decode_indexer import T_PAD as INDEXER_T_PAD
+from .decode_indexer import indexer, indexer_build_query_groups
 from .decode_indexer_compressor import indexer_compressor
 from .decode_o_proj import LOCAL_T, LOCAL_T_PAD, decode_o_proj_tp1
 from .decode_sparse_attn_csa import T_PAD, sparse_attn_csa_tp1
@@ -356,10 +358,36 @@
         # 符号那段走 SPMD，靠 deps 保证偏移先算好。
         # 元数据也是RMS/cache消费者的实际前置，必须允许其参与预派发资格判定。
         token_request = pl.create_tensor([t_dim], dtype=pl.INT32)
+        rgroup_row = pl.create_tensor([INDEXER_T_PAD], dtype=pl.INT32)
+        rgroup_rows = pl.create_tensor([INDEXER_T_PAD], dtype=pl.INT32)
+        rgroup_count = pl.create_tensor([1], dtype=pl.INT32)
+        pgroup_row = pl.create_tensor([INDEXER_T_PAD], dtype=pl.INT32)
+        pgroup_rows = pl.create_tensor([INDEXER_T_PAD], dtype=pl.INT32)
+        pgroup_count = pl.create_tensor([1], dtype=pl.INT32)
         with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_row_offsets", allow_early_resolve=True) as offsets_tid:
             cmp_row_offsets = build_compact_row_offsets(cmp_query_start_loc, cmp_seq_lens, cmp_row_offsets)
             idx_row_offsets = build_compact_row_offsets(idx_query_start_loc, kv_seq_lens, idx_row_offsets)
             token_request = build_token_request(query_start_loc, kv_seq_lens, token_request)
+            for token in pl.range(pl.tensor.dim(position_ids, 0)):
+                if pl.read(token_request, [token]) < 0:
+                    pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [token, 0], idx_topk_scores)
+                    pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [token, 0], idx_topk)
+            indexer_build_query_groups(
+                query_start_loc,
+                kv_seq_lens,
+                rgroup_row,
+                rgroup_rows,
+                rgroup_count,
+                S,
+            )
+            indexer_build_query_groups(
+                query_start_loc,
+                kv_seq_lens,
+                pgroup_row,
+                pgroup_rows,
+                pgroup_count,
+                2,
+            )

         # 向上取整分块并保留 valid_shape：t_dim = batch*6 不保证是 4 的倍数，
         # 上游 csa_rope_interleave 用的 t_dim // 4 会丢掉尾行（batch=1 时 t_dim=6 只覆盖 0~3）。
@@ -504,6 +532,13 @@
                     kv_seq_lens,
                     token_request,
                     query_start_loc,
+                    rgroup_row,
+                    rgroup_rows,
+                    rgroup_count,
+                    pgroup_row,
+                    pgroup_rows,
+                    pgroup_count,
+                    offsets_tid,
                     late_dep,
                     idx_cache_write_tid,
                     max_seq_len,
--- control/decode_indexer.py
+++ metadata/decode_indexer.py
@@ -1531,6 +1531,13 @@
     token_request: pl.Tensor[[T_DYN], pl.INT32],
     # TND：组不跨请求，组边界从这里推。
     query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
+    rgroup_row: pl.Tensor[[T_PAD], pl.INT32],
+    rgroup_rows: pl.Tensor[[T_PAD], pl.INT32],
+    rgroup_count: pl.Tensor[[1], pl.INT32],
+    pgroup_row: pl.Tensor[[T_PAD], pl.INT32],
+    pgroup_rows: pl.Tensor[[T_PAD], pl.INT32],
+    pgroup_count: pl.Tensor[[1], pl.INT32],
+    groups_tid: pl.Scalar[pl.TASK_ID],
     topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
     topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
     qh_quant_tid: pl.Scalar[pl.TASK_ID],
@@ -1539,35 +1546,6 @@
     host_max_seq_len: pl.Scalar[pl.INT32],
 ):
     """Score leaves and merge Top-K roots; long S6 publishes one root per leaf."""
-    # 组表只跟 query_start_loc 和组大小有关，与走哪个分档无关。四个 native_cube
-    # 调用点共用这一个任务，避免每个点各起一个（整请求组 / 双 query 组各一份）。
-    rgroup_row = pl.create_tensor([T_PAD], dtype=pl.INT32)
-    rgroup_rows = pl.create_tensor([T_PAD], dtype=pl.INT32)
-    rgroup_count = pl.create_tensor([1], dtype=pl.INT32)
-    pgroup_row = pl.create_tensor([T_PAD], dtype=pl.INT32)
-    pgroup_rows = pl.create_tensor([T_PAD], dtype=pl.INT32)
-    pgroup_count = pl.create_tensor([1], dtype=pl.INT32)
-    with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_query_groups") as groups_tid:
-        for token in pl.range(pl.tensor.dim(position_ids, 0)):
-            if pl.read(token_request, [token]) < 0:
-                pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [token, 0], topk_scores)
-                pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [token, 0], topk_idxs)
-        indexer_build_query_groups(
-            query_start_loc,
-            kv_seq_lens,
-            rgroup_row,
-            rgroup_rows,
-            rgroup_count,
-            S,
-        )
-        indexer_build_query_groups(
-            query_start_loc,
-            kv_seq_lens,
-            pgroup_row,
-            pgroup_rows,
-            pgroup_count,
-            2,
-        )
     native_page_bytes = pl.tensor.dim(idx_native_kv_cache, 1)
     # Zero-copy GM descriptors inside orchestration, as validation log §116.
     # One writable root allocation avoids partial-overlap Torch ABI arguments.
@@ -2184,6 +2162,13 @@
     token_request: pl.Tensor[[T_DYN], pl.INT32],
     # TND：组不跨请求，组边界从这里推。
     query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
+    rgroup_row: pl.Tensor[[T_PAD], pl.INT32],
+    rgroup_rows: pl.Tensor[[T_PAD], pl.INT32],
+    rgroup_count: pl.Tensor[[1], pl.INT32],
+    pgroup_row: pl.Tensor[[T_PAD], pl.INT32],
+    pgroup_rows: pl.Tensor[[T_PAD], pl.INT32],
+    pgroup_count: pl.Tensor[[1], pl.INT32],
+    groups_tid: pl.Scalar[pl.TASK_ID],
     cache_write_dep: pl.Scalar[pl.TASK_ID],
     weights_gate_dep: pl.Scalar[pl.TASK_ID],
     qh_quant_tid: pl.Scalar[pl.TASK_ID],
@@ -2203,6 +2188,13 @@
         kv_seq_lens,
         token_request,
         query_start_loc,
+        rgroup_row,
+        rgroup_rows,
+        rgroup_count,
+        pgroup_row,
+        pgroup_rows,
+        pgroup_count,
+        groups_tid,
         topk_scores,
         topk_idxs,
         qh_quant_tid,
@@ -2234,6 +2226,13 @@
     token_request: pl.Tensor[[T_DYN], pl.INT32],
     # TND：组不跨请求，组边界从这里推。
     query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
+    rgroup_row: pl.Tensor[[T_PAD], pl.INT32],
+    rgroup_rows: pl.Tensor[[T_PAD], pl.INT32],
+    rgroup_count: pl.Tensor[[1], pl.INT32],
+    pgroup_row: pl.Tensor[[T_PAD], pl.INT32],
+    pgroup_rows: pl.Tensor[[T_PAD], pl.INT32],
+    pgroup_count: pl.Tensor[[1], pl.INT32],
+    groups_tid: pl.Scalar[pl.TASK_ID],
     late_dep: pl.Scalar[pl.TASK_ID],
     cache_write_dep: pl.Scalar[pl.TASK_ID],
     host_max_seq_len: pl.Scalar[pl.INT32],
@@ -2266,6 +2265,13 @@
         kv_seq_lens,
         token_request,
         query_start_loc,
+        rgroup_row,
+        rgroup_rows,
+        rgroup_count,
+        pgroup_row,
+        pgroup_rows,
+        pgroup_count,
+        groups_tid,
         cache_write_dep,
         weights_gate_dep,
         qh_quant_tid,
```

### qr_capacity

```diff
--- control/decode_indexer.py
+++ qr_capacity/decode_indexer.py
@@ -107,7 +107,7 @@
 # 再在其上遍历行块。weights 投影仍用 MM_ROW_TILE，两者分开，故另起名字。
 QR_MM_ROW_TILE = 64
 QR_MM_N_TILE = 256
-QR_MM_T_PAD = ((T_PAD + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE) * QR_MM_ROW_TILE
+QR_MM_T_PAD = ((T + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE) * QR_MM_ROW_TILE

 MM_ROW_TILE = 16

@@ -1910,7 +1910,7 @@

     bs = pl.tensor.dim(x, 0)
     row_blocks = (bs + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE
-    # QR_MM_T_PAD >= T_PAD，缓冲只会变大不会变小；读者只读到 bs 为止。
+    # Accumulator covers ceil(bs / 64) tiles; query tail loads use separate qr_bf16 capacity.
     qr_acc_pad = pl.create_tensor([QR_MM_T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
     with pl.spmd(
         QR_PROJ_WORKERS,
```

### rope

```diff
--- control/decode_csa.py
+++ rope/decode_csa.py
@@ -367,7 +367,6 @@
         with pl.spmd(
             pl.min(rope_sign_blocks, CSA_ROPE_WORKERS),
             name_hint="csa_rope_sign",
-            deps=[offsets_tid],
             allow_early_resolve=True,
         ) as rope_tid:
             for rope_rb in pl.range(
```
