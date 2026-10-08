# Metadata + QR 组合效率对拍（2026-10-08）

## 实验身份与边界

分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`；正式HEAD `45857a048`，kernel同`ad8b69520`。
BSH固定`8e83adda1f5959da0d407f2d27d56f52a16466e5`，23份实际源码SHA均强制核对。
独立候选只组合前轮metadata合并及QR accumulator 448→384行；不包含counts或rope删等待边。
正式kernel没有修改。六臂：Native、BSH、当前TND/control、metadata、qr_capacity、combined。
CANN9.2.0-beta.2、ATB9.2.0-beta.2、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1；
PyPTO3e87a843、Simplera54c0509、PTOAS0.66，复用现有框架，不重装。
真实第4层权重、seed1024、合成hidden/history；HC pre + norm + CSA + HC post，
不是整模型DP/EP16、EPLB、真实DSpark接收率或吞吐测试。

## 协议与完成状态

同进程同卡，各臂独立cache、注册名、图owner。Native static+SuperKernel经实际profile确认；PTO禁止回退。
确定性level1/HCCL=true、atomic_add=0。输出、TopK及六类cache/state对control完整逐位一致。
编译/raw、连续两次、100次连续回放及恢复初态后均做逐位与保护区门禁。
正反循环移位平衡次序，不删除尾样本；单次与100次连续回放口径独立。

| case | 物理卡 | T | 连续轮数×100 | 单次样本/臂 |
| --- | --- | --- | --- | --- |
| combo-h131072-b16 | 8 | 96 | 36 | 120 |
| combo-h131072-b4 | 8 | 24 | 36 | 120 |
| combo-h131072-b40 | 8 | 240 | 36 | 120 |
| combo-h8192-b16 | 8 | 96 | 36 | 120 |
| combo-h8192-b24 | 8 | 144 | 36 | 120 |
| ragged-combo-h131072-b16 | 8 | 60 | 30 | 100 |
| ragged-combo-h8192-b4 | 8 | 11 | 30 | 100 |

## 组合直接配对

延迟变化为同轮配对中位数，负数加速。单次/BSH为单次均值比。

| case | 组合/当前TND 连续 | 组合/BSH 连续 | 组合/metadata 连续 | 组合/QR 连续 | 组合/BSH 单次 |
| --- | --- | --- | --- | --- | --- |
| combo-h131072-b16 | +0.816% | +2.195% | +0.992% | +0.935% | +1.874% |
| combo-h131072-b4 | -0.458% | +0.882% | -0.569% | +1.131% | +1.208% |
| combo-h131072-b40 | +0.072% | +1.269% | -0.377% | -1.493% | +1.621% |
| combo-h8192-b16 | +0.286% | +4.069% | +1.034% | +1.037% | +3.071% |
| combo-h8192-b24 | +0.284% | +1.897% | +0.806% | +0.900% | +2.391% |
| ragged-combo-h131072-b16 | -0.552% | — | -0.037% | +0.178% | — |
| ragged-combo-h8192-b4 | -2.721% | — | -0.693% | -3.213% | — |

## 连续100次回放轮均值

单位μs；连续轮次的P95不等于单次P95。

| case | Native | BSH | 当前TND | metadata | QR | 组合 |
| --- | --- | --- | --- | --- | --- | --- |
| combo-h131072-b16 | 1087.114 | 925.130 | 937.515 | 935.838 | 936.426 | 945.064 |
| combo-h131072-b4 | 642.145 | 569.854 | 576.718 | 577.737 | 567.812 | 574.501 |
| combo-h131072-b40 | 1809.449 | 1808.875 | 1830.289 | 1838.344 | 1859.919 | 1831.350 |
| combo-h8192-b16 | 777.372 | 731.158 | 758.775 | 753.621 | 753.037 | 761.258 |
| combo-h8192-b24 | 903.113 | 909.964 | 925.449 | 919.996 | 919.523 | 927.869 |
| ragged-combo-h131072-b16 | 1003.177 | — | 874.096 | 869.075 | 867.231 | 868.892 |
| ragged-combo-h8192-b4 | 392.418 | — | 439.151 | 430.287 | 441.315 | 427.143 |

## 恢复初态后的单次回放均值

单位μs；连续轮次的P95不等于单次P95。

| case | Native | BSH | 当前TND | metadata | QR | 组合 |
| --- | --- | --- | --- | --- | --- | --- |
| combo-h131072-b16 | 1130.703 | 965.577 | 971.672 | 973.985 | 976.039 | 983.670 |
| combo-h131072-b4 | 765.875 | 654.875 | 653.882 | 654.800 | 667.148 | 662.784 |
| combo-h131072-b40 | 1863.497 | 1849.195 | 1872.122 | 1885.233 | 1905.803 | 1879.173 |
| combo-h8192-b16 | 762.199 | 728.598 | 746.826 | 748.506 | 745.293 | 750.973 |
| combo-h8192-b24 | 919.792 | 928.657 | 942.881 | 938.681 | 940.130 | 950.861 |
| ragged-combo-h131072-b16 | 959.316 | — | 927.625 | 921.793 | 920.103 | 921.851 |
| ragged-combo-h8192-b4 | 425.265 | — | 473.033 | 459.404 | 476.179 | 462.675 |

## 精度与TND格式

所有候选/control的输出、TopK和六类cache/state完整逐位一致；等长control/BSH亦一致。
T11长度[6,3,1,1]；T60长度[1,2,3,4,5,6,6,6,1,2,3,4,5,6,4,2]。没有扩展为B×6，BSH均拒绝。
零新增误差不等于与Native逐位一致；性能版继承的Native差异如下。

| case | 组合/Native output relative L2 | max_abs | ULP P99 | ULP max |
| --- | --- | --- | --- | --- |
| combo-h131072-b16 | 0.0041684294 | 0.03125 | 35.0 | 30858.0 |
| combo-h131072-b4 | 0.0040970143 | 0.03125 | 34.0 | 30706.0 |
| combo-h131072-b40 | 0.0041421347 | 0.03125 | 35.0 | 30858.0 |
| combo-h8192-b16 | 0.0033384276 | 0.03125 | 26.0 | 30678.0 |
| combo-h8192-b24 | 0.0033479758 | 0.03125 | 26.0 | 30678.0 |
| ragged-combo-h131072-b16 | 0.0042388153 | 0.03125 | 37.0 | 30779.0 |
| ragged-combo-h8192-b4 | 0.0031709467 | 0.0234375 | 25.0 | 30757.0 |

## 静态审查

生成代码六组表在外层alloc_0，offsets task生产；四条Cube分支均保留显式metadata依赖。
QR INT32缓冲实际384×8192；query BF16仍432×64×128，未缩掉必要尾行。
metadata省一个任务，同时延长部分前置链；QR缩容不改变实际矩阵行数。不能据源码规模推算收益。

## 原始证据

任务：`task_20261008_174554_8665020746`。全部7case完整结果；物理卡以完成表为准。
runner SHA256：`f8c9a7b06404ce7ba3b5a3630dbdbf4af54159c354ec7cdeca75f954daad997e`。
原始results.zip SHA256：`d7508994ef9195ea75ae1e094ab6cb6d3b3b7e4e83ccda8fb15a12205fa02bb0`。
所有计时样本、配对百分比及摘要见[完整样本](DSV4_FLASH_CSA_TND_COMBINATION_20261008_SAMPLES.md)。

## 决策：不合入组合

七档均完成、任务exit0，设备8锁释放；正式kernel保持原状。
五个等长档位，组合连续对BSH配对中位数慢0.882%～4.069%；单次均值慢1.208%～3.071%。
128K/B16组合36轮全部慢于control、metadata和QR；8K/B16、B24也相对control回退。
128K/B4连续快control0.458%，但单次慢control1.361%；B40连续+0.072%接近零，不能称为收益。
T11组合对control连续快2.721%、单次快2.190%；T60连续快0.552%、单次快0.622%，
但T60连续仍比QR单项慢0.178%。T11组合单次462.675μs也慢于metadata单项459.404μs。
变长收益不足以支持统一采用，也不能称为所有口径均最优。

这是组合第一次硬件矩阵，支持“本轮未持平BSH且有多个回退”的结论；
不把同进程36轮当作跨任务独立复测，也不声称永远无法持平。
明确的多个回退已经否定本轮全面持平目标，不再扩展可丢弃候选的等长复测。
两个单项在前轮的局部收益不能相加估计组合，组合实测体现了明显的非加性。
没有取得内部任务独立耗时，所以不能把组合回退全部归因于某个任务或GM布局。
Native现有精度差异仍在，本轮仅证明没有新增误差。

## 可复现组合补丁

在正式代码基线上对performance包应用以下补丁；不修改precision包。

```diff
--- a/vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_csa.py
+++ b/vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_csa.py
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
--- a/vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_indexer.py
+++ b/vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_indexer.py
@@ -107,7 +107,7 @@
 # 再在其上遍历行块。weights 投影仍用 MM_ROW_TILE，两者分开，故另起名字。
 QR_MM_ROW_TILE = 64
 QR_MM_N_TILE = 256
-QR_MM_T_PAD = ((T_PAD + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE) * QR_MM_ROW_TILE
+QR_MM_T_PAD = ((T + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE) * QR_MM_ROW_TILE

 MM_ROW_TILE = 16

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
@@ -1910,7 +1888,7 @@

     bs = pl.tensor.dim(x, 0)
     row_blocks = (bs + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE
-    # QR_MM_T_PAD >= T_PAD，缓冲只会变大不会变小；读者只读到 bs 为止。
+    # Accumulator covers ceil(bs / 64) tiles; query tail loads use separate qr_bf16 capacity.
     qr_acc_pad = pl.create_tensor([QR_MM_T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
     with pl.spmd(
         QR_PROJ_WORKERS,
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
