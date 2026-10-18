# v3 已有 IR 小片段

### 35_after_InitMemRef.py:5854 — qk_l1_local_inline5359__ssa_v1

```python
qk_l1_local_inline5359__ssa_v1: pl.Tile[[256, 512], pl.BF16, pl.MemRef(mem_mat_702, pl.const(0, pl.INT64), 262144), pl.Mem.Mat] = pl.tile.gather_row(
                                    qk_l1_local_inline5359__ssa_v0, kv_ring__ssa_v0, [qk_l1_row_inline5349__ssa_v0, 0], [qk_kv_row_inline5362__ssa_v0, 0], [128, 512], transpose=False
                                )
```

### 35_after_InitMemRef.py:5857 — qk_l1_t_inline5348__ssa_v0

```python
qk_l1_t_inline5348__ssa_v0: pl.Tile[
                                    [512, 256],
                                    pl.BF16,
                                    pl.MemRef(mem_mat_702, pl.const(0, pl.INT64), 262144),
                                    pl.Mem.Mat,
                                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                                ] = pl.tile.transpose_view(qk_l1_local_inline5359__ssa_v1)
```

### 36_after_MaterializeSemanticAliases.py:5854 — qk_l1_local_inline5359__ssa_v1

```python
qk_l1_local_inline5359__ssa_v1: pl.Tile[[256, 512], pl.BF16, pl.MemRef(mem_mat_486, pl.const(0, pl.INT64), 262144), pl.Mem.Mat] = pl.tile.gather_row(
                                    qk_l1_local_inline5359__ssa_v0, kv_ring__ssa_v0, [qk_l1_row_inline5349__ssa_v0, 0], [qk_kv_row_inline5362__ssa_v0, 0], [128, 512], transpose=False
                                )
```

### 36_after_MaterializeSemanticAliases.py:5857 — qk_l1_t_inline5348__ssa_v0

```python
qk_l1_t_inline5348__ssa_v0: pl.Tile[
                                    [512, 256],
                                    pl.BF16,
                                    pl.MemRef(mem_mat_702, pl.const(0, pl.INT64), 262144),
                                    pl.Mem.Mat,
                                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                                ] = pl.tile.transpose_view(qk_l1_local_inline5359__ssa_v1)
```

### 37_after_MemoryReuse.py:4835 — qk_l1_local_inline5359__ssa_v1

```python
qk_l1_local_inline5359__ssa_v1: pl.Tile[[256, 512], pl.BF16, pl.MemRef(mem_mat_1195, 0, 262144), pl.Mem.Mat] = pl.tile.gather_row(
                                    qk_l1_local_inline5359__ssa_v0, kv_ring__ssa_v0, [qk_l1_row_inline5349__ssa_v0, 0], [qk_kv_row_inline5362__ssa_v0, 0], [128, 512], transpose=False
                                )
```

### 37_after_MemoryReuse.py:4838 — qk_l1_t_inline5348__ssa_v0

```python
qk_l1_t_inline5348__ssa_v0: pl.Tile[
                                    [512, 256],
                                    pl.BF16,
                                    pl.MemRef(mem_mat_702, pl.const(0, pl.INT64), 262144),
                                    pl.Mem.Mat,
                                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                                ] = pl.tile.transpose_view(qk_l1_local_inline5359__ssa_v1)
```

### 37_after_MemoryReuse.py:2387 — score_query_inline2896__ssa_v0

```python
score_query_inline2896__ssa_v0: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_487, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                    [384, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat
                )
```

### 37_after_MemoryReuse.py:2390 — score_coeff_inline2934__ssa_v0

```python
score_coeff_inline2934__ssa_v0: pl.Tile[[16, 384], pl.FP16, pl.MemRef(mem_left_502, 49152, 12288), pl.Mem.Left] = pl.tile.create([16, 384], dtype=pl.FP16, target_memory=pl.Mem.Left)
```

### 37_after_MemoryReuse.py:2801 — query_left_inline2917__ssa_v0

```python
query_left_inline2917__ssa_v0: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_left_502, pl.const(0, pl.INT64), 49152), pl.Mem.Left] = pl.tile.move(
                                query_ready_inline2895__rv_v0, target_memory=pl.Mem.Left
                            )
```

### 37_after_MemoryReuse.py:2751 — buf_coefficient_pair_inline5311__ssa_v0

```python
buf_coefficient_pair_inline5311__ssa_v0: pl.Tile[[16, 384], pl.FP16, pl.MemRef(mem_left_497, pl.const(0, pl.INT64), 12288), pl.Mem.Left] = pl.tile.move(
                                    buf_coefficients_l1_inline5312__ssa_v0, target_memory=pl.Mem.Left
                                )
```

### 37_after_MemoryReuse.py:2762 — query_iter_mv

```python
query_iter_mv: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_489, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.move(query_iter, target_memory=pl.Mem.Mat)
```

### 37_after_MemoryReuse.py:2763 — coeff_iter_mv

```python
coeff_iter_mv: pl.Tile[[16, 384], pl.FP16, pl.MemRef(mem_left_497, pl.const(0, pl.INT64), 12288), pl.Mem.Left] = pl.tile.move(coeff_iter, target_memory=pl.Mem.Left)
```

### 37_after_MemoryReuse.py:4886 — query_after_inline2884__rv_v0_mv

```python
query_after_inline2884__rv_v0_mv: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_487, pl.const(0, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.move(
                        query_after_inline2884__rv_v0, target_memory=pl.Mem.Mat
                    )
```

### 37_after_MemoryReuse.py:4883 — coeff_after_inline2925__rv_v0_mv

```python
coeff_after_inline2925__rv_v0_mv: pl.Tile[[16, 384], pl.FP16, pl.MemRef(mem_left_502, 49152, 12288), pl.Mem.Left] = pl.tile.move(
                        coeff_after_inline2925__rv_v0, target_memory=pl.Mem.Left
                    )
```
