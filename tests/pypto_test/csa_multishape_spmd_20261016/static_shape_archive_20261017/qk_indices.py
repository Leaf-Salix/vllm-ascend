# SPDX-License-Identifier: Apache-2.0
"""QK/PV 在本核 UB 批量计算压缩 KV 物理行，减少逐行两次 GM Scalar 读取。

每块64行的索引、当前请求页表通过 MTE2 读取；整数 Vector 位移和 gather
计算 page*32+offset。保留原负索引判定、逐行 KV DMA、所有 FFTS 和点积。
缓存只活在当前块内，不跨调用保存，不假定 Scalar 跨核缓存一致。
"""

import ast
import copy
from pathlib import Path


def cache_qk_indices(schedule, catalog):
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "qk_pv_aiv")
    calls = [call for call in schedule["call_bindings"].values() if call["logical_function"] == "qk_pv"]
    if len(calls) != 1:
        raise ValueError("当前候选要求唯一的 QK/PV 调用")
    table = next(arg["value"] for arg in calls[0]["arguments"] if arg["parameter"] == "cmp_block_table__ssa_v0")
    columns = table["shape"][1]
    if columns <= 0 or table["dtype"] != "INT32" or table["strides"][1] != 1:
        raise ValueError("压缩页表需要原连续 INT32 列")
    padded_columns = (columns + 7) // 8 * 8
    if padded_columns > 8192:
        raise ValueError("本候选未为超过32KiB的页表规划UB")
    if schedule["root_arguments"]["cmp_kv"]["shape"][0] * 32 >= 2**31:
        raise ValueError("物理KV行数超出当前INT32地址候选范围")
    names = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)}
    if any(name.startswith("qk_meta_") for name in names):
        raise ValueError("重复插入 QK 索引缓存")
    required = {
        "qk_sb_inline4367__ssa_v1",
        "qk_lane_kv_inline4438__ssa_v0",
        "qk_t_inline4408__ssa_v1",
        "qk_b_inline4274__ssa_v0",
        "cmp_sparse_indices_inline4329__rv_v2",
        "qk_row_inline4264__idx_v0",
    }
    if not required <= names:
        raise ValueError("冻结 QK/PV 的局部绑定改变")
    prefix = ast.parse(f"""
qk_meta_start = (qk_sb_inline4367__ssa_v1 - 1) * 128 + qk_lane_kv_inline4438__ssa_v0
qk_meta_raw = pl.tile.load(cmp_sparse_indices_inline4329__rv_v2,
    [qk_t_inline4408__ssa_v1, qk_meta_start], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
qk_meta_nonnegative = pl.tile.maximums(qk_meta_raw, 0)
qk_meta_page_columns = pl.tile.shrs(qk_meta_nonnegative, 5)
qk_meta_table = pl.tile.load(cmp_block_table__ssa_v0,
    [qk_b_inline4274__ssa_v0, 0], [1, {padded_columns}], [1, {columns}], target_memory=pl.Mem.Vec)
qk_meta_scratch = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
qk_meta_pages = pl.tile.gather(qk_meta_table, qk_meta_page_columns, qk_meta_scratch)
qk_meta_remainders = pl.tile.sub(qk_meta_nonnegative, pl.tile.shls(qk_meta_page_columns, 5))
qk_meta_physical_rows = pl.tile.add(pl.tile.shls(qk_meta_pages, 5), qk_meta_remainders)
""").body
    matches = 0

    class CacheLoop(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal matches
            if "qk_row_inline4264__idx_v0" not in {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}:
                return self.generic_visit(node)
            if ast.unparse(node.iter.args[0]) != "64":
                raise ValueError("QK 行块不是64行")
            matches += 1
            valid_column = node.body[1]
            read_raw = valid_column.body[0]
            valid_index = valid_column.body[1]
            if (
                ast.unparse(read_raw.value.func) != "pl.tensor.read"
                or ast.unparse(read_raw.value.args[0]) != "cmp_sparse_indices_inline4329__rv_v2"
                or len(valid_index.body) != 5
            ):
                raise ValueError("QK 稀疏索引读取结构改变")
            before_gather = ast.dump(valid_index.body[3])
            read_raw.value = ast.parse("pl.tile.read(qk_meta_raw, [0, qk_row_inline4264__idx_v0])", mode="eval").body
            source = valid_index.body[2]
            if not isinstance(source, ast.AnnAssign) or source.target.id != "qk_src_inline4261__ssa_v0":
                raise ValueError("QK 物理行绑定改变")
            source.value = ast.parse(
                "pl.cast(pl.tile.read(qk_meta_physical_rows, [0, qk_row_inline4264__idx_v0]), pl.INDEX)", mode="eval"
            ).body
            valid_index.body = [source] + valid_index.body[3:]
            if ast.dump(valid_index.body[1]) != before_gather:
                raise ValueError("改变了原 KV DMA")
            return copy.deepcopy(prefix) + [node]

    CacheLoop().visit(fn)
    if matches != 1:
        raise ValueError("QK 压缩行循环不是唯一的")
    ast.fix_missing_locations(module)
    audit = {
        "kernel": "qk_pv_aiv",
        "row_block": 64,
        "table_columns": columns,
        "table_ub_columns": padded_columns,
        "table_load_valid_columns": columns,
        "all_address_arithmetic_int32": True,
        "negative_index_branch_preserved": True,
        "kv_dma_unchanged": True,
        "qk_pv_math_unchanged": True,
        "communication_unchanged": True,
        "host_roundtrip_added": False,
        "extra_gm_metadata_buffer": False,
        "native_inputs_unchanged": True,
        "scope": "当前块内UB地址缓存；需完整CPU编译和原exact真机门槛验证",
    }
    return ast.unparse(module) + "\n", audit
