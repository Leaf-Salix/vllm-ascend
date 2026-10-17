# SPDX-License-Identifier: Apache-2.0
"""QK候选：Cube直接从Native KV读入L1，Vector只准备地址和计算softmax/PV。

只供独立实验使用。保留三槽及原QK/PV/在线归约数学；改写的内部稀疏索引
临时区没有其他消费者，外部TopK不变。事件4发布地址/零行，事件0改为
两AIV读完旧PV槽的确认，Cube复用三槽前等待。没有假定Scalar缓存一致。
"""

import ast
import copy
import json
from pathlib import Path


def nodes(source):
    return ast.parse(source).body


def address_plan(columns):
    padded = (columns + 7) // 8 * 8
    return nodes(f"""
# 每个lane写自己不重叠的256列。临时区在下一次调用由原valid-plan重新生成。
for direct_query in pl.range(qk_query_count_inline4331__ssa_v0):
    direct_t = qk_core_inline4347__ssa_v0 + direct_query * 24
    direct_b = direct_t // 6
    direct_start = qk_aiv_inline4434__ssa_v0 * 256
    direct_raw = pl.tile.load(cmp_sparse_indices_inline4329__rv_v2,
        [direct_t, direct_start], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
    direct_safe = pl.tile.maximums(direct_raw, 0)
    direct_page_columns = pl.tile.shrs(direct_safe, 5)
    direct_table = pl.tile.load(cmp_block_table__ssa_v0, [direct_b, 0],
        [1, {padded}], [1, {columns}], target_memory=pl.Mem.Vec)
    direct_scratch = pl.tile.create([1, 256], dtype=pl.INT32, target_memory=pl.Mem.Vec)
    direct_pages = pl.tile.gather(direct_table, direct_page_columns, direct_scratch)
    direct_offset = pl.tile.sub(direct_safe, pl.tile.shls(direct_page_columns, 5))
    direct_rows = pl.tile.add(pl.tile.shls(direct_pages, 5), direct_offset)
    direct_negative = pl.tile.minimums(direct_raw, 0)
    direct_physical = pl.tile.or_(direct_rows, direct_negative)
    pl.tile.store(direct_physical, [direct_t, direct_start], cmp_sparse_indices_inline4329__rv_v2)
if qk_aiv_inline4434__ssa_v0 == 0:
    direct_zero = pl.tile.full([32, 512], dtype=pl.BF16, value=0.0)
    pl.tile.store(direct_zero, [qk_core_inline4347__ssa_v0 * 384, 0], kv_transfer_inline4342__ssa_v0)
pl.system.sync_set(4, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
""")


def direct_gather():
    return nodes("""
direct_l1 = qk_l1_inline4426__iter_v1
if qk_sb_inline4367__ssa_v0 == 0:
    direct_position = pl.cast(pl.tensor.read(position_ids_t1__ssa_v0, [qk_t_inline4408__ssa_v0, 0]), pl.INDEX)
    direct_length = pl.min(direct_position + 1, 128)
    direct_begin = direct_position - direct_length + 1
    direct_head = direct_begin % 32
    # 五段覆盖完整128行；不足的尾部和负页补零，避免上一轮L1内容泄漏。
    for direct_segment in pl.range(5):
        direct_lo = pl.max(direct_segment * 32 - direct_head, 0)
        direct_hi = pl.min(direct_segment * 32 - direct_head + 32, 128)
        if direct_lo < direct_hi:
            direct_written = direct_lo
            if direct_lo < direct_length:
                direct_absolute = direct_begin + direct_lo
                direct_page = pl.cast(pl.tensor.read(ori_block_table__ssa_v0,
                    [qk_t_inline4408__ssa_v0 // 6, direct_absolute // 32]), pl.INDEX)
                if direct_page >= 0:
                    direct_valid_hi = pl.min(direct_hi, direct_length)
                    direct_l1 = pl.tile.gather_row(direct_l1, ori_kv_flat_inline4326__ssa_v1,
                        [qk_l1_row_inline4415__ssa_v0 + direct_lo, 0],
                        [direct_page * 32 + direct_absolute % 32, 0], [32, 512],
                        valid_shape=[direct_valid_hi - direct_lo, 512])
                    direct_written = direct_valid_hi
            if direct_written < direct_hi:
                direct_l1 = pl.tile.gather_row(direct_l1, kv_transfer_inline4342__ssa_v0,
                    [qk_l1_row_inline4415__ssa_v0 + direct_written, 0],
                    [qk_core_inline4347__ssa_v0 * 384, 0], [32, 512],
                    valid_shape=[direct_hi - direct_written, 512])
else:
    for direct_row in pl.range(128):
        direct_physical_row = pl.cast(pl.tensor.read(cmp_sparse_indices_inline4329__rv_v2,
            [qk_t_inline4408__ssa_v0, (qk_sb_inline4367__ssa_v0 - 1) * 128 + direct_row]), pl.INDEX)
        if direct_physical_row >= 0:
            direct_l1 = pl.tile.gather_row(direct_l1, cmp_kv_flat_inline4388__ssa_v0,
                [qk_l1_row_inline4415__ssa_v0 + direct_row, 0], [direct_physical_row, 0], [1, 512])
        else:
            direct_l1 = pl.tile.gather_row(direct_l1, kv_transfer_inline4342__ssa_v0,
                [qk_l1_row_inline4415__ssa_v0 + direct_row, 0],
                [qk_core_inline4347__ssa_v0 * 384, 0], [1, 512])
qk_l1_inline4426__ssa_v3 = direct_l1
""")


def reuse_wait(work):
    return nodes(f"""
direct_ack_work = {work}
direct_ack_mask = pl.tensor.read(valid_block_mask_inline4413__ssa_v0,
    [qk_core_inline4347__ssa_v0 + direct_ack_work // 5 * 24, direct_ack_work % 5])
if pl.cast(direct_ack_mask, pl.INDEX) > 0:
    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
""")


def use_direct_l1(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("直接L1候选首先只验证B16/S6")
    call = [c for c in schedule["call_bindings"].values() if c["logical_function"] == "qk_pv"]
    if len(call) != 1:
        raise ValueError("QK调用不唯一")
    arguments = {a["parameter"]: a["value"] for a in call[0]["arguments"]}
    table = arguments["cmp_block_table__ssa_v0"]
    columns = table["shape"][1]
    if table["dtype"] != "INT32" or table["strides"][1] != 1 or not 1 <= columns <= 8192:
        raise ValueError("压缩页表布局或容量不支持")
    # 原始TopK仍用其独立输出。本变换只写valid-plan生成的私有中间Tensor。
    intermediate = arguments["cmp_sparse_indices_inline4329__rv_v2"]
    aliases = []
    for binding in schedule["call_bindings"].values():
        for arg in binding["arguments"]:
            if arg["value"] == intermediate:
                aliases.append((binding["logical_function"], arg["parameter"]))
    if {name for name, _ in aliases} != {"qk_pv", "csa_slots_build_valid_qk_plan"}:
        raise ValueError("内部稀疏索引有额外使用者：" + str(aliases))
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    cube, vector = functions["qk_pv_aic"], functions["qk_pv_aiv"]
    original_cube, original_vector = copy.deepcopy(cube), copy.deepcopy(vector)
    cube_loop = next(n for n in cube.body if isinstance(n, ast.For))
    vector_loop = next(n for n in vector.body if isinstance(n, ast.For))
    if len(cube_loop.body) != 3 or len(vector_loop.body) != 9:
        raise ValueError("QK流水结构改变")
    # Vector删除KV准备和原KV_READY两个策略分支；保留原softmax/PV数学。
    vector_loop.body = [vector_loop.body[i] for i in (0, 1, 2, 5, 7, 8)]
    softmax = vector_loop.body[3]
    wait_branch = [n for n in ast.walk(softmax) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "t_dim_inline4335__ssa_v0 < 144"]
    if len(wait_branch) != 1:
        raise ValueError("原score等待分支改变")
    # 即使未来扩至大档，也不再需要以前用于KV交接的特殊前置等待。
    for parent in ast.walk(softmax):
        body = getattr(parent, "body", None)
        if isinstance(body, list) and wait_branch[0] in body:
            i = body.index(wait_branch[0])
            body[i:i + 1] = wait_branch[0].body
            break
    pv = vector_loop.body[4]
    valid_pv = pv.body[4]
    if not isinstance(valid_pv, ast.If) or "next_right_inline4247__ssa_v0" not in ast.unparse(valid_pv):
        raise ValueError("原PV合并分支改变")
    valid_pv.body[-1:-1] = nodes("""
# MTE2发布意味着两个PV加载都已完成；不要求后续Vector计算完成。
pl.system.sync_set(0, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
""")
    # 地址/零行准备置于任何大running-state分配之前，减少同时活跃UB。
    insert = next(i for i, n in enumerate(vector.body) if isinstance(n, ast.AnnAssign)
                  and n.target.id == "m_idx_inline4310__ssa_v0")
    vector.body[insert:insert] = address_plan(columns)
    for parameter in vector.args.args:
        if parameter.arg == "cmp_sparse_indices_inline4329__rv_v2":
            parameter.annotation = ast.Subscript(
                value=ast.Attribute(ast.Name("pl", ast.Load()), "InOut", ast.Load()),
                slice=parameter.annotation, ctx=ast.Load())
    # Cube在开始读地址前等两lane写回完成，再丢弃本核旧Scalar缓存。
    insertion = cube.body.index(cube_loop)
    cube.body[insertion:insertion] = nodes("""
pl.system.sync_wait(4, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
pl.system.fence()
pl.system.cacheinvalid()
pl.system.fence()
""")
    qk = cube_loop.body[0]
    valid_qk = qk.body[4]
    if not isinstance(valid_qk, ast.If):
        raise ValueError("原Cube有效块结构改变")
    old = valid_qk.body
    wait = next(i for i, n in enumerate(old) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "pl.system.sync_wait")
    if (ast.unparse(old[wait].value.args[0]) != "0"
            or old[wait + 2].target.id != "qk_l1_inline4426__ssa_v3"):
        raise ValueError("原L1搬运结构改变")
    valid_qk.body[wait:wait + 3] = [old[wait + 1]] + direct_gather()
    barrier = nodes("if qk_tick_inline4364__idx_v0 >= 3:\n    pass\n")[0]
    barrier.body = reuse_wait("qk_tick_inline4364__idx_v0 - 3")
    cube_loop.body.insert(0, barrier)
    tail = nodes("if qk_block_count_inline4405__ssa_v0 > 0:\n    pass\n")[0]
    tail.body = reuse_wait("qk_block_count_inline4405__ssa_v0 - 1")
    cube.body.insert(cube.body.index(cube_loop) + 1, tail)
    # 点积/指数/在线合并/归一化/输出写入保持；只允许上面明确的供数和交接变更。
    def math_calls(function):
        selected = {"pl.tile.matmul", "pl.tile.matmul_acc", "pl.tile.exp", "pl.tile.row_max",
                    "pl.tile.row_sum", "pl.tile.row_expand_div", "pl.tile.row_expand_mul",
                    "pl.tile.col_expand_mul", "pl.tile.concat"}
        return [ast.dump(n) for n in ast.walk(function) if isinstance(n, ast.Call)
                and ast.unparse(n.func) in selected]
    for before, after in ((original_cube, cube), (original_vector, vector)):
        if sorted(math_calls(before)) != sorted(math_calls(after)):
            raise ValueError("意外改变QK/PV数学")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "core_bodies_qk_direct_l1.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    audit = {"table_columns": columns, "qk_math_unchanged": True, "pv_math_unchanged": True,
             "external_topk_unchanged": True, "address_intermediate_users": aliases,
             "extra_root_arguments": False, "extra_gm_allocations": False,
             "zero_rows_per_core": 32, "zero_storage": "原kv_transfer每组首32行",
             "index_storage": "原valid-plan输出temporary，下一调用重建",
             "event_4": "两AIV MTE3发布地址与零行，AIC等待后fence/cacheinvalid/fence",
             "event_0": "两AIV MTE2发布PV消费完成，AIC复用三槽前等待；末尾排空",
             "original_event_1_2_3": "score/probability/PV协议保持",
             "device_validated": False, "performance_proven": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
