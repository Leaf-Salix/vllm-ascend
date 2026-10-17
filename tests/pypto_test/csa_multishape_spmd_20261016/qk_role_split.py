# SPDX-License-Identifier: Apache-2.0
"""CPU候选：AIV0准备128行KV，AIV1处理64个head的softmax/PV/输出。

Cube数学和三槽保持。两条AIV都消费原Score/PV事件，都发送KV/Prob事件；
无数学工作的那条只发送/消费对应credit，不能擅自减少FFTS参与者。
此变换改变packed输出所有权，禁止直接套用旧row_dependencies。
"""

import ast
import copy
import json
from pathlib import Path


def body_source(table_columns, *, head_tile=8, softmax_tile=8, publication_tile=None):
    if head_tile not in (8, 16) or softmax_tile not in (8, 16, 32):
        raise ValueError("head Tile只评估8/16，softmax只评估8/16/32，禁止任意扩容")
    publication_tile = head_tile if publication_tile is None else publication_tile
    if publication_tile not in (8, 16) or head_tile % publication_tile:
        raise ValueError("发布必须按8/16head，并且整除常驻状态的head分块")
    padded_columns = (table_columns + 7) // 8 * 8
    lines = [
        "core = original_block_idx",
        "lane = original_subblock_idx",
        "blocks = ((tokens - core + 23) // 24) * 5",
        "pl.system.set_ffts(ffts)",
        "if lane == 0:",
        "    for tick in pl.range(blocks + 3):",
        "        query = core + tick // 5 * 24",
        "        block = tick % 5",
        "        if tick < blocks:",
        "            if pl.tensor.read(valid, [query, block]) > 0:",
        "                kvbuf = pl.tile.full([128, 512], dtype=pl.BF16, value=0.0)",
        "                if block == 0:",
        "                    position = pl.cast(pl.tensor.read(positions, [query, 0]), pl.INDEX)",
        "                    length = pl.min(position + 1, 128)",
        "                    begin = position - length + 1",
        "                    head = begin % 32",
    ]

    def emit(indent, statements):
        lines.extend(" " * indent + text for text in statements)

    for run in range(5):
        emit(20, [
            f"lo = pl.max({run * 32} - head, 0)",
            f"hi = pl.min({(run + 1) * 32} - head, length)",
            "if hi > lo:",
            "    absolute = begin + lo",
            "    page = pl.tensor.read(raw_table, [query // 6, absolute // 32])",
            "    if page >= 0:",
            "        source = pl.cast(page, pl.INDEX) * 32 + absolute % 32",
            "        kvbuf = pl.tile.gather_row(kvbuf, raw_kv, [lo, 0], [source, 0],",
            "            [128, 512], valid_shape=[hi - lo, 512], transpose=False)",
        ])
    emit(20, ["pl.tile.store(kvbuf, [(core * 3 + tick % 3) * 128, 0], kv_transfer)"])
    emit(16, [
        "else:",
        "    raw_indices = pl.tile.load(indices, [query, (block - 1) * 128], [1, 128], [1, 128], target_memory=pl.Mem.Vec)",
        "    nonnegative = pl.tile.maximums(raw_indices, 0)",
        "    page_columns = pl.tile.shrs(nonnegative, 5)",
        f"    table = pl.tile.load(cmp_table, [query // 6, 0], [1, {padded_columns}], [1, {table_columns}], target_memory=pl.Mem.Vec)",
        "    page_tmp = pl.tile.create([1, 128], dtype=pl.INT32, target_memory=pl.Mem.Vec)",
        "    pages = pl.tile.gather(table, page_columns, page_tmp)",
        "    remainder = pl.tile.sub(nonnegative, pl.tile.shls(page_columns, 5))",
        "    physical = pl.tile.add(pl.tile.shls(pages, 5), remainder)",
        "    for row, (kv_iter,) in pl.range(128, init_values=(kvbuf,)):",
        "        if pl.tile.read(raw_indices, [0, row]) >= 0:",
        "            source = pl.cast(pl.tile.read(physical, [0, row]), pl.INDEX)",
        "            kv_next = pl.yield_(pl.tile.gather_row(kv_iter, cmp_kv, [row, 0], [source, 0], [1, 512], transpose=False))",
        "        else:",
        "            kv_next = pl.yield_(kv_iter)",
        "        kv_done = pl.yield_(kv_next)",
        "    pl.tile.store(kv_done, [(core * 3 + tick % 3) * 128, 0], kv_transfer)",
    ])

    def wait_score(indent):
        emit(indent, [
            "if tick > 0 and tick <= blocks:",
            "    work = tick - 1",
            "    if pl.tensor.read(valid, [core + work // 5 * 24, work % 5]) > 0:",
            "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)",
            "if tick < blocks:",
            "    if pl.tensor.read(valid, [core + tick // 5 * 24, tick % 5]) > 0:",
            "        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)",
        ])

    wait_score(8)
    emit(8, [
        "if tick > 0 and tick <= blocks:",
        "    work = tick - 1",
        "    if pl.tensor.read(valid, [core + work // 5 * 24, work % 5]) > 0:",
        "        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)",
        "if tick >= 3:",
        "    work = tick - 3",
        "    if pl.tensor.read(valid, [core + work // 5 * 24, work % 5]) > 0:",
        "        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)",
    ])
    emit(0, ["else:"])
    emit(4, [
        "swap_row = pl.tile.ci(0, [1, 64], dtype=pl.INT32)",
        "swap_tmp = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)",
        "swap_lane = pl.tile.rems(swap_row, 2, swap_tmp)",
        "swap_col = pl.tile.adds(pl.tile.sub(swap_row, pl.tile.muls(swap_lane, 2)), 449)",
        f"swap_base = pl.tile.create([{publication_tile}, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)",
        "swap_rows = pl.tile.col_expand(swap_base, swap_col)",
        f"row_offsets = pl.tile.muls(pl.tile.ci(0, [1, {publication_tile}], dtype=pl.INT32), 512)",
        f"swap_indices = pl.tile.reshape(pl.tile.row_expand_add(swap_rows, pl.tile.reshape(row_offsets, [{publication_tile}, 1])), [1, {publication_tile * 64}])",
        "initial_m = pl.tile.load(sink, [0, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec)",
        "initial_l = pl.tile.muls(initial_m, 0.0)",
    ])
    state_names = [f"{side}{chunk}" for side in ("left", "right") for chunk in range(64 // head_tile)]
    for name in state_names:
        emit(4, [f"initial_{name} = pl.tile.full([{head_tile}, 256], dtype=pl.FP32, value=0.0)"])
    values = ["m", "l", *state_names]
    iter_values = ", ".join(f"{name}_iter" for name in values)
    initial_values = ", ".join(f"initial_{name}" for name in values)
    emit(4, [f"for tick, ({iter_values}) in pl.range(blocks + 3, init_values=({initial_values})):"])
    wait_score(8)
    emit(8, [
        "if tick > 0 and tick <= blocks:",
        "    work = tick - 1",
        "    query = core + work // 5 * 24",
        "    block = work % 5",
        "    if pl.tensor.read(valid, [query, block]) > 0:",
        "        transfer_row = (core * 3 + work % 3) * 64",
    ])
    for chunk in range(64 // softmax_tile):
        row = chunk * softmax_tile
        emit(16, [
            f"sf_scores = pl.tile.load(score_transfer, [transfer_row + {row}, 0], [{softmax_tile}, 128], [{softmax_tile}, 128], target_memory=pl.Mem.Vec)",
            "sf_bias = pl.tile.load(bias, [query, block * 128], [1, 128], [1, 128], target_memory=pl.Mem.Vec)",
            "sf_masked = pl.tile.col_expand_add(pl.tile.muls(sf_scores, 0.04419417382415922), sf_bias)",
            f"sf_tmp = pl.tile.create([{softmax_tile}, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)",
            "sf_m = pl.tile.row_max(sf_masked, sf_tmp)",
            "sf_exp = pl.tile.exp(pl.tile.row_expand_sub(sf_masked, sf_m))",
            "sf_l = pl.tile.row_sum(sf_exp, sf_tmp)",
            "sf_prob = pl.tile.cast(sf_exp, target_type=pl.BF16, mode='rint')",
            f"pl.tile.store(sf_prob, [transfer_row + {row}, 0], probability_transfer)",
            f"pl.tile.store(sf_m, [transfer_row + {row}, 0], mi_transfer)",
            f"pl.tile.store(sf_l, [transfer_row + {row}, 0], li_transfer)",
        ])
    emit(16, ["pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)"])
    emit(8, [
        "if tick >= 3:",
        "    work = tick - 3",
        "    query = core + work // 5 * 24",
        "    block = work % 5",
        "    if pl.tensor.read(valid, [query, block]) > 0:",
        "        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)",
        "        transfer_row = (core * 3 + work % 3) * 64",
        "        incoming_m = pl.tile.load(mi_transfer, [transfer_row, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec)",
        "        incoming_l = pl.tile.load(li_transfer, [transfer_row, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec)",
        "        next_m = pl.tile.maximum(m_iter, incoming_m)",
        "        alpha = pl.tile.exp(pl.tile.sub(m_iter, next_m))",
        "        beta = pl.tile.exp(pl.tile.sub(incoming_m, next_m))",
        "        next_l = pl.tile.add(pl.tile.mul(alpha, l_iter), pl.tile.mul(beta, incoming_l))",
    ])
    for chunk in range(64 // head_tile):
        row = chunk * head_tile
        for coefficient in ("alpha", "beta"):
            emit(16, [f"{coefficient}{chunk} = pl.tile.reshape(pl.tile.extract(pl.tile.reshape({coefficient}, [1, 64]), 0, {row}, [1, {head_tile}], target_memory=pl.Mem.Vec), [{head_tile}, 1])"])
        for side, col in (("left", 0), ("right", 256)):
            emit(16, [
                f"incoming = pl.tile.load(pv_transfer, [transfer_row + {row}, {col}], [{head_tile}, 256], [{head_tile}, 256], target_memory=pl.Mem.Vec)",
                f"scaled_old = pl.tile.row_expand_mul({side}{chunk}_iter, alpha{chunk})",
                f"scaled_incoming = pl.tile.row_expand_mul(incoming, beta{chunk})",
                f"next_{side}{chunk} = pl.tile.add(scaled_old, scaled_incoming)",
            ])

    def yields(indent, output, source):
        emit(indent, [", ".join(f"{output}_{name}" for name in values)
                      + " = pl.yield_(" + ", ".join(f"{source}_{name}" for name in values) + ")"])

    yields(16, "valid", "next")
    emit(12, ["else:"])
    emit(16, [", ".join(f"valid_{name}" for name in values) + " = pl.yield_(" + iter_values + ")"])
    emit(12, ["if block == 4:"])
    for chunk in range(64 // publication_tile):
        row = chunk * publication_tile
        state_chunk = row // head_tile
        # 发布只读已有状态的连续行；不新增GM中转，也不改变PV状态更新顺序。
        left = f"valid_left{state_chunk}"
        right = f"valid_right{state_chunk}"
        if publication_tile != head_tile:
            left = f"pl.tile.slice({left}, [{publication_tile}, 256], [{row % head_tile}, 0])"
            right = f"pl.tile.slice({right}, [{publication_tile}, 256], [{row % head_tile}, 0])"
        emit(16, [
            f"pub_m = pl.tile.reshape(pl.tile.extract(pl.tile.reshape(valid_m, [1, 64]), 0, {row}, [1, {publication_tile}], target_memory=pl.Mem.Vec), [{publication_tile}, 1])",
            f"pub_l = pl.tile.reshape(pl.tile.extract(pl.tile.reshape(valid_l, [1, 64]), 0, {row}, [1, {publication_tile}], target_memory=pl.Mem.Vec), [{publication_tile}, 1])",
            f"pub_oi = pl.tile.concat({left}, {right})",
            f"pub_sink = pl.tile.load(sink, [{row}, 0], [{publication_tile}, 1], [{publication_tile}, 1], target_memory=pl.Mem.Vec)",
            "pub_bias = pl.tile.add(pl.tile.sub(pub_m, pub_m), pub_sink)",
            "pub_denom = pl.tile.add(pub_l, pl.tile.exp(pl.tile.sub(pub_bias, pub_m)))",
            "pub_full = pl.tile.row_expand_div(pub_oi, pub_denom)",
            "pub_bf16 = pl.tile.cast(pub_full, target_type=pl.BF16, mode='rint')",
            f"pub_rope = pl.tile.slice(pub_full, [{publication_tile}, 64], [0, 448])",
            "pub_cos = pl.tile.load(cos, [query, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)",
            "pub_sin = pl.tile.load(sin_signed, [query, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)",
            f"pub_tmp = pl.tile.create([1, {publication_tile * 64}], dtype=pl.INT32, target_memory=pl.Mem.Vec)",
            f"pub_swap_flat = pl.tile.gather(pl.tile.reshape(pub_full, [1, {publication_tile * 512}]), swap_indices, pub_tmp)",
            f"pub_swap = pl.tile.reshape(pub_swap_flat, [{publication_tile}, 64])",
            "pub_rot = pl.tile.add(pl.tile.col_expand_mul(pub_rope, pub_cos), pl.tile.col_expand_mul(pub_swap, pub_sin))",
            "pub_rot_bf16 = pl.tile.cast(pub_rot, target_type=pl.BF16, mode='rint')",
            f"pub_result = pl.tile.concat(pl.tile.slice(pub_bf16, [{publication_tile}, 448], [0, 0]), pub_rot_bf16)",
        ])
        if publication_tile == 8:
            emit(16, [f"pl.tile.store(pl.tile.reshape(pub_result, [1, 4096]), [{chunk * 384} + query, 0], packed)"])
        else:
            emit(16, ["pub_packed = pl.tile.reshape(pub_result, [2, 4096])"])
            for half in range(2):
                emit(16, [f"pl.tile.store(pl.tile.slice(pub_packed, [1, 4096], [{half}, 0]), [{(chunk * 2 + half) * 384} + query, 0], packed)"])
    emit(16, [
        "reset_m = pl.tile.load(sink, [0, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec)",
        "reset_l = pl.tile.muls(reset_m, 0.0)",
    ])
    for name in state_names:
        emit(16, [f"reset_{name} = pl.tile.full([{head_tile}, 256], dtype=pl.FP32, value=0.0)"])
    yields(16, "published", "reset")
    emit(12, ["else:"])
    yields(16, "published", "valid")
    yields(12, "after", "published")
    emit(8, ["else:"])
    emit(12, [", ".join(f"after_{name}" for name in values) + " = pl.yield_(" + iter_values + ")"])
    yields(8, "running", "after")
    return "\n".join(lines)


def check_output_ownership(function):
    """为row deps检查真实写入分支与全部八组地址，不复用旧两半head公式。"""
    branches = [node for node in function.body if isinstance(node, ast.If)]
    if len(branches) != 1 or ast.unparse(branches[0].test) != "lane == 0":
        raise ValueError("AIV角色分支改变")
    branch = branches[0]
    stores = []
    for side in (branch.body, branch.orelse):
        stores.append([node for statement in side for node in ast.walk(statement)
                       if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.store"])
    if any(ast.unparse(node.args[2]) != "kv_transfer_inline4342__ssa_v0" for node in stores[0]):
        raise ValueError("AIV0不应写packed或softmax结果")
    packed = [node for node in stores[1] if ast.unparse(node.args[2]) == "o_packed_heads__ssa_v0"]
    publication_count = sum(isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.row_expand_div"
                            for node in ast.walk(function))
    if publication_count not in (4, 8):
        raise ValueError("发布head Tile不是已核对的8或16")
    head_tile = 64 // publication_count
    if head_tile == 8:
        expected = [f"pl.tile.store(pl.tile.reshape(pub_result, [1, 4096]), [{chunk * 384} + query, 0], o_packed_heads__ssa_v0)"
                    for chunk in range(8)]
    else:
        packing = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "pub_packed" for target in node.targets)]
        if len(packing) != 4 or any(ast.unparse(node.value) != "pl.tile.reshape(pub_result, [2, 4096])" for node in packing):
            raise ValueError("16head发布必须明确拆成原有两个8head packed组")
        expected = [f"pl.tile.store(pl.tile.slice(pub_packed, [1, 4096], [{group % 2}, 0]), [{group * 384} + query, 0], o_packed_heads__ssa_v0)"
                    for group in range(8)]
    if [ast.unparse(node) for node in packed] != expected:
        raise ValueError("AIV1未独占完整八组packed行")
    if len([node for node in ast.walk(function) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "pl.tile.store"
            and ast.unparse(node.args[2]) == "o_packed_heads__ssa_v0"]) != 8:
        raise ValueError("角色分支外出现packed写入")
    return ["lane == 0", "(t_dim_inline4335__ssa_v0 - core + 23) // 24",
            "core + work // 5 * 24", *expected]


def use_roles(schedule, catalog, output, *, head_tile=8, softmax_tile=8, publication_tile=None):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("双AIV分工候选先验证T96")
    calls = [call for call in schedule["call_bindings"].values() if call["logical_function"] == "qk_pv"]
    table = next(arg["value"] for arg in calls[0]["arguments"] if arg["parameter"] == "cmp_block_table__ssa_v0")
    if table["strides"][1] != 1 or table["dtype"] != "INT32":
        raise ValueError("页表需要连续INT32列")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    functions = {node.name: node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}
    before = {name: ast.dump(node) for name, node in functions.items()}
    fn = functions["qk_pv_aiv"]
    aliases = dict(zip(
        ("tokens", "original_block_idx", "original_subblock_idx", "ffts", "q_flat", "valid", "kv_transfer",
         "score_transfer", "probability_transfer", "pv_transfer", "sink", "positions", "raw_table", "raw_kv",
         "indices", "cmp_table", "cmp_kv", "bias", "mi_transfer", "li_transfer", "cos", "sin_signed", "packed"),
        (arg.arg for arg in fn.args.args), strict=True))
    source = body_source(table["shape"][1], head_tile=head_tile, softmax_tile=softmax_tile,
                         publication_tile=publication_tile)
    tree = ast.parse(source)

    class BindArguments(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in aliases:
                node.id = aliases[node.id]
            return node

    BindArguments().visit(tree)
    fn.body = [copy.deepcopy(fn.body[0]), *tree.body, copy.deepcopy(fn.body[-1])]
    check_output_ownership(fn)
    if any(ast.dump(node) != before[name] for name, node in functions.items() if name != fn.name):
        raise ValueError("意外修改其他函数")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "core_bodies_qk_roles.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      + ast.unparse(ast.fix_missing_locations(module)) + "\n")
    (directory / "role_body_readable.py").write_text(source + "\n")
    audit = {"aiv0_role": "gather_all_128_kv_rows", "aiv1_role": "all_64_head_softmax_pv_and_publish",
             "running_state_bytes": 64 * 512 * 4, "state_tile_shape": [head_tile, 256],
             "softmax_head_tile": softmax_tile,
             "publish_head_tile": head_tile if publication_tile is None else publication_tile,
             "table_columns": table["shape"][1], "aic_and_other_function_asts_unchanged": True,
             "new_gm_buffers": False, "both_aiv_participate_in_original_ffts": True,
             "packed_output_owner": "AIV1_only", "old_row_dependencies_compatible": False,
             "cpu_codegen_validated": False, "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
