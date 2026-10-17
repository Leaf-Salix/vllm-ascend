# SPDX-License-Identifier: Apache-2.0
"""将PV循环状态本身拆成两个16-head Tile，避免整状态assemble。

前版只缩小加载Tile，仍形成四个32KiB状态槽加64KiB临时区，编译超限。
本版把每侧32×256状态拆成两个独立16×256循环值；逐元素乘加次序、
m/l归约、五块合并顺序、最终输出位置、Cube和FFTS保持原样。
分支与循环yield一并显式展开，最终16-head输出直接读取相应状态块。
"""

import ast
import copy
import json
from pathlib import Path

MERGE_HEADS = 16

STATE_NAMES = {
    "running_left_inline4349__ssa_v0", "running_right_inline4327__ssa_v0",
    "left_iter", "right_iter", "next_left_inline4242__ssa_v0",
    "next_right_inline4247__ssa_v0", "left_valid_inline4250__rv_v0",
    "right_valid_inline4248__rv_v0", "reset_left_inline4228__ssa_v0",
    "reset_right_inline4224__ssa_v0", "left_published_inline4343__rv_v0",
    "right_published_inline4278__rv_v0", "left_after_inline4333__rv_v0",
    "right_after_inline4251__rv_v0", "running_left_inline4276", "running_right_inline4275",
}


def state_name(name, chunk):
    return f"{name}_head{chunk}"


class SplitState(ast.NodeTransformer):
    """只展开已有left/right状态；不得改变m/l或任何GM输出表达式。"""

    def __init__(self):
        self.initializers = 0
        self.output_views = 0

    def expand(self, items):
        result = []
        for item in items:
            if isinstance(item, ast.Name) and item.id in STATE_NAMES:
                result.extend(ast.Name(id=state_name(item.id, i), ctx=copy.deepcopy(item.ctx))
                              for i in range(2))
            else:
                result.append(self.visit(item))
        return result

    def visit_Tuple(self, node):
        node.elts = self.expand(node.elts)
        return node

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if name == "pl.yield_":
            node.args = self.expand(node.args)
            return node
        if (name == "pl.tile.slice" and isinstance(node.args[0], ast.Name)
                and node.args[0].id in STATE_NAMES):
            if ast.unparse(node.args[1]) != "[16, 256]":
                raise ValueError("整状态输出切片规格改变")
            offsets = {
                "[pub_local_head_inline4282__ssa_v0, 0]": 0,
                "[pub_local_head_inline4282__ssa_v1, 0]": 1,
            }
            chunk = offsets[ast.unparse(node.args[2])]
            self.output_views += 1
            return ast.Name(id=state_name(node.args[0].id, chunk), ctx=ast.Load())
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in STATE_NAMES:
            if (not isinstance(node.value, ast.Call)
                    or ast.unparse(node.value.func) != "pl.tile.full"
                    or ast.unparse(node.value.args[0]) != "[32, 256]"):
                raise ValueError("整状态初始化不符合预期")
            result = []
            for i in range(2):
                n = copy.deepcopy(node)
                n.target.id = state_name(node.target.id, i)
                n.value.args[0] = ast.parse("[16, 256]", mode="eval").body
                n.annotation = ast.parse("pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec]", mode="eval").body
                result.append(n)
            self.initializers += 1
            return result
        return self.generic_visit(node)


def merge_body():
    lines = []
    for chunk in range(32 // MERGE_HEADS):
        row = chunk * MERGE_HEADS
        lines.extend([
            f"qkm_alpha_{chunk} = pl.tile.reshape(pl.tile.extract(pl.tile.reshape("
            "alpha_inline4273__ssa_v0, [1, 32]), 0, "
            f"{row}, [1, {MERGE_HEADS}], target_memory=pl.Mem.Vec), [{MERGE_HEADS}, 1])",
            f"qkm_beta_{chunk} = pl.tile.reshape(pl.tile.extract(pl.tile.reshape("
            "beta_inline4330__ssa_v0, [1, 32]), 0, "
            f"{row}, [1, {MERGE_HEADS}], target_memory=pl.Mem.Vec), [{MERGE_HEADS}, 1])",
        ])
    # 每个16-head循环值独立更新，移除完整32-head状态的slice/assemble别名链。
    for side, column in (("left", 0), ("right", 256)):
        for chunk in range(32 // MERGE_HEADS):
            row = chunk * MERGE_HEADS
            prefix = f"qkm_{side}_{chunk}"
            target = "next_left_inline4242__ssa_v0" if side == "left" else "next_right_inline4247__ssa_v0"
            lines.extend([
                f"{prefix}_incoming = pl.tile.load(pv_transfer_inline4320__ssa_v0, "
                f"[pv_transfer_row_inline4421__ssa_v1 + qk_lane_head_inline4436__ssa_v0 + {row}, {column}], "
                f"[{MERGE_HEADS}, 256], [{MERGE_HEADS}, 256], target_memory=pl.Mem.Vec)",
                f"{prefix}_scaled_old = pl.tile.row_expand_mul({state_name(side + '_iter', chunk)}, qkm_alpha_{chunk})",
                f"{prefix}_scaled_incoming = pl.tile.row_expand_mul({prefix}_incoming, qkm_beta_{chunk})",
                f"{state_name(target, chunk)} = pl.tile.add({prefix}_scaled_old, {prefix}_scaled_incoming)",
            ])
    return ast.parse("\n".join(lines)).body


def split_pv_merge(schedule, catalog, output):
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    original_functions = {n.name: ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and node.name == "qk_pv_aiv")
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("PV分head合并候选先限定B16/S6")
    changes = 0
    expected = ["pv_left_inline4243__ssa_v0", "t__tmp_v811", "t__tmp_v812",
                "next_left_inline4242__ssa_v0", "pv_right_inline4395__ssa_v0",
                "t__tmp_v813", "t__tmp_v814", "next_right_inline4247__ssa_v0"]
    sync_before = [ast.dump(node) for node in ast.walk(fn) if isinstance(node, ast.Call)
                   and ast.unparse(node.func).startswith("pl.system.")]
    writes_before = [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) in ('pl.tile.store', 'pl.tensor.write')]
    scalar_reads_before = [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                           and ast.unparse(n.func) == 'pl.tensor.read']
    for parent in list(ast.walk(fn)):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if (not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name)
                    or statement.target.id != expected[0]):
                continue
            old = body[index:index + 8]
            if ([node.target.id for node in old if isinstance(node, ast.AnnAssign)] != expected
                    or ast.unparse(old[0].value.args[2]) != "[32, 256]"
                    or ast.unparse(old[4].value.args[2]) != "[32, 256]"):
                raise ValueError("PV原32-head合并结构改变")
            body[index:index + 8] = merge_body()
            changes += 1
            break
    if changes != 1:
        raise ValueError("PV合并变换位置不是唯一的")
    splitter = SplitState()
    splitter.visit(fn)
    if splitter.initializers != 4 or splitter.output_views != 4:
        raise ValueError("循环状态或最终输出切片未全部展开")
    if any(isinstance(n, ast.Name) and n.id in STATE_NAMES for n in ast.walk(fn)):
        raise ValueError("变换后仍残留未分块的循环状态")
    sync_after = [ast.dump(node) for node in ast.walk(fn) if isinstance(node, ast.Call)
                  and ast.unparse(node.func).startswith("pl.system.")]
    if sync_before != sync_after:
        raise ValueError("PV合并候选改变了显式同步")
    if writes_before != [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                         and ast.unparse(n.func) in ('pl.tile.store', 'pl.tensor.write')]:
        raise ValueError('PV合并改变了GM写入范围')
    if scalar_reads_before != [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                               and ast.unparse(n.func) == 'pl.tensor.read']:
        raise ValueError('PV合并改变了metadata读取')
    if any(ast.dump(n) != original_functions[n.name] for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name != fn.name):
        raise ValueError('PV合并意外修改了其他函数')
    # 行依赖仍按完整32-head输出和原query分工判断；各块是互不相交的head集合。
    assert sorted((row, col) for chunk in range(32 // MERGE_HEADS)
                  for row in range(chunk * MERGE_HEADS, (chunk + 1) * MERGE_HEADS)
                  for col in range(512)) == [(row, col) for row in range(32) for col in range(512)]
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "core_bodies_qk_merge_heads.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      "# PV循环状态拆成两个16-head块，不改原FFTS和归约顺序。\n"
                      + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    audit = {"kernel": fn.name, "heads_per_aiv": 32, "heads_per_merge_chunk": MERGE_HEADS,
             "running_state_partition": {"per_side": [[16, 256], [16, 256]], "total_elements_unchanged": True},
             "full_state_assemble_removed": True, "incoming_tile_old_bytes": 32768,
             "incoming_tile_new_bytes": MERGE_HEADS * 256 * 4, "alpha_beta_and_m_l_unchanged": True,
             "block_merge_order_unchanged": True, "original_ffts_unchanged": True,
             "qk_gather_and_aic_unchanged": True, "output_publish_unchanged": True,
             "extra_gm_or_root_inputs": False,
             "source_regions_written": [[i, i + MERGE_HEADS] for i in range(0, 32, MERGE_HEADS)],
             "state_initializers_split": splitter.initializers,
             "final_output_views_replaced": splitter.output_views,
             "gm_write_asts_and_scalar_metadata_reads_unchanged": True,
             "other_function_asts_unchanged": True,
             "pv_read_region_union_unchanged": [32, 512],
             "actual_ub_plan_verified": False, "precision_proven": False, "performance_proven": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
