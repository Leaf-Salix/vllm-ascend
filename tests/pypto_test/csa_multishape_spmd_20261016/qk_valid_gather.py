# SPDX-License-Identifier: Apache-2.0
"""QK 全有效压缩块直接覆盖缓冲，负索引块保留原清零与逐行回退。

与此前 full_block 候选的区别：整块最小值确认无负索引之后，64 次搬运
覆盖所有元素，不再执行分支外和压缩分支内的两次清零。SWA 仍初始化。
此模块只生成实验源码，是否获益须用完整 CSA 的 L1 测量确认。
"""

import ast
import copy
import json
from pathlib import Path

from qk_full_block import use_full_block
from qk_scoped_zero import scope_zero


def use_valid_gather(catalog, output):
    directory = Path(output)
    scoped, _ = scope_zero(catalog, directory / "scoped")
    full, _ = use_full_block(scoped, directory / "full")
    module = ast.parse(Path(full["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "qk_pv_aiv")
    branch = next(node for node in ast.walk(fn)
                  if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "qk_sb_inline4367__ssa_v1 == 0")
    zero = branch.orelse[0]
    if (not isinstance(zero, ast.AnnAssign)
            or zero.target.id != "qk_kv_half_inline4272__ssa_v0"
            or ast.unparse(zero.value.func) != "pl.tile.full"):
        raise ValueError("压缩块原清零结构改变")
    select = next(node for node in branch.orelse if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "pl.tile.read(qk_full_min, [0, 0]) >= 0")
    # 在分支前创建共同的目标，避免两条互斥路径各持有一份 64 KiB 分配。
    # create 不产生初始化指令；全有效时直接覆盖。
    # 回退将有效行设为0再原地填充，整块均为padding，因此得到全零缓冲。
    # fillpad_inplace明确保证复用输入，而普通expands没有这个别名保证。
    shared = ast.parse(
        "qk_valid_blank = pl.tile.create([64, 512], dtype=pl.BF16, target_memory=pl.Mem.Vec)"
    ).body[0]
    blank = copy.deepcopy(zero)
    blank.value = ast.Name(id="qk_valid_blank", ctx=ast.Load())
    zero.value = ast.parse(
        "pl.tile.fillpad_inplace(pl.tile.set_validshape(qk_valid_blank, 0, 512))",
        mode="eval").body
    branch.orelse.insert(branch.orelse.index(select), shared)
    select.body.insert(0, blank)
    select.orelse.insert(0, zero)
    branch.orelse.pop(0)
    # 原地填充带 pad=zero 元数据，直接覆盖路径没有此标记。三条互斥路径
    # 的局部 SSA 各用独立名字，避免嵌套生成后被识别为同名值的类型重赋值。
    for statements, name in ((select.body, "qk_valid_tile_start"),
                             (select.orelse, "qk_fallback_tile_start")):
        for statement in statements:
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and node.id == "qk_kv_half_inline4272__ssa_v0":
                    node.id = name
    # 三条路径都输出完全有效的64x512 Tile，因此不存在需要填充的区域。
    # 用恒等slice统一pad类型标记，避免静态入口展开yield时出现类型重赋值；
    # 这只是别名视图，不执行填充或拷贝。负索引路径的实际填零仍由上面完成。
    for statement in (select.body[-1], branch.body[-1]):
        value = statement.value
        if not isinstance(value, ast.Call) or ast.unparse(value.func) != "pl.yield_":
            raise ValueError("全有效/SWA分支的结果结构改变")
        value.args[0] = ast.parse(
            f"pl.tile.slice({ast.unparse(value.args[0])}, [64, 512], [0, 0], "
            "pad_value=pl.PadValue.zero)", mode="eval").body
    ast.fix_missing_locations(module)
    path = directory / "core_bodies_qk_valid_gather.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# 全有效块不清零；负索引保持原清零回退。\n"
                    + ast.unparse(module) + "\n")
    audit = {"kernel": fn.name, "rows_per_block": 64,
             "validity": "device INT32 row_min >= 0",
             "full_valid_zero_fills": 0, "fallback_zero_fills": 1,
             "swa_zero_fills": 1, "unroll_factor": 8,
             "shared_destination_across_validity_branches": True,
             "fallback_zero_method": "fillpad_inplace(valid_rows=0, valid_cols=512)",
             "full_valid_results_share_pad_type_via_identity_view": True,
             "fallback": "原64行条件循环，结果变量重命名",
             "qk_math_ffts_unchanged": True, "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit


def boundary_body(catalog):
    """提取实际候选的规约和两条搬运分支，供独立负索引检查复用。"""
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "qk_pv_aiv")
    branch = next(node for node in ast.walk(fn)
                  if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "qk_sb_inline4367__ssa_v1 == 0")
    start = next(i for i, node in enumerate(branch.orelse)
                 if isinstance(node, ast.Assign) and node.targets[0].id == "qk_full_base")
    end = next(i for i, node in enumerate(branch.orelse)
               if isinstance(node, ast.If)
               and ast.unparse(node.test) == "pl.tile.read(qk_full_min, [0, 0]) >= 0")
    return copy.deepcopy(branch.orelse[start:end + 1])
