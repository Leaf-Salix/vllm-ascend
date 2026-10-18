#!/usr/bin/env python3
"""仅用标准库复制/提取源码并核对AST；不导入PyPTO，不编译或运行设备。"""

import argparse
import ast
import copy
import difflib
import io
import json
from pathlib import Path
import textwrap
import tokenize


HERE = Path(__file__).resolve().parent
NAME = "indexer_score_cube_step_g6"
STRATEGIES = {
    "query_group_size": ("G6_QUERY_GROUP", 6),
    "qk_panel_cols": ("G6_QK_PANEL", 64),
    "score_tile": ("G6_SCORE_TILE", 1024),
    "key_prefetch_panels": ("G6_PREFETCH", 1),
    "key_prefetch_to_l0": ("G6_TO_L0", True),
    "balance_leaves": ("G6_BALANCED", True),
}
OVERVIEW = """
独立单步源码准备
    indexer_score_cube_step_g6只提取AIC的一次完整score step，固定数学策略为
    G6/N64/score1024/balanced/prefetch1/toL0；原Cube/Vector整leaf入口不变，
    尚未由任何schedule调用。本入口显式接收只读Left Query[384,128] INT8
    与Coeff[16,384] FP16，原始Native Tensor和页/leaf/step地址仍在设备动态
    计算。caller必须先按原leaf初始化set_validshape(384,128)并补齐尾组，
    再在每次调用前将驻Mat Query重新TMOV成full-valid Left；Coeff也须保持
    完整有效，不能被Attention覆盖。这里不支持任意动态valid的Tile入参。
    step内部保留复用ACK、所有Key页/16个N64 panel、末WS及FIX READY；
    leaf最后min(steps,2)个ACK drain仍由caller完成，不能把helper返回当完成。
    新入口仅为typed-Tile源码准备，尚无PyPTO/CCEC/设备可调用性证据。
"""
SIGNATURE = """@pl.jit.inline(auto_scope=False)
def indexer_score_cube_step_g6(
    buf_query_left: pl.Tile[[384, 128], pl.INT8, pl.MemorySpace.Left],
    buf_coefficient_pair: pl.Tile[[16, 384], pl.FP16, pl.MemorySpace.Left],
    idx_kv_cache: pl.Tensor[[KEY0_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_cache_shift64: pl.Tensor[[KEY64_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    buf_score_transfer: pl.InOut[pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32]],
    buf_worker: pl.Scalar[pl.INDEX],
    buf_batch_idx: pl.Scalar[pl.INDEX],
    buf_cache_len: pl.Scalar[pl.INDEX],
    buf_logical_begin: pl.Scalar[pl.INDEX],
    buf_lane_span: pl.Scalar[pl.INDEX],
    buf_score_step: pl.Scalar[pl.INDEX],
    native_page_bytes: pl.Scalar[pl.INDEX],
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
) -> pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32]:
    \"\"\"执行原balanced G6的一步；只读两份Left，返回原FP32 transfer别名。

    caller已取得query/系数/Native cache和plan，设置相同FFTS域，并从同一
    leaf几何给出worker、batch、cache_len、logical_begin、lane_span与step。
    native_page_bytes必须是Native cache实际第二维，不能固定为某一页容量。
    Query先沿原leaf初始化set_validshape(384,128)并补齐，再TMOV传full-valid
    Left；Coeff保持完整16x384及无效query零系数，非任意动态valid Tile接口。
    本入口不重新加载Query/Coeff、不前移READY、不替代两AIV消费或根发布。
    buf_lane_span仅为保留原完整step AST而显式传入；balanced地址仍由原
    balance分支覆盖，不把该值改写成Host已知shape。六个固定策略常量保留
    原if/unroll形态，由后续前端按数学策略折叠，不在此生成展开的panel代码。
    \"\"\"
    # step>=2先等两个AIV的原MTE2 ACK，随后才复用本组双槽。
"""


def find_step(tree):
    stage = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_cube_stage")
    matches = [n for n in ast.walk(stage) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
               and n.target.id == "buf_score_step"]
    assert len(matches) == 1
    return matches[0]


def rename_tokens(source):
    """只改六个NAME token，原注释/字符串、算式、分支和调用顺序均保留。"""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in STRATEGIES:
            begin = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            edits.append((begin, end, STRATEGIES[token.string][0]))
    for begin, end, value in reversed(edits):
        source = source[:begin] + value + source[end:]
    return source


class ReverseStrategyNames(ast.NodeTransformer):
    def visit_Name(self, node):
        inverse = {new: old for old, (new, _value) in STRATEGIES.items()}
        if node.id in inverse:
            node.id = inverse[node.id]
        return node


def dump_body(body):
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def prepare(reference_path):
    original = reference_path.read_text()
    tree = ast.parse(original)
    step = find_step(tree)
    lines = original.splitlines(keepends=True)
    step_source = textwrap.dedent("".join(lines[step.body[0].lineno - 1:step.body[-1].end_lineno]))
    constants = "\n".join(f"{name} = {value!r}" for name, value in STRATEGIES.values())
    body = textwrap.indent(rename_tokens(step_source), "    ")
    ready = "    pl.system.sync_set(SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)"
    assert body.count(ready) == 1
    body = body.replace(ready, "    # 最后WS已store后仍由FIX发布READY；叶末ACK drain留给caller。\n" + ready)
    added = "\n\n# 只固定数学tile策略，T/B/Q/L/页表/leaf地址与有效范围继续动态。\n" + constants
    added += "\n\n\n" + SIGNATURE + body + "    return buf_score_transfer\n"
    anchor = "\n\n@pl.jit.inline(auto_scope=False)\ndef indexer_score_vector_lane_stage("
    assert original.count(anchor) == 1
    candidate = original.replace(anchor, added + anchor, 1)
    overview_anchor = "\nimport pypto.language as pl\n"
    assert candidate.count(overview_anchor) == 1
    doc = tree.body[0]
    assert isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
    doc_end = doc.end_lineno - 1
    candidate_lines = candidate.splitlines(keepends=True)
    assert candidate_lines[doc_end].strip() == '\"\"\"'
    candidate_lines.insert(doc_end, OVERVIEW)
    candidate = "".join(candidate_lines)

    destination = HERE / "source_candidate" / reference_path.name
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(candidate)
    assert candidate.replace(OVERVIEW, "", 1).replace(added, "", 1) == original
    actual_tree = ast.parse(candidate)
    helper = next(n for n in actual_tree.body if isinstance(n, ast.FunctionDef) and n.name == NAME)
    assert isinstance(helper.body[0], ast.Expr) and isinstance(helper.body[0].value, ast.Constant)
    assert isinstance(helper.body[-1], ast.Return) and ast.unparse(helper.body[-1].value) == "buf_score_transfer"
    normalized = ReverseStrategyNames().visit(ast.Module(body=copy.deepcopy(helper.body[1:-1]), type_ignores=[])).body
    assert dump_body(normalized) == dump_body(step.body), "完整step数学/控制/同步AST发生变化"

    assigned = {n.id for n in ast.walk(ast.Module(body=helper.body, type_ignores=[]))
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loaded = {n.id for n in ast.walk(ast.Module(body=helper.body, type_ignores=[]))
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    globals_available = {n.targets[0].id for n in actual_tree.body
                         if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    globals_available.add("pl")
    arguments = {a.arg for a in helper.args.args}
    assert not (loaded - assigned - globals_available - arguments), "存在未显式传入的动态依赖"
    assert arguments == loaded - assigned - globals_available, "存在无用或遗漏参数"
    for name in ("buf_query_left", "buf_coefficient_pair"):
        assert name not in assigned
        uses = [n for n in ast.walk(helper) if isinstance(n, ast.Call)
                and any(isinstance(a, ast.Name) and a.id == name for a in n.args)]
        assert uses and all(ast.unparse(n.func) == "pl.tile.matmul" and n.args[0].id == name for n in uses)

    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    actual_functions = {n.name: n for n in actual_tree.body if isinstance(n, ast.FunctionDef)}
    assert set(actual_functions) == set(functions) | {NAME}
    for name, function in functions.items():
        assert ast.get_source_segment(original, function) == ast.get_source_segment(candidate, actual_functions[name])
    syntax_calls = [ast.unparse(n.func) for n in ast.walk(helper) if isinstance(n, ast.Call)]
    report = {
        "passed": True,
        "scope": "源码提取及直接AST对照，不导入/编译PyPTO，不运行设备",
        "reference": str(reference_path), "candidate": str(destination),
        "reference_bytes": reference_path.stat().st_size, "candidate_bytes": destination.stat().st_size,
        "helper": NAME, "source_step_lines": [step.lineno, step.end_lineno],
        "strategy_name_mapping": {old: {"constant": new, "value": value} for old, (new, value) in STRATEGIES.items()},
        "complete_step_ast_equal_after_only_strategy_name_reversal": True,
        "complete_original_file_text_restored_after_removing_added_module_doc_constants_helper": True,
        "existing_functions_text_unchanged": sorted(functions),
        "docstring_change": "只在模块总览新增一段，并为新增helper编写docstring；不声称完整文件AST原样一致",
        "typed_tile_inputs": {a.arg: ast.unparse(a.annotation) for a in helper.args.args[:2]},
        "readonly_tile_uses_are_matmul_lhs_only": True,
        "explicit_arguments": {a.arg: ast.unparse(a.annotation) for a in helper.args.args},
        "unresolved_free_names": [], "source_step_statement_count": len(step.body),
        "sync_wait_syntax_sites": syntax_calls.count("pl.system.sync_wait"),
        "sync_set_syntax_sites": syntax_calls.count("pl.system.sync_set"),
        "panel_loop_retained_without_manual_expansion": True,
        "leaf_initialization_and_tail_ack_drain_not_extracted": True,
        "schedule_callers_added": False,
        "pypto_compilation_performed": False, "device_execution": False,
        "typed_tile_callability_verified": False,
    }
    (HERE / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (HERE / "candidate.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="source_reference/indexer_score_stages.py", tofile="source_candidate/indexer_score_stages.py")))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = HERE / "source_reference" / "indexer_score_stages.py"
    if args.reference:
        reference.parent.mkdir(exist_ok=True)
        contents = args.reference.resolve().read_bytes()
        if reference.exists():
            assert reference.read_bytes() == contents, "不得覆盖不同的冻结参考"
        else:
            reference.write_bytes(contents)
        (HERE / "reference_origin.json").write_text(json.dumps({
            "path": str(args.reference.resolve()), "bytes": len(contents),
            "snapshot": str(reference), "direct_bytes_equal": reference.read_bytes() == contents,
        }, ensure_ascii=False, indent=2) + "\n")
    prepare(reference)


if __name__ == "__main__":
    main()
