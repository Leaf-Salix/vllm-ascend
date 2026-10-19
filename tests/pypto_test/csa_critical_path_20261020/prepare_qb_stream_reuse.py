# SPDX-License-Identifier: Apache-2.0
"""构造首个M块边算边驻留权重的完整动态CSA候选，不修改正式源码。

与此前整K前置驻留的区别是：四份K256权重分别加载，后两份置于前两次
矩阵计算之后；保留它们的Mat句柄给后续M块。生成代码审计必须确认没有
把四份载入又全部移到首个矩阵乘之前，且Right仍只有一份64KiB。
"""

import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FLAG = "QB_STREAM_REUSE"


def wire(source, definitions, calls):
    """只在指定签名和调用尾部传constexpr，不格式化其它源码。"""
    lines = source.splitlines(keepends=True)
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    edits, seen_defs, seen_calls = [], set(), set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in definitions:
            last = node.args.defaults[-1] if node.args.defaults else node.args.args[-1]
            edits.append((offsets[last.end_lineno], f"    {FLAG}: pl.constexpr = False,\n"))
            seen_defs.add(node.name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in calls:
            seen_calls.add(node.func.id)
            if node.lineno == node.end_lineno:
                close = offsets[node.end_lineno - 1] + node.end_col_offset - 1
                assert source[close] == ")"
                edits.append((close, f", {FLAG}"))
            else:
                close = offsets[node.end_lineno - 1]
                assert source[:close].rstrip().endswith(",")
                line = lines[node.args[-1].lineno - 1]
                indent = line[:len(line) - len(line.lstrip())]
                edits.append((close, f"{indent}{FLAG},\n"))
    assert seen_defs == set(definitions) and seen_calls == set(calls)
    for offset, addition in sorted(edits, reverse=True):
        source = source[:offset] + addition + source[offset:]
    ast.parse(source)
    return source


def qr_load(index, row, rows, indent, prefix="stream"):
    return f"""{indent}{prefix}_qr{index} = pl.load(
{indent}    qr_i8_matmul, [{row}, {index} * QPROJ_PIPE_K_TILE],
{indent}    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
{indent}    valid_shape=[{rows}, QPROJ_PIPE_K_TILE], target_memory=pl.MemorySpace.Mat,
{indent})
"""


def weight_load(index, indent):
    return f"""{indent}stream_w{index} = pl.load(
{indent}    wq_b, [{index} * QPROJ_PIPE_K_TILE, stream_col],
{indent}    [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Mat,
{indent})
"""


def mma(index, indent, prefix="stream"):
    result = f"""{indent}{prefix}_right{index} = pl.tile.extract(
{indent}    stream_w{index}, 0, 0, [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
{indent}    target_memory=pl.MemorySpace.Right,
{indent})
"""
    if index == 0:
        return result + f"{indent}{prefix}_acc = pl.tile.matmul({prefix}_qr0, {prefix}_right0)\n"
    return result + f"{indent}{prefix}_acc = pl.tile.matmul_acc({prefix}_acc, {prefix}_qr{index}, {prefix}_right{index})\n"


def branch():
    code = """    if QB_STREAM_REUSE:
        # 仅长历史且Host已有T容量处于两个M128块区间时启用。这里仍读动态
        # tokens控制有效行和剩余M循环；不以B28、S6或历史测试值替代device值。
        # 四份Mat权重各64KiB，跨M保持活跃；QR最多两份32KiB，Right单份。
        # 不用pipeline：避免其IO重排提前两个Right载入而超过64KiB上限。
        for stream_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + WORKERS - 1) // WORKERS):
            stream_n = qproj_worker + stream_round * WORKERS
            stream_col = stream_n * QPROJ_MM_N_TILE
            stream_first_rows = pl.min(QPROJ_PIPE_M_TILE, tokens)
            # 首次只预载两段；后两段在已有MMA发出后载入，保留首M供数流水。
"""
    indent = "            "
    for i in (0, 1):
        code += weight_load(i, indent) + qr_load(i, "0", "stream_first_rows", indent)
    code += mma(0, indent)
    code += weight_load(2, indent) + qr_load(2, "0", "stream_first_rows", indent)
    code += mma(1, indent)
    code += weight_load(3, indent) + qr_load(3, "0", "stream_first_rows", indent)
    code += mma(2, indent) + mma(3, indent)
    code += """            q_proj_i32 = pl.store(stream_acc, [0, stream_col], q_proj_i32)
            # 剩余M只载QR，四份权重从本核L1取；每个输出的K256累加顺序不变。
            # 循环长度由实际tokens决定，保留尾行valid，四份权重不跨列块复用。
            for stream_t0 in pl.range(QPROJ_PIPE_M_TILE, qproj_t_matmul, QPROJ_PIPE_M_TILE):
                stream_rows = pl.min(QPROJ_PIPE_M_TILE, tokens - stream_t0)
"""
    indent = "                "
    code += qr_load(0, "stream_t0", "stream_rows", indent, "reuse")
    code += qr_load(1, "stream_t0", "stream_rows", indent, "reuse")
    code += mma(0, indent, "reuse")
    code += qr_load(2, "stream_t0", "stream_rows", indent, "reuse") + mma(1, indent, "reuse")
    code += qr_load(3, "stream_t0", "stream_rows", indent, "reuse") + mma(2, indent, "reuse") + mma(3, indent, "reuse")
    code += "                q_proj_i32 = pl.store(reuse_acc, [stream_t0, stream_col], q_proj_i32)\n"
    return code


def main():
    target = HERE / "source_qb_stream_reuse"
    target.mkdir(exist_ok=False)
    for path in (HERE / "source_reference").glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    for name, definitions, calls in (
        ("entry.py", ("_csa_core_group", "decode_csa_spmd"), ("_csa_core_group", "cube_schedule")),
        ("cube_schedule.py", ("cube_schedule",), ("projection_overlap_stage", "q_b_stage")),
        ("projection_overlap_stages.py", ("projection_overlap_stage",), ("q_b_stage",)),
        ("query_projection_stages.py", ("_q_b_nd_stage", "_q_b_nz_stage"), ()),
    ):
        path = target / name
        path.write_text(wire(path.read_text(), definitions, calls))
    path = target / "query_projection_stages.py"
    source = path.read_text()
    start = source.index("    for qproj_round in pl.range(", source.index("def _q_b_nz_stage("))
    end = source.index("    return q_proj_i32", start)
    original = source[start:end]
    source = source[:start] + branch() + "    else:\n" + "".join("    " + line for line in original.splitlines(keepends=True)) + source[end:]
    doc_end = source.index('"""', source.index('"""') + 3)
    source = source[:doc_end] + """
候选变更总览
    QB_STREAM_REUSE仅改变NZ权重搬运的生命周期：第一块M边加载边计算并保留
    四段K权重，后续M仅重载QR。与旧整列前置驻留不同，不等待完整K载入才
    开始矩阵计算。保留M128/N256/K256、真实尾行、原20核列owner与全部同步。
    只在独立候选包验证；数学/资源/完整L1通过前不进入正式算子。
""" + source[doc_end:]
    path.write_text(source)
    path = target / "policy.py"
    source = path.read_text()
    for old, new in (
        ("    plan_physical_rows: bool = False\n", "    plan_physical_rows: bool = False\n    qb_stream_reuse: bool = False\n"),
        ('            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n', '            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n            "QB_STREAM_REUSE": self.qb_stream_reuse,\n'),
        ("        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,\n", """        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,
        # 长历史、两个M128块的资源区间；Host只读既有容量描述符。
        # 首次M载入完整四段后供第二M复用，短历史先保留原路径等待扩档门槛。
        qb_stream_reuse=long_history and 128 < tokens <= 256,
"""),
    ):
        assert source.count(old) == 1
        source = source.replace(old, new)
    path.write_text(source)
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    spec = importlib.util.spec_from_file_location("qb_stream_policy", target / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    values = {}
    for b in (16, 28):
        values[str(b)] = module.select_policy(tokens=b*6, requests=b, max_query_tokens=6, max_seq_len=131078).constexprs()
        assert values[str(b)][FLAG] == (b == 28)
        (HERE / f"qb_stream_controls_b{b}.json").write_text(json.dumps(values[str(b)], indent=2) + "\n")
    changed = [p.name for p in target.glob("*.py") if p.read_bytes() != (HERE / "source_reference" / p.name).read_bytes()]
    report = dict(base="source_reference/E49", changed=sorted(changed), controls=values,
                  requested_weight_mat_bytes=4*256*256, requested_qr_mat_bytes=2*128*256,
                  original_math_tiles=[128,256,256], source_shapes_dynamic=True,
                  formal_changed=False, device_validated=False, performance_measured=False)
    (HERE / "QB_STREAM_PREPARE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({"changed":changed, "B16":values['16'][FLAG], "B28":values['28'][FLAG]})


if __name__ == "__main__":
    main()
