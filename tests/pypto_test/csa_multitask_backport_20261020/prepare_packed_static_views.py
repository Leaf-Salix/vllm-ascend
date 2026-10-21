# SPDX-License-Identifier: Apache-2.0
"""在同一动态容量策略下建立定长GM视图，使两种步长分别编译为常量。"""

import argparse
import ast
import json
import re
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once
from prepare_query_score import append_call_arguments, append_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="packed_capacity_long_v3")
    parser.add_argument("--candidate", default="packed_capacity_static_v6")
    parser.add_argument("--no-constexpr-specialization", action="store_true", help="复现v4泛型返回类型混用失败")
    parser.add_argument("--capture-stride-scalar", action="store_true", help="复现v5将常量赋给普通编排Scalar的额外参数")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / name for name in (args.base, args.candidate))
    source = (base / "decode_csa.py").read_text()
    tree = ast.parse(source)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "sparse_attn_csa_tp1"
    ]
    scopes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.With)
        and any(isinstance(k, ast.Call) and ast.unparse(k.func) == "decode_o_proj_tp1" for k in ast.walk(n))
    ]
    if len(calls) != 1 or len(scopes) != 2:
        # 包含该调用的原外层scope，以及调用附近的输出scope。
        raise ValueError("需要唯一Attention调用和两层原有scope")
    call = calls[0]
    scope = max(scopes, key=lambda n: n.lineno)
    if call.col_offset != scope.col_offset or scope.lineno <= call.end_lineno:
        raise ValueError("Attention和输出scope必须同层相邻")
    lines = source.splitlines(keepends=True)
    tail = textwrap.dedent("".join(lines[call.lineno - 1 : scope.end_lineno]))

    def branch(suffix, rows):
        renamed = tail
        for name in ("o_packed_heads", "heads_dep", "layer_output"):
            renamed = re.sub(r"\b" + name + r"\b", name + "_" + suffix, renamed)
        renamed = renamed.replace(
            f"layer_output_{suffix} = decode_o_proj_tp1(\n",
            f"layer_output_{suffix} = decode_o_proj_tp1(  # noqa: F841 - 分支内SSA绑定，公共输出为x_out\n",
        )
        if not args.no_constexpr_specialization:
            renamed = append_call_arguments(
                renamed, {"sparse_attn_csa_tp1", "decode_o_proj_tp1"}, [f"PACKED_GROUP_ROWS={rows}"]
            )
        # packed连续且策略保证实际容量等于选中分支；reshape仅修改GM描述符。
        return f"o_packed_heads_{suffix} = pl.reshape(o_packed_heads, [O_GROUPS * {rows}, O_GROUP_IN])\n" + renamed

    replacement = (
        "# 容量选择仍在同一算子内；两条分支共用数学helper，只固定内部GM视图。\n"
        "if packed_row_stride == PACKED_ROW_ALIGN:\n"
        + textwrap.indent(branch("compact", "PACKED_ROW_ALIGN"), "    ")
        + "else:\n"
        + textwrap.indent(branch("full", "T_PAD"), "    ")
    )
    lines[call.lineno - 1 : scope.end_lineno] = [textwrap.indent(replacement, " " * call.col_offset)]
    source = "".join(lines)
    source = replace_once(source, "        return layer_output\n", "        return x_out\n")
    changed = {"decode_csa.py": source}
    for name, count in (("decode_sparse_attn_csa.py", 2), ("decode_o_proj.py", 4)):
        source = (base / name).read_text()
        shape = "pl.Tensor[[PACKED_ROWS_DYN, O_GROUP_IN], pl.BF16]"
        if source.count(shape) != count:
            raise ValueError(f"{name}的packed参数数量不符")
        source = source.replace(shape, "pl.Tensor")
        source = replace_once(source, 'PACKED_ROWS_DYN = pl.dynamic("CSA_PACKED_ROWS_DYN")\n', "")
        if not args.no_constexpr_specialization:
            functions = (
                ("sparse_attn_csa", "sparse_attn_csa_tp1")
                if name == "decode_sparse_attn_csa.py"
                else ("_proj_a_mm_nz", "_proj_a_mm_nd", "_decode_o_proj_tp1_tiled", "decode_o_proj_tp1")
            )
            for function_name in functions:
                source = append_parameters(source, function_name, ["PACKED_GROUP_ROWS: pl.constexpr = T_PAD"])
            calls = (
                {"sparse_attn_csa"}
                if name == "decode_sparse_attn_csa.py"
                else {"proj_a_mm", "_decode_o_proj_tp1_tiled"}
            )
            source = append_call_arguments(source, calls, ["PACKED_GROUP_ROWS=PACKED_GROUP_ROWS"])
            old_stride = (
                "packed_row_stride = pl.tensor.dim(o_packed_heads, 0) // O_GROUPS"
                if name == "decode_sparse_attn_csa.py"
                else "packed_row_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS"
            )
            source = replace_once(source, old_stride, "packed_row_stride = PACKED_GROUP_ROWS")
            if not args.capture_stride_scalar:
                source = replace_once(source, "    packed_row_stride = PACKED_GROUP_ROWS\n", "")
                source = re.sub(r"\bpacked_row_stride\b", "PACKED_GROUP_ROWS", source)
        changed[name] = source
    for source in changed.values():
        ast.parse(source)
    shutil.copytree(base, target)
    for name, source in changed.items():
        (target / name).write_text(source)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        change="保留v3容量策略与分配位置；建立128/384行组步长的零复制定长视图，两分支共用Attention及O-A数学",
        expected_codegen="关闭缩容路径的Attention/O-A不再因内部容量增加动态维度/标量参数，必须检查实际生成物",
        scope_note="源码输出scope出现两次但运行时只选一条，不增加scope深度或新的数学task",
        constexpr_specialization=not args.no_constexpr_specialization,
        capture_stride_scalar=args.capture_stride_scalar,
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
