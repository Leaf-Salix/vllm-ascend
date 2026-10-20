# SPDX-License-Identifier: Apache-2.0
"""迁移冻结单SPMD的HC_ROW_DEPENDENCIES到多task局部前段。"""

import argparse
import ast
import json
import shutil

from prepare import HERE, section


def infer_inline_tensor_contracts(source, names=None):
    """inline数学从调用者继承Tensor合同，避免独立dynamic轴残留到局部MIX。"""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = []
    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and (names is None or node.name in names)
    ]
    for node in (argument for function in functions for argument in function.args.args):
        if not isinstance(node, ast.arg) or node.annotation is None:
            continue
        for annotation in ast.walk(node.annotation):
            if (
                isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Attribute)
                and isinstance(annotation.value.value, ast.Name)
                and annotation.value.value.id == "pl"
                and annotation.value.attr == "Tensor"
            ):
                start = offsets[annotation.lineno - 1] + annotation.col_offset
                end = offsets[annotation.end_lineno - 1] + annotation.end_col_offset
                replacements.append((start, end, "pl.Tensor"))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="hc_rows_v1")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    source = (HERE / "sources/spmd/hc_stages.py").read_text()
    source = source.replace("from ..deepseek_v4_flash_csa.rmsnorm import", "from .rmsnorm import")
    source = infer_inline_tensor_contracts(source)
    (target / "hc_stages.py").write_text(source)
    source = (HERE / "sources/spmd/phase_sync.py").read_text()
    source = source.replace('SIGNAL_WORDS = pl.dynamic("CSA_PHASE_SIGNAL_WORDS")', "SIGNAL_WORDS = 4 * 72 * 128", 1)
    (target / "hc_row_phase_sync.py").write_text(source)
    shutil.copy2(HERE / "sources/spmd/signals.py", target / "signals.py")
    template = (HERE / "hc_row_mixed_template.py").read_text()
    template = infer_inline_tensor_contracts(template, {"hc_row_cube", "hc_row_vector", "hc_row_cluster"})
    (target / "hc_row_mixed.py").write_text(template)
    path = target / "decode_csa.py"
    source = path.read_text()
    tree = ast.parse(source)
    import_node = next(n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "hc_pre")
    lines = source.splitlines(keepends=True)
    lines.insert(import_node.end_lineno, "from .hc_row_mixed import hc_comb_after_prefix, hc_row_prefix\n")
    source = "".join(lines)
    old = section(source, "        x_hc32 = pl.create_tensor(", "        # 27711336前段证明")
    replacement = """        # HC_ROW_DEPENDENCIES：前段按真实行owner推进；comb另起任务。
        post_t, x_normed_t, inv_rms, hc_raw, hc_partials, _widen_tid = hc_row_prefix(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, post_t, x_normed_t,
        )

"""
    source = source.replace(old, replacement, 1)
    old = section(
        source,
        "        with pl.scope():\n            pre_val_store =",
        "        wb_blocks =",
    )
    source = source.replace(
        old,
        "        comb_t = hc_comb_after_prefix(\n"
        "            inv_rms, hc_raw, hc_partials, hc_attn_scale, hc_attn_base, comb_t, _widen_tid,\n"
        "        )\n",
        1,
    )
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="2417166f，冻结13259f92已启用HC_ROW_DEPENDENCIES",
                scope="HC widen/投影/RAW/pre-post/norm局部MIX；Sinkhorn独立；其它CSA保持多task",
                preserved="原四K左折叠、BF16边界、RMS分段、动态实际T及尾块；CONSUMER_REDUCE=False",
                lifetime="单调用4阶段信号显式清零；真实行owner发布/等待，业务cache维护沿来源",
                wait_policy={"backoff": "cached100", "unit_family": "hc"},
                status="source_prepared_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
