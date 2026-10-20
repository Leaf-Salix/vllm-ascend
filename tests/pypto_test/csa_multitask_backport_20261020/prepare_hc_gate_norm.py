# SPDX-License-Identifier: Apache-2.0
"""复用来源同AIV的gate→mix/norm组织，融合当前数学，保持HC其余task。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_fused_topk import add_parameter
from prepare_scale_early import add_calls


def text(source, nodes):
    lines = source.splitlines(keepends=True)
    return "".join(lines[nodes[0].lineno - 1 : nodes[-1].end_lineno])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="hc_gate_norm_v1")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    path = target / "hc_pre.py"
    source = path.read_text()
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    gate = functions["hc_pre_gates_from_rms"]
    mix = functions["hc_mix_norm"]
    gate_loop = next(n for n in gate.body if isinstance(n, ast.For) and "split_pre_post" in ast.unparse(n.iter))
    mix_loop = next(n for n in mix.body if isinstance(n, ast.With))
    if [ast.unparse(n.targets[0]) for n in mix_loop.body[:2]] != ["t0", "valid_rows"]:
        raise ValueError("mix的行映射结构改变，须人工核对")
    mix_math = textwrap.dedent(text(source, mix_loop.body[2:]))
    allocations = [
        n
        for n in mix.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("t_pad", "x_mixed", "x_normed_tail_store")
    ]
    if len(allocations) != 3:
        raise ValueError("mix的三项工作区配置不符")
    prelude = "".join(text(source, [n]) for n in allocations)
    begin = gate.decorator_list[0].lineno
    original = text(source, [gate])
    lines = original.splitlines(keepends=True)
    # 后插入的数学保持在gate的每个M8行块内，避免先做完所有gate再读回。
    lines[gate_loop.end_lineno - gate.lineno + 1 : gate_loop.end_lineno - gate.lineno + 1] = [
        textwrap.indent(mix_math, "            ")
    ]
    lines[gate_loop.lineno - gate.lineno : gate_loop.lineno - gate.lineno] = [prelude]
    combined = "".join(source.splitlines(keepends=True)[begin - 1 : gate.lineno - 1]) + "".join(lines)
    combined = combined.replace("def hc_pre_gates_from_rms(", "def hc_pre_gates_and_norm(", 1)
    combined = add_parameter(
        combined,
        "hc_pre_gates_and_norm",
        "    norm_w: pl.Tensor[[D], pl.BF16],\n    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],\n",
    )
    combined = combined.replace("PRE_POST_WORKERS", "HC_GATE_NORM_WORKERS")
    combined = combined.replace('name_hint="split_pre_post"', 'name_hint="hc_gate_mix_norm"', 1)
    source += "\n\n# 来源hc_gates_range和hc_mix_norm_range共用48路M8行owner。\nHC_GATE_NORM_WORKERS = 48\n\n" + combined
    ast.parse(source)
    path.write_text(source)
    path = target / "decode_csa.py"
    source = path.read_text().replace("    hc_pre_gates_from_rms,\n", "    hc_pre_gates_and_norm,\n", 1)
    source = source.replace("            hc_pre_gates_from_rms(\n", "            hc_pre_gates_and_norm(\n", 1)
    source = add_calls(source, ("hc_pre_gates_and_norm",), ("attn_norm_w", "x_normed_t"))
    old = "            hc_mix_norm(x_hc32, pre_val_store, attn_norm_w, x_normed_t)\n"
    if source.count(old) != 1:
        raise ValueError("原norm调用应唯一")
    source = source.replace(old, "", 1)
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="冻结13259f92的HC_ROW_DEPENDENCIES：gate与mix/norm归同一AIV",
                scope="仅融合每个M8的gate→mix/norm；HC widen、Cube投影、RAW归约、comb仍独立task",
                preserved="直接提取当前数学；四项混合树、BF16舍入边界、RMS分段及partial row保护保持",
                worker_policy="沿来源48路行owner，独立任务只派发min(ceil(T/8),48)个worker",
                lifetime="保留原x_mixed及唯一尾块暂存；没有GM信号或软等待",
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
