# SPDX-License-Identifier: Apache-2.0
"""私有诊断候选恢复Native BF16边界；保留原注释/调度，不代表答案质量更好。

正式算子仍使用原FP32融合路径。用户不要求逐bit/逐token对齐，候选仅用于
解释数值差异，是否采用必须由decode答案质量与性能共同决定。
"""
import argparse
import ast
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def round_after(path, names):
    text = path.read_text()
    lines = text.splitlines(True)
    additions = []
    counts = {name: 0 for name in names}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                counts[name] += 1
                indent = " " * node.col_offset
                additions.append((node.end_lineno, indent + "# 恢复Native可见的BF16结果，再以FP32执行后续融合计算。\n" +
                                  indent + f"{name} = pl.cast(pl.cast({name}, pl.BF16, mode=\"rint\"), pl.FP32)\n"))
    assert all(counts.values()), (path, counts)
    for line, value in sorted(additions, reverse=True):
        lines.insert(line, value)
    path.write_text("".join(lines))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--variant", choices=("baseline", "qa_bf16", "all_bf16"), required=True)
    args = parser.parse_args()
    target = HERE / "experiments" / args.name
    target.mkdir(parents=True, exist_ok=False)
    source = HERE / "experiments/v04_graph_service/source/vllm_ascend/ops/pypto"
    operator = target / "operator"
    operator.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, operator / path.name)
    for name in ("deepseek_v4_flash_csa", "deepseek_v4_flash_hca", "deepseek_v4_flash_hca_tnd"):
        shutil.copytree(source / name, operator / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(HERE / "qkv_probe_entry.py", operator / "deepseek_v4_flash_hca_tnd/qkv_probe_entry.py")
    changes = {}
    common = operator / "deepseek_v4_flash_csa/qkv_proj_rope.py"
    if args.variant != "baseline":
        names = {"qr_input_ub"}
        if args.variant == "all_bf16":
            names |= {"kv_chunk", "kv_chunk_tail", "kv_rope_chunk", "kv_rope_chunk_tail",
                      "kv_rope_norm_chunk", "kv_rope_norm_tail"}
        changes["common"] = round_after(common, names)
    if args.variant == "all_bf16":
        changes["q_stream"] = round_after(operator / "deepseek_v4_flash_hca/q_projection_streamed.py",
                                           {"q_head_dq", "q_head_dq_tail", "q_rope_chunk", "q_rope_chunk_tail"})
    (target / "manifest.json").write_text(json.dumps({"variant": args.variant, "changes": changes,
        "source": str(source), "scope": "仅恢复BF16边界；数学源码为私有副本，不改正式算子或SDK"},ensure_ascii=False,indent=2)+'\n')
    print(target)


if __name__ == "__main__":
    main()
