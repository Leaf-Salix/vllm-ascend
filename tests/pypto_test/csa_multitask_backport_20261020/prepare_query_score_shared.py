# SPDX-License-Identifier: Apache-2.0
"""保留动态前段分派，共用Indexer；关闭融合时不接回不适用的Q返回别名。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_dispatch_v6")
    parser.add_argument("--candidate", default="query_score_shared_v9")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    indexer = (base / "decode_indexer.py").read_text()
    indexer = replace_once(
        indexer,
        "                q_stage_out = q_dequant_rope_stage(\n",
        "                _query_written = q_dequant_rope_stage(\n",
    )
    entry = (base / "decode_csa.py").read_text()
    tree = ast.parse(entry)
    branch = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "defer_q_to_score == 1"]
    if len(branch) != 1:
        raise ValueError("需要唯一前段分派")
    lines = entry.splitlines(keepends=True)
    for kind, body in reversed((("fused", branch[0].body), ("standalone", branch[0].orelse))):
        first, last = body[0].lineno - 1, body[-1].end_lineno
        text = "".join(lines[first:last])
        text = replace_once(text, "_index_scores, index_indices =", f"_{kind}_scores, _{kind}_indices =")
        lines[first:last] = [text]
    entry = "".join(lines)
    ast.parse(entry)
    ast.parse(indexer)
    shutil.copytree(base, target)
    (target / "decode_csa.py").write_text(entry)
    (target / "decode_indexer.py").write_text(indexer)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="共用Indexer但不把仅融合分支的Q返回别名赋回入参；前段两个分支使用独立未使用返回名",
        validation_required=[
            "关闭融合的Score必须与原实现等价且不捕获七个Q参数",
            "开启融合必须保留真实QB依赖与Q写入，完整图切换必须验证",
            "不把CPU解析通过代替设备编译、功能或性能",
        ],
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
