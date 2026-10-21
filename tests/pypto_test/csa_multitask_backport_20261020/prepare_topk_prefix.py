# SPDX-License-Identifier: Apache-2.0
"""从已提交并完成七档的单SPMD v5迁入精确Top-K短前缀，保持原多task调度。"""

import argparse
import ast
import json
import shutil
import subprocess
import textwrap

from prepare import HERE, REPO
from prepare_query_ready import replace_once

DONOR_COMMIT = "c49ee5b2"
DONOR_DIRECTORY = "tests/pypto_test/csa_roofline_optimization_20261020/source_topk_payload"


def committed_source(name):
    return subprocess.check_output(["git", "show", f"{DONOR_COMMIT}:{DONOR_DIRECTORY}/{name}"], cwd=REPO, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="plan_workers_v1")
    parser.add_argument("--candidate", default="topk_prefix_v1")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    helper = committed_source("topk_threshold_stages.py")
    ast.parse(helper)
    (target / "topk_threshold_stages.py").write_text(helper)

    # 原有balanced路径已经保留同一份single_roots。直接在原排序/归并位置
    # 插入来源的纯Vector helper，不能把UB root带到另一个task或改变ACK。
    path = target / "decode_indexer.py"
    source = path.read_text()
    source = replace_once(
        source,
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .topk_threshold_stages import threshold_merge_top512\n",
    )
    begin_marker = "                                        logical_i32 = pl.cast(single_segment_begin, pl.INT32)\n"
    end_marker = (
        "                        for single_local in pl.unroll(query_group_size // 2):\n"
        "                            single_query_lane = buf_score_lane * (query_group_size // 2) + single_local\n"
        "                            single_root = pl.tile.extract(\n"
    )
    if source.count(begin_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError("需要唯一的balanced分段排序与最终根发布位置")
    begin = source.index(begin_marker)
    end = source.index(end_marker, begin)
    original_sort_merge = source[begin:end]

    donor = committed_source("indexer_score_stages.py")
    donor_begin = donor.index(
        "                                    if single_step >= 2 and single_segment_valid > 512:\n"
    )
    donor_end = donor.index("                                    else:\n", donor_begin)
    prefix = textwrap.indent(textwrap.dedent(donor[donor_begin:donor_end]), " " * 40)
    replacement = prefix + " " * 40 + "else:\n" + textwrap.indent(original_sort_merge, " " * 4)
    source = source[:begin] + replacement + source[end:]
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            {
                "candidate": args.candidate,
                "base": args.base,
                "donor_commit": DONOR_COMMIT,
                "donor_files": [
                    f"{DONOR_DIRECTORY}/topk_threshold_stages.py",
                    f"{DONOR_DIRECTORY}/indexer_score_stages.py",
                ],
                "changes": "仅balanced长历史的已有UB根后续分段采用精确GE筛选及512/1024短前缀",
                "helper": "从提交直接读取，保留完整helper源码与原回退，不读另一session未提交候选",
                "preserved": "Score算术、AIC代码、READY/ACK、TaskId依赖、同分incoming优先、原动态范围",
                "source_evidence": "../csa_roofline_optimization_20261020/PAYLOAD_PERFORMANCE.json",
                "source_limit": "来源七档功能通过但未整体采用；本轮只作独立迁移验证，不预设各档收益",
                "toolchain": "已验证TopK隔离PyPTO 654779734、PTOAS b48c84eb及Simpler 57e499238；四侧统一",
                "status": "source_prepared_no_device_evidence",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
