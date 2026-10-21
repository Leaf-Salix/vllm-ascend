# SPDX-License-Identifier: Apache-2.0
"""让来源的4核Hadamard/20核QB分工按组准入，保留全部核内数学。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="cooperative_query_ready_v1")
    parser.add_argument("--candidate", default="cooperative_query_groups_v1")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)

    path = target / "q_projection.py"
    source = replace_once(
        path.read_text(),
        'with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", deps=[qproj_dep]) as qproj_tid:',
        'with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", deps=[qproj_dep], sync_start=True) as qproj_tid:',
    )
    ast.parse(source)
    path.write_text(source)
    path = target / "decode_indexer.py"
    source = replace_once(
        path.read_text(),
        '        name_hint="qr_hadamard_matmul",\n        deps=[qh_mm_dep],\n        allow_early_resolve=True,\n',
        '        name_hint="qr_hadamard_matmul",\n'
        "        deps=[qh_mm_dep],\n"
        "        allow_early_resolve=True,\n"
        "        sync_start=True,\n",
    )
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            {
                "candidate": args.candidate,
                "base": args.base,
                "donor": "13259f92的4核Hadamard/20核QB互不重叠分工；多task按组准入适配",
                "changes": "仅NZ QB20与Hadamard4两组sync_start=True，原early resolve及数学不变",
                "evidence": "QUERY_READY_QB_DISTRIBUTION.json：QB20在长B16五窗实际只用12至17个AIC，B28四窗只用16个",
                "historical_limit": "第220节旧24核QB单独sync_start曾回退；此处只验证新的4/20协同分工和已观察重复占核",
                "preserved": "ND QB、所有依赖、Q/Score顺序、tile/K遍历、cache/state及页保护",
                "pending": "验证真实20/4核覆盖、首启离散和完整CSA mean/max，不能只凭启动整齐采用",
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
