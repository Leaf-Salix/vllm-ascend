# SPDX-License-Identifier: Apache-2.0
"""迁移c0d27e9a已验证的score/index成对压缩，只替换阈值归并叶实现。"""

import argparse
import ast
import json
import shutil
import statistics
import subprocess

from prepare import HERE, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="hc_post_workers_v1")
    parser.add_argument("--candidate", default="topk_paired_v1")
    args = parser.parse_args()
    if not all(value.isidentifier() for value in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    donor = "c0d27e9a"
    root = "tests/pypto_test/csa_roofline_optimization_20261020"
    source = subprocess.check_output(
        ["git", "show", f"{donor}:{root}/source_topk_paired/topk_threshold_stages.py"],
        cwd=REPO,
        text=True,
    )
    performance = json.loads(
        subprocess.check_output(["git", "show", f"{donor}:{root}/PAIR_PERFORMANCE.json"], cwd=REPO, text=True)
    )
    evidence = []
    for row in performance["rows"]:
        if not row["passed"] or not row["full_csa_validated"] or not row["graph_replay_validated"]:
            raise ValueError("来源缺完整CSA/图验证")
        sides = {}
        for side in ("spmd_reference", "spmd_candidate"):
            samples = row["statistics_us"][side]["samples_us"]
            sides[side] = dict(min_us=min(samples), mean_us=statistics.mean(samples), max_us=max(samples))
        evidence.append(dict(batch=row["batch"], statistics_us=sides, result=row["result"]))
    ast.parse(source)
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    (target / "topk_threshold_stages.py").write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor=f"{donor}:{root}/source_topk_paired/topk_threshold_stages.py",
                changed_files=["topk_threshold_stages.py"],
                changes="GE掩码同时压缩score与INT32索引，减少索引fillpad/物化和随机gather",
                preserved="原调用范围、阈值/同分规则/回退排序、AIC数学、TaskId及原同步/依赖",
                source_evidence=evidence,
                required_pypto="8c360367e0239779a93b495ecf7910e26a9f1fc6",
                required_ptoas="b27d5a0fe9f010d6d50e0287b92b8ac413cfc732",
                status="source_frozen_no_multitask_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
