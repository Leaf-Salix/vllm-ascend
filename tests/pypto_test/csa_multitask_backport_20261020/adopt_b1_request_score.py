# SPDX-License-Identifier: Apache-2.0
"""在B1完整CSA及同图边界通过后，仅迁入G6/叶计划/归并条件的联动。"""

import ast
import json
import subprocess

from adopt_core_v2 import RUFF, TARGET, equivalent
from prepare import HERE
from prepare_b1_request_score import migrate


def operational_source(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.unparse(tree)


def main():
    measurement = HERE / "runs/b1_request_score_v2/h131072_b1_request_v1/result.json"
    boundary = HERE / "runs/b1_request_score_boundary_v1/h131072_b1_boundary_v1/result.json"
    result, replay = (json.loads(path.read_text()) for path in (measurement, boundary))
    if not result.get("passed") or not replay.get("passed") or not replay["boundary_replay"]["passed"]:
        raise ValueError("B1完整或同图边界尚未通过")
    before = result["l1"]["statistics_us"]["multitask_current"]
    after = result["l1"]["statistics_us"]["multitask_candidate"]
    if result["current"] != "hc_post_workers_v1" or any(
        after[key] >= before[key] for key in ("min_us", "mean_us", "max_us")
    ):
        raise ValueError("B1采用参照或完整收益不符合预期")
    files = ("decode_indexer.py", "decode_csa.py")
    formal = [(TARGET / name).read_text() for name in files]
    for name, source in zip(files, formal):
        base = (HERE / "sources/hc_post_workers_v1" / name).read_text()
        if not equivalent(operational_source(source), operational_source(base)):
            raise ValueError(f"正式{name}已偏离验证基底，拒绝覆盖")
    updated = list(migrate(*formal))
    updated[0] = (
        updated[0]
        .replace(
            "Balance four/eight/sixteen request groups over the 24 Score workers.",
            "Balance the supported request groups over the 24 Score workers.",
        )
        .replace(
            "pair arena. Other request counts preserve the original leaf partition.",
            "pair arena. B1 targets 24 leaves, capped by the actual tile count;\n"
            "    other request counts preserve the original leaf partition.",
        )
    )
    stage = HERE / "sources/adoption_b1_request_score_v2"
    stage.mkdir(exist_ok=False)
    for name, source in zip(files, updated):
        tested = (HERE / "sources/b1_request_score_v2" / name).read_text()
        if not equivalent(operational_source(source), operational_source(tested)):
            raise ValueError(f"正式候选{name}与实测算法不一致")
        (stage / name).write_text(source)
    subprocess.run([str(RUFF), "check", *(str(stage / name) for name in files)], check=True)
    for name in files:
        (TARGET / name).write_text((stage / name).read_text())
    report = dict(
        candidate="b1_request_score_v2",
        base="b067af15 / hc_post_workers_v1",
        files=list(files),
        policy="长历史T=S=6选择G6；按24个Score worker分叶并联动layout证明/单根归并",
        source="用户明确授权B1分支；既有长G6核内实现及leaf计划的资源适配",
        algorithm_ast_and_imports_equal=True,
        full_csa_l1=result["l1"]["statistics_us"],
        measurement=str(measurement.relative_to(HERE)),
        boundary=str(boundary.relative_to(HERE)),
        boundary_states=[step["state"] for step in replay["boundary_replay"]["steps"]],
        same_graph_replays=replay["boundary_replay"]["same_graph_replays"],
        numeric_validation="本次B1输出/TopK/score/cache/state严格一致；131071倒序页及2047短历史同图通过",
        limits="仍S6，未验证零活动请求、混合query或整模型token/DSpark；其它档位未为少量公共Scalar判断重测",
        required_toolchain=result["toolchain_identity"],
        shared_install_changed=False,
        goal_complete=False,
        pushed=False,
    )
    (HERE / "ADOPTION_b1_request_score.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("B1正式采用完成，八档全胜目标仍未完成。")


if __name__ == "__main__":
    main()
