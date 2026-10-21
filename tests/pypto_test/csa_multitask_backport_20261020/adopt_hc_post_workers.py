# SPDX-License-Identifier: Apache-2.0
"""将已通过两档完整CSA对照的小token输出owner策略迁入正式目录。"""

import ast
import json

from prepare import HERE, REPO


def main():
    base = HERE / "sources/topk_prefix_v1/decode_o_proj.py"
    candidate = HERE / "sources/hc_post_workers_v1/decode_o_proj.py"
    target = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_o_proj.py"
    if ast.dump(ast.parse(base.read_text())) != ast.dump(ast.parse(target.read_text())):
        raise ValueError("正式O投影已被其它改动推进，需重新确认")
    rows = []
    for batch in (4, 8):
        path = HERE / f"runs/hc_post_workers_v1/h131072_b{batch}_workers_v1/result.json"
        result = json.loads(path.read_text())
        if not result.get("passed") or result["current"] != "topk_prefix_v1":
            raise ValueError("同轮对照未完整通过")
        timing = result["l1"]["statistics_us"]
        before, after = timing["multitask_current"], timing["multitask_candidate"]
        if any(after[key] >= before[key] for key in ("min_us", "mean_us", "max_us")):
            raise ValueError("两代表档的完整CSA收益不足以采用")
        if batch == 4 and not result["boundary_replay"]["passed"]:
            raise ValueError("B4边界回放未通过")
        rows.append(dict(batch=batch, result=str(path.relative_to(HERE)), full_csa_l1_us=timing))
    source = candidate.read_text().replace(
        "PROJ_B_ACT_WORKERS = 48  # A3 AIV capacity; use only the measured T=96 strategy.",
        "PROJ_B_ACT_WORKERS = 48  # A3 AIV capacity; T<=48 and T96 use synchronized token owners.",
    )
    if ast.dump(ast.parse(source)) != ast.dump(ast.parse(candidate.read_text())):
        raise ValueError("正式内容与实测候选算法不一致")
    target.write_text(source)
    report = dict(
        candidate="hc_post_workers_v1",
        baseline="topk_prefix_v1 / 6ba55b45",
        donor="13259f92/output_projection_b_stages.py:o_dequant_hc_post_stage",
        native_reference="cann/cann-recipes-infer@6d060311ae75f6105fe4216e20566875da5a8838",
        file=str(target.relative_to(REPO)),
        policy="T<=48按一token一AIV且sync_start；原T96与T>48其余策略不变",
        algorithm_ast_equal=True,
        numeric_change=False,
        validation="两档S6完整输出/索引/score/cache/state/保护区及图检查；B4六次边界同图回放",
        limitations="非整模型token/DSpark验收；未为未改分支重测其余五档；S6入口限制未变",
        measurements=rows,
        goal_complete=False,
        pushed=False,
    )
    (HERE / "ADOPTION_hc_post_workers.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
