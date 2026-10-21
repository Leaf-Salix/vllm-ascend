# SPDX-License-Identifier: Apache-2.0
"""复用完整三方五窗，在全部计时结束后核对真实device映射；不改原检查。"""

import importlib
import json
from pathlib import Path
import runpy
import sys

from score_owner_model import base, expected, pad16


def main():
    here = Path(__file__).resolve().parent
    loader = here.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/mgather_physical_plan_retest_20261019"
    sys.path.insert(0, str(loader))
    matrix = importlib.import_module("output_policy_matrix")
    original = matrix.run_matrix

    def audited(**kwargs):
        result = original(**kwargs)
        existing = kwargs["existing"]
        assert existing.policy.score_prefix_balance
        requests = existing.buffers["idx_lengths"].shape[0]
        tokens = existing.buffers["x"].shape[0]
        plan = existing.buffers["group_plan"].cpu().tolist()
        workspace = existing.buffers["group_offsets"].cpu().tolist()
        visible = [workspace[pad16(requests + 1) + 4 * g + 3] for g in range(plan[0])]
        begin = base(requests, tokens)
        got = workspace[begin:begin + 96]
        want = expected(plan, visible, requests, tokens)
        forward, inverse = got[48:72], got[72:96]
        report = dict(passed=got == want, outside_timing_windows=True, plan=plan, visible=visible,
                      score_blocks=got[:24], prefix_rank=got[24:48],
                      physical_to_logical=forward, logical_to_physical=inverse,
                      score_blocks_after=[got[i] for i in forward] if sorted(forward) == list(range(24)) else None,
                      changed_physical_cores=sum(i != v for i, v in enumerate(forward)),
                      expected_words=want)
        output = Path(kwargs["output"])
        (output / "SCORE_OWNER_DEVICE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if not report["passed"]:
            raise AssertionError("真实Score核归属/逆映射与本轮device计划不一致")
        return result

    matrix.run_matrix = audited
    sys.argv[0] = str(here / "run_address_audited_full.py")
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        matrix.run_matrix = original


if __name__ == "__main__":
    main()
