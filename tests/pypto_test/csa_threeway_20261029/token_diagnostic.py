# SPDX-License-Identifier: Apache-2.0
"""接受事件首差只记录；逐token及长度仍由原Checker严格检查、首差停止。

这是独立诊断，不能改变原严格postflight的失败结果，也不构成性能同口径验收。
"""

import json

from .scheduler import CheckedScheduler


class TokenDiagnosticScheduler(CheckedScheduler):
    def _fail(self, kind, key, index, actual, expected):
        if kind not in ("acceptance_event", "completed_acceptance_events"):
            return super()._fail(kind, key, index, actual, expected)
        payload = {
            "kind": kind,
            "rank": self.parallel_config.data_parallel_rank,
            "key": key,
            "zero_based_index": index,
            "actual": actual,
            "native": expected,
            "scope": "event difference recorded; original token/length checks remain fatal",
        }
        path = self._threeway_output / f"first_acceptance_difference_rank{payload['rank']}.json"
        try:
            with path.open("x") as stream:
                json.dump(payload, stream, indent=2)
        except FileExistsError:
            return
        print("ACCEPTANCE_DIFFERENCE_TOKENS_STILL_CHECKED " + json.dumps(payload), flush=True)
