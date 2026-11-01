# SPDX-License-Identifier: Apache-2.0
"""nalinaly已知不对齐时的独立性能测量；只记录首差，不改变生成结果。"""

import json

from .scheduler import CheckedScheduler


class NalPerformanceScheduler(CheckedScheduler):
    def _fail(self, kind, key, index, actual, expected):
        if getattr(self, "_nal_performance_recorded", False):
            return
        path = self._threeway_output / f"performance_difference_rank{self.parallel_config.data_parallel_rank}.json"
        payload = dict(kind=kind, key=key, zero_based_index=index, actual=actual, native=expected)
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2)
        self._nal_performance_recorded = True
        print("NAL_PERFORMANCE_DIFFERENCE_RECORDED " + json.dumps(payload), flush=True)
