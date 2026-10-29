# SPDX-License-Identifier: Apache-2.0
"""在已有CPU调度回调逐token对Native，首差写现场并终止；不读NPU Tensor。"""

import json
from pathlib import Path

from low_acceptance_20261001.scheduler import AcceptanceScheduler


class CheckedScheduler(AcceptanceScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = args[0] if args else kwargs["vllm_config"]
        self._threeway_output = Path(config.additional_config["offline_acceptance_output"])
        specification = json.loads((self._threeway_output / "experiment.json").read_text())
        self._threeway_reference = None
        self._threeway_events = None
        reference = specification["native_reference"]
        if reference is not None:
            rank = self.parallel_config.data_parallel_rank
            root = Path(reference)
            rows = json.loads((root / f"rank{rank}.mixed.json").read_text())["requests"]
            self._threeway_reference = {row["key"]: row["output_token_ids"] for row in rows}
            records = [json.loads(line) for line in (root / f"request_stats_rank{rank}.jsonl").read_text().splitlines()]
            self._threeway_events = {row["key"]: row["events"] for row in records}

    def _fail(self, kind, key, index, actual, expected):
        payload = {
            "kind": kind,
            "rank": self.parallel_config.data_parallel_rank,
            "key": key,
            "zero_based_index": index,
            "actual": actual,
            "native": expected,
        }
        try:
            with (self._threeway_output / "FIRST_DIFFERENCE.json").open("x") as stream:
                json.dump(payload, stream, indent=2)
        except FileExistsError:
            pass
        (self._threeway_output / "STOP").touch(exist_ok=True)
        raise RuntimeError(f"THREEWAY_FIRST_DIFFERENCE {payload}")

    def _update_request_with_output(self, request, new_token_ids):
        events = list(self.acceptance_events.get(request.request_id, []))
        result = super()._update_request_with_output(request, new_token_ids)
        if self._threeway_reference is not None:
            key = (request.kv_transfer_params or {}).get("offline_key")
            actual = list(request.output_token_ids)
            expected = self._threeway_reference[key]
            for index, token in enumerate(actual):
                if index >= len(expected) or token != expected[index]:
                    self._fail("token", key, index, token, expected[index] if index < len(expected) else None)
            if result[1] and len(actual) != len(expected):
                self._fail("output_length", key, len(actual), len(actual), len(expected))
            if result[1] and events != self._threeway_events[key]:
                self._fail("completed_acceptance_events", key, len(events), events, self._threeway_events[key])
        return result

    def make_spec_decoding_stats(
        self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens, request_id
    ):
        result = super().make_spec_decoding_stats(
            spec_decoding_stats, num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens, request_id
        )
        if self._threeway_events is not None and self.log_stats and num_draft_tokens:
            request = self.requests[request_id]
            key = (request.kv_transfer_params or {}).get("offline_key")
            actual = self.acceptance_events[request_id]
            expected = self._threeway_events[key]
            index = len(actual) - 1
            if index >= len(expected) or actual[index] != expected[index]:
                self._fail(
                    "acceptance_event", key, index, actual[index], expected[index] if index < len(expected) else None
                )
        return result
