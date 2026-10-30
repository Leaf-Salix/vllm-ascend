# SPDX-License-Identifier: Apache-2.0
"""共同轻量观测：真实FULL240目标forward及周期，热路径不读device Tensor。"""

import math
import time
from collections import Counter

import torch
from csa_threeway_20261029.runtime import configure_runtime
from csa_tnd_model_validation_20261023.observer import TNDObserver
from offline_pd.forward_timing import prewarm_timing_events
from vllm.forward_context import get_forward_context

WARMUP_STEPS = 8
TIMING_STEPS = 20

configure_runtime()


class TimedObserver(TNDObserver):
    def offline_begin_observation(self):
        result = super().offline_begin_observation()
        self._threeway_events = prewarm_timing_events(TIMING_STEPS)
        self._threeway_samples = []
        self._threeway_seen = 0
        self._threeway_descriptors = Counter()
        self._threeway_original = self.model_runner._model_forward
        prefix = self._offline_csa_layer_names[0]

        def forward(num_tokens_padded, *args, **kwargs):
            context = get_forward_context()
            metadata = (context.attn_metadata or {}).get(prefix)
            if (
                not context.capturing
                and not getattr(context, "is_draft_model", False)
                and metadata is not None
                and not metadata.num_prefills
                and metadata.decode is not None
            ):
                request = metadata.decode
                lengths = getattr(request, "seq_lens", None)
                table = getattr(request, "block_table", None)
                self._threeway_descriptors[
                    (
                        num_tokens_padded,
                        lengths.numel() if lengths is not None else request.num_reqs_actual,
                        request.num_reqs_actual,
                        metadata.num_actual_tokens,
                        getattr(request, "max_seq_lens", None),
                        table.shape[1] if table is not None else None,
                        context.cudagraph_runtime_mode.name,
                    )
                ] += 1
            selected = (
                not context.capturing
                and not getattr(context, "is_draft_model", False)
                and metadata is not None
                and not metadata.num_prefills
                and metadata.decode is not None
                and metadata.decode.num_reqs_actual == 40
                and num_tokens_padded == 240
                and context.cudagraph_runtime_mode.name == "FULL"
            )
            if not selected:
                return self._threeway_original(num_tokens_padded, *args, **kwargs)
            ordinal = self._threeway_seen
            self._threeway_seen += 1
            if ordinal < WARMUP_STEPS or len(self._threeway_samples) >= TIMING_STEPS:
                return self._threeway_original(num_tokens_padded, *args, **kwargs)
            begin, end = self._threeway_events[len(self._threeway_samples)]
            row = {
                "ordinal": ordinal,
                "actual_requests": metadata.decode.num_reqs_actual,
                "actual_tokens": metadata.num_actual_tokens,
                "padded_tokens": num_tokens_padded,
                "positions": self.model_runner._dsa_positions_cpu_buf[: metadata.num_actual_tokens].tolist(),
                "host_begin": time.perf_counter(),
                "begin": begin,
                "end": end,
            }
            self._threeway_samples.append(row)
            begin.record()
            try:
                return self._threeway_original(num_tokens_padded, *args, **kwargs)
            finally:
                end.record()

        self.model_runner._model_forward = forward
        return result

    def offline_end_observation(self):
        self.model_runner._model_forward = self._threeway_original
        # All requests have completed before this RPC. No step-level waits.
        torch.npu.synchronize()
        samples = []
        for index, row in enumerate(self._threeway_samples):
            samples.append({key: value for key, value in row.items() if key not in ("begin", "end")})
            samples[-1]["forward_us"] = row["begin"].elapsed_time(row["end"]) * 1000
            if index + 1 < len(self._threeway_samples):
                next_begin = self._threeway_samples[index + 1]["begin"]
                samples[-1]["cycle_us"] = row["begin"].elapsed_time(next_begin) * 1000
        result = super().offline_end_observation()
        result["policy_descriptors"] = [
            dict(
                tokens=t,
                requests=b,
                actual_requests=actual_b,
                actual_tokens=actual_t,
                max_seq_len=length,
                compressed_table_columns=columns,
                requested_mode=mode,
                steps=steps,
            )
            for (t, b, actual_b, actual_t, length, columns, mode), steps in self._threeway_descriptors.items()
        ]
        result["threeway_timing"] = {
            "scope": (
                "physical FULL240 bucket, actual T recorded; "
                "cycle includes draft/sampling/metadata and CPU reference checks"
            ),
            "warmup_steps": WARMUP_STEPS,
            "requested_steps": TIMING_STEPS,
            "matched_steps": self._threeway_seen,
            "samples": samples,
            "complete": len(samples) == TIMING_STEPS
            and all(row["ordinal"] == WARMUP_STEPS + i for i, row in enumerate(samples))
            and all(math.isfinite(row["forward_us"]) and row["forward_us"] > 0 for row in samples)
            and all(math.isfinite(row["cycle_us"]) and row["cycle_us"] > 0 for row in samples[:-1]),
        }
        return result
