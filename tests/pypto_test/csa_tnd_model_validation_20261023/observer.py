# SPDX-License-Identifier: Apache-2.0
"""生成结束后记录TND绑定/工作区；不向forward增加同步或tensor读取。"""

from collections import Counter

from offline_pd.observer import OfflineCSAObserver

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import TNDGraphContract


def install_graph_observer():
    original = TNDGraphContract.allows_replay

    def counted(contract, context):
        allowed = original(contract, context)
        runtime = contract.runtime
        if not hasattr(runtime, "_validation_graph_counts"):
            runtime._validation_graph_counts = {"allowed": 0, "rejected": 0}
        runtime._validation_graph_counts["allowed" if allowed else "rejected"] += 1
        return allowed

    TNDGraphContract.allows_replay = counted


install_graph_observer()


class TNDObserver(OfflineCSAObserver):
    def offline_begin_observation(self):
        from vllm.forward_context import get_forward_context

        result = super().offline_begin_observation()
        self._tnd_decode_occupancy = Counter()
        self._tnd_occupancy_origin = self.model_runner._model_forward
        prefix = self._offline_csa_layer_names[0]

        def count_occupancy(num_tokens_padded, *args, **kwargs):
            context = get_forward_context()
            if not context.capturing and not getattr(context, "is_draft_model", False):
                metadata = (context.attn_metadata or {}).get(prefix)
                if metadata is not None and not metadata.num_prefills and metadata.decode is not None:
                    # 只用Native已有Host计数，不能把图padding容量当成真实batch。
                    requests = metadata.decode.num_reqs_actual
                    if requests is not None:
                        self._tnd_decode_occupancy[(requests, metadata.num_actual_tokens, num_tokens_padded)] += 1
            return self._tnd_occupancy_origin(num_tokens_padded, *args, **kwargs)

        self.model_runner._model_forward = count_occupancy
        self._tnd_before = {}
        for index, layer in enumerate(self.model_runner.get_model().model.layers):
            runtime = getattr(layer.self_attn.dsa_attn, "_pto_csa_runtime", None)
            if runtime is not None:
                self._tnd_before[index] = (runtime.submissions, dict(getattr(runtime, "_validation_graph_counts", {})))
        return result

    def offline_end_observation(self):
        import torch

        self.model_runner._model_forward = self._tnd_occupancy_origin
        result = super().offline_end_observation()
        result["decode_occupancy"] = [
            {"actual_requests": b, "actual_tokens": t, "padded_tokens": capacity, "steps": steps}
            for (b, t, capacity), steps in sorted(self._tnd_decode_occupancy.items())
        ]
        rows = []
        unique_storage = {}
        for index, layer in enumerate(self.model_runner.get_model().model.layers):
            runtime = getattr(layer.self_attn.dsa_attn, "_pto_csa_runtime", None)
            if runtime is None:
                continue
            before_submits, before_graph = self._tnd_before[index]
            graph = dict(getattr(runtime, "_validation_graph_counts", {}))
            for binding in runtime._bindings.values():
                for value in binding.allocations.values():
                    storage = value.untyped_storage()
                    unique_storage[(value.device, storage.data_ptr())] = storage.nbytes()
            rows.append(
                {
                    "layer": index,
                    "runtime_class": type(runtime).__module__,
                    "submissions": runtime.submissions,
                    "decode_submissions": runtime.submissions - before_submits,
                    "decode_graph_checks": {
                        k: graph.get(k, 0) - before_graph.get(k, 0) for k in ("allowed", "rejected")
                    },
                    "empty_calls": runtime.empty_calls,
                    "bindings": [
                        {
                            "policy": binding.policy.constexprs(),
                            "workspace_bytes": sum(t.numel() * t.element_size() for t in binding.allocations.values()),
                        }
                        for binding in runtime._bindings.values()
                    ],
                }
            )
        result["tnd_bindings"] = rows
        result["tnd_physical_workspace_bytes"] = sum(unique_storage.values())
        result["memory"] = {
            "allocated": torch.npu.memory_allocated(),
            "reserved": torch.npu.memory_reserved(),
            "peak_allocated": torch.npu.max_memory_allocated(),
            "free_total": list(torch.npu.mem_get_info()),
        }
        return result
