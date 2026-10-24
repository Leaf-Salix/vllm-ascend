# SPDX-License-Identifier: Apache-2.0
"""生成结束后记录TND绑定/工作区；不向forward增加同步或tensor读取。"""

import json
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
        if not allowed:
            reasons = getattr(runtime, "_validation_graph_rejections", [])
            if len(reasons) < 3:
                current = runtime.graph_signature(context, contract.token_capacity)
                reason = {
                    "token_capacity": contract.token_capacity,
                    "captured_signature_present": contract.signature is not None,
                    "current_signature_present": current is not None,
                }
                if current is not None and contract.signature is not None:
                    old, new = dict(contract.signature[0]), dict(current[0])
                    reason["changed_descriptors"] = [
                        key for key in old.keys() | new.keys() if old.get(key) != new.get(key)
                    ]
                    reason["policy_changed"] = contract.signature[1] != current[1]
                reasons.append(reason)
                runtime._validation_graph_rejections = reasons
                print("TND_GRAPH_REJECT " + json.dumps(reason, ensure_ascii=False), flush=True)
        return allowed

    TNDGraphContract.allows_replay = counted


install_graph_observer()


class TNDObserver(OfflineCSAObserver):
    def offline_begin_observation(self):
        from vllm.forward_context import get_forward_context

        result = super().offline_begin_observation()
        self._tnd_decode_occupancy = Counter()
        self._tnd_execution_routes = Counter()
        self._tnd_runtimes = [
            layer.self_attn.dsa_attn._pto_csa_runtime
            for layer in self.model_runner.get_model().model.layers
            if getattr(layer.self_attn.dsa_attn, "_pto_csa_runtime", None) is not None
        ]
        self._tnd_occupancy_origin = self.model_runner._model_forward
        prefix = self._offline_csa_layer_names[0]

        def count_occupancy(num_tokens_padded, *args, **kwargs):
            context = get_forward_context()
            route = None
            if not context.capturing and not getattr(context, "is_draft_model", False):
                metadata = (context.attn_metadata or {}).get(prefix)
                if metadata is not None and not metadata.num_prefills and metadata.decode is not None:
                    # 只用Native已有Host计数，不能把图padding容量当成真实batch。
                    requests = metadata.decode.num_reqs_actual
                    if requests is not None:
                        self._tnd_decode_occupancy[(requests, metadata.num_actual_tokens, num_tokens_padded)] += 1
                        route = (
                            requests,
                            metadata.num_actual_tokens,
                            num_tokens_padded,
                            metadata.decode.max_seq_lens,
                            context.cudagraph_runtime_mode.name,
                        )
            before = [
                (runtime.submissions, getattr(runtime, "_validation_graph_counts", {}).get("allowed", 0))
                for runtime in self._tnd_runtimes
            ]
            value = self._tnd_occupancy_origin(num_tokens_padded, *args, **kwargs)
            if route is not None and self._tnd_runtimes:
                replayed = all(
                    getattr(runtime, "_validation_graph_counts", {}).get("allowed", 0) > counters[1]
                    for runtime, counters in zip(self._tnd_runtimes, before)
                )
                resubmitted = any(
                    runtime.submissions > counters[0] for runtime, counters in zip(self._tnd_runtimes, before)
                )
                self._tnd_execution_routes[(*route, replayed, resubmitted)] += 1
            return value

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
        result["tnd_execution_routes"] = [
            dict(
                actual_requests=b,
                actual_tokens=t,
                padded_tokens=capacity,
                max_seq_len=length,
                requested_mode=mode,
                all_csa_graph_replayed=replayed,
                python_resubmitted=resubmitted,
                steps=steps,
            )
            for (b, t, capacity, length, mode, replayed, resubmitted), steps in sorted(
                self._tnd_execution_routes.items()
            )
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
                    "graph_rejection_examples": getattr(runtime, "_validation_graph_rejections", []),
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
