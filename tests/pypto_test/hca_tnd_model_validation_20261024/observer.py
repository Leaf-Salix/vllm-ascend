# SPDX-License-Identifier: Apache-2.0
"""整模型HCA真实路由观测：区分捕获选择、实际replay和Python重新提交。"""
from collections import Counter

from offline_pd.observer import OfflineCSAObserver, _CSA_SELECTION
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import TNDGraphContract
from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.graph_service import HCATNDGraphServiceRuntime

_original_eligible = HCATNDGraphServiceRuntime.eligible
_original_allows = TNDGraphContract.allows_replay


def counted_eligible(runtime, context, hidden, positions):
    verdict = _original_eligible(runtime, context, hidden, positions)
    counts = _CSA_SELECTION.setdefault(runtime.layer_name, {})
    key = f"{'pto' if verdict else 'native'}_tokens{hidden.shape[0]}"
    counts[key] = counts.get(key, 0) + 1
    return verdict


def counted_replay(contract, context):
    allowed = _original_allows(contract, context)
    runtime = contract.runtime
    if isinstance(runtime, HCATNDGraphServiceRuntime):
        if not hasattr(runtime, "validation_graph_counts"):
            runtime.validation_graph_counts = Counter()
        runtime.validation_graph_counts["allowed" if allowed else "rejected"] += 1
        if not allowed:
            reasons = getattr(runtime, "validation_graph_rejections", [])
            if len(reasons) < 3:
                current = runtime.graph_signature(context, contract.token_capacity)
                reasons.append({"capacity": contract.token_capacity,
                                "captured_signature_present": contract.signature is not None,
                                "current_signature_present": current is not None,
                                "policy_changed": bool(current and contract.signature and current[1] != contract.signature[1])})
            runtime.validation_graph_rejections = reasons
    return allowed


HCATNDGraphServiceRuntime.eligible = counted_eligible
TNDGraphContract.allows_replay = counted_replay


class HCAObserver(OfflineCSAObserver):
    def offline_begin_observation(self):
        from vllm.forward_context import get_forward_context

        result = super().offline_begin_observation()
        self._hca_occupancy = Counter()
        self._hca_routes = Counter()
        self._hca_runtimes = []
        for index, layer in enumerate(self.model_runner.get_model().model.layers):
            runtime = getattr(layer.self_attn.dsa_attn, "_pto_hca_runtime", None)
            if runtime is not None:
                self._hca_runtimes.append((index, runtime))
        self._hca_origin = self.model_runner._model_forward
        prefix = self._offline_csa_layer_names[0]
        self._hca_before = {index: (runtime.submissions, dict(getattr(runtime, "validation_graph_counts", {})))
                            for index, runtime in self._hca_runtimes}

        def observed(num_tokens_padded, *args, **kwargs):
            context = get_forward_context()
            route = None
            if not context.capturing and not getattr(context, "is_draft_model", False):
                metadata = (context.attn_metadata or {}).get(prefix)
                if metadata is not None and not metadata.num_prefills and metadata.decode is not None:
                    requests = metadata.decode.num_reqs_actual
                    if requests is not None:
                        route = (requests, metadata.num_actual_tokens, num_tokens_padded,
                                 metadata.decode.max_seq_lens, context.cudagraph_runtime_mode.name)
                        self._hca_occupancy[route[:3]] += 1
            before = [(runtime.submissions, getattr(runtime, "validation_graph_counts", {}).get("allowed", 0))
                      for _, runtime in self._hca_runtimes]
            value = self._hca_origin(num_tokens_padded, *args, **kwargs)
            if route is not None and self._hca_runtimes:
                replayed = all(getattr(runtime, "validation_graph_counts", {}).get("allowed", 0) > counts[1]
                               for (_, runtime), counts in zip(self._hca_runtimes, before))
                submitted = any(runtime.submissions > counts[0] for (_, runtime), counts in zip(self._hca_runtimes, before))
                self._hca_routes[(*route, replayed, submitted)] += 1
            return value

        self.model_runner._model_forward = observed
        return result

    def offline_end_observation(self):
        import torch

        self.model_runner._model_forward = self._hca_origin
        result = super().offline_end_observation()
        result["decode_occupancy"] = [dict(actual_requests=b, actual_tokens=t, padded_tokens=c, steps=n)
                                      for (b, t, c), n in sorted(self._hca_occupancy.items())]
        result["hca_execution_routes"] = [
            dict(actual_requests=b, actual_tokens=t, padded_tokens=c, max_seq_len=length,
                 requested_mode=mode, all_hca_graph_replayed=replay, python_resubmitted=submitted, steps=n)
            for (b, t, c, length, mode, replay, submitted), n in sorted(self._hca_routes.items())]
        layers = []
        for index, runtime in self._hca_runtimes:
            before, graph_before = self._hca_before[index]
            graph = getattr(runtime, "validation_graph_counts", {})
            layers.append(dict(layer=index, runtime_class=type(runtime).__module__,
                               decode_submissions=runtime.submissions - before,
                               graph_counts={k: graph.get(k, 0)-graph_before.get(k, 0) for k in ("allowed", "rejected")},
                               empty_calls=runtime.empty_calls,
                               graph_rejections=getattr(runtime, "validation_graph_rejections", [])))
        result["hca_layers"] = layers
        result["memory"] = dict(allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved(),
                                peak_allocated=torch.npu.max_memory_allocated(), free_total=list(torch.npu.mem_get_info()))
        return result
