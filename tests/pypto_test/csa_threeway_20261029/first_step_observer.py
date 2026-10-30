# SPDX-License-Identifier: Apache-2.0
"""真实首个普通decode逐层同输入诊断；不用于FULL replay或性能验收。

仅rank0额外运行Native半层，始终恢复实际PTO的cache、共享TopK和输出。
测试runner在首差或首步完成后立即结束，不继续生成192token。
"""

import json
import os
import threading
from pathlib import Path

import torch
from vllm.forward_context import get_forward_context

from .runtime_audit import RuntimeAuditObserver

CACHE_NAMES = ("cache_rows", "main_cache", "main_state", "idx_state", "idx_cache")
MAX_SNAPSHOT_BYTES = 512 << 20


def cpu(value):
    return value.detach().cpu().clone()


def bit_metrics(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    changed = a.ne(b)
    return {
        "bit_equal": not bool(changed.any()),
        "different_bytes": int(changed.sum()),
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
    }


def snapshot(values):
    """CPU保留完整物理carrier；完全重合的view只存一份，部分重合仍各自保留。"""
    aliases, result = {}, {}
    for name, value in values.items():
        key = (value.data_ptr(), tuple(value.shape), value.stride(), value.dtype)
        if key not in aliases:
            aliases[key] = cpu(value)
        result[name] = aliases[key]
    return result


def check_complete(report, expected_layers):
    rows = report["layers"]
    return (
        report["first_forward"] is not None
        and report["first_forward"]["graph_mode"] == "NONE"
        and all(r["forward_id"] == report["first_forward"]["forward_id"] for r in rows)
        and all(r["actual_context"]["graph_mode"] == "NONE" and not r["actual_context"]["capturing"] for r in rows)
        and [r["layer"] for r in rows] == expected_layers
        and all(
            r["restored_exact"]
            and r["input_unchanged"]
            and r["output_unchanged"]
            and r["metadata_restored_exact"]
            and not r["changed_readonly_inputs"]
            for r in rows
        )
        and all(
            r["output"]["bit_equal"]
            and set(r["caches"]) == set(CACHE_NAMES)
            and all(m["bit_equal"] for m in r["caches"].values())
            and (r["shared_topk"] is None or r["shared_topk"]["bit_equal"])
            for r in rows
        )
    )


class FirstStepObserver(RuntimeAuditObserver):
    def _first_step_guard(self):
        with self._first_step_lock:
            if self._first_step_claimed or self._first_step_terminal:
                raise RuntimeError("CSA_SAME_INPUT_DIAGNOSTIC_ALREADY_CLAIMED")

    def _first_step_claim(self, descriptor):
        with self._first_step_lock:
            if self._first_step_claimed or self._first_step_terminal:
                raise RuntimeError("CSA_SAME_INPUT_DIAGNOSTIC_ALREADY_CLAIMED")
            self._first_step_claimed = True
            self._first_step_forward_id = 1
            self._first_step_report["first_forward"] = {**descriptor, "forward_id": self._first_step_forward_id}
            self._first_step_save()

    def _first_step_save(self):
        path = self._first_step_root / "same_input_first_step.json"
        path.write_text(json.dumps(self._first_step_report, ensure_ascii=False, indent=2) + "\n")

    def _first_step_stop(self, reason, **details):
        with self._first_step_lock:
            if self._first_step_terminal:
                raise RuntimeError("CSA_SAME_INPUT_DIAGNOSTIC_ALREADY_STOPPED")
            self._first_step_terminal = True
            self._first_step_report.update(status=reason, stop_details=details)
            self._first_step_save()
            name = "FIRST_STEP_DIAGNOSTIC.json" if reason == "first_step_complete" else "FIRST_DIFFERENCE.json"
            payload = {"kind": reason, "rank": 0, "forward_id": self._first_step_forward_id, **details}
            try:
                with (self._first_step_root / name).open("x") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)
            except FileExistsError:
                pass
            (self._first_step_root / "STOP").touch()
        raise RuntimeError(f"CSA_SAME_INPUT_DIAGNOSTIC_STOP {reason} {details}")

    def _first_step_pair(self, runtime, context, hidden, positions, output, kv_cache, original):
        from vllm_ascend.ops.dsv4_csa import _native_attention_half

        index = self._first_step_indices[id(runtime)]
        actual_context = {
            "graph_mode": context.cudagraph_runtime_mode.name,
            "capturing": bool(context.capturing),
            "hidden_shape": list(hidden.shape),
            "groups": {
                name: {
                    "actual_tokens": context.attn_metadata[prefix].num_actual_tokens,
                    "requests": context.attn_metadata[prefix].decode.num_reqs_actual,
                    "query_start_loc": context.attn_metadata[prefix].decode.query_start_loc.cpu().tolist(),
                    "seq_lens": context.attn_metadata[prefix].decode.seq_lens.cpu().tolist(),
                }
                for name, prefix in runtime.prefixes.items()
            },
        }
        if actual_context["graph_mode"] != "NONE" or actual_context["capturing"]:
            self._first_step_stop("actual_layer_context_not_ordinary", layer=index, context=actual_context)
        before_hidden = cpu(hidden)
        self._first_step_pending = None
        result = original(runtime, context, hidden, positions, output, kv_cache)
        torch.npu.synchronize()
        pending = self._first_step_pending
        if pending is None:
            self._first_step_stop("missing_actual_prepared_call", layer=index)
        call, initial, readonly = pending
        values = {name: call.buffers[name] for name in CACHE_NAMES}
        # IndexCache is currently disabled, but preserve this optional Native side effect too.
        impl = runtime.wrapper.dsa_attn.impl
        shared_topk = getattr(impl, "topk_indices_buffer", None)
        if shared_topk is not None:
            values["shared_topk"] = shared_topk
        after = snapshot(values)
        # Native lazily fills sas/qli metadata and clears its factory. Preserve both
        # the original Python fields and any already-bound device Tensor contents.
        requests = {
            id(context.attn_metadata[p].decode): context.attn_metadata[p].decode for p in runtime.prefixes.values()
        }
        metadata_initial = [
            (req, dict(vars(req)), {k: cpu(v) for k, v in vars(req).items() if isinstance(v, torch.Tensor)})
            for req in requests.values()
        ]
        pto_output = cpu(output)
        native_output = torch.empty_like(output)
        native_after = {}
        try:
            for name, value in values.items():
                value.copy_(initial[name])
            torch.npu.synchronize()
            _native_attention_half(runtime.wrapper, hidden, positions, native_output)
            torch.npu.synchronize()
            native_cpu = cpu(native_output)
            native_after = snapshot(values)
        finally:
            # Also wait on Native's auxiliary stream before restoring shared storage.
            torch.npu.synchronize()
            changed_readonly = [
                name for name, value in readonly.items() if not bit_metrics(cpu(call.native[name]), value)["bit_equal"]
            ]
            for name in changed_readonly:
                call.native[name].copy_(readonly[name])
            for name, value in values.items():
                value.copy_(after[name])
            for req, attributes, tensors in metadata_initial:
                for name, tensor in tensors.items():
                    attributes[name].copy_(tensor)
                vars(req).clear()
                vars(req).update(attributes)
            torch.npu.synchronize()
            restored = all(bit_metrics(cpu(value), after[name])["bit_equal"] for name, value in values.items())
            metadata_restored = all(
                vars(req).keys() == attributes.keys()
                and all(vars(req)[name] is value for name, value in attributes.items())
                and all(bit_metrics(cpu(vars(req)[name]), value)["bit_equal"] for name, value in tensors.items())
                for req, attributes, tensors in metadata_initial
            )
            input_intact = bit_metrics(cpu(hidden), before_hidden)["bit_equal"]
            output_intact = bit_metrics(cpu(output), pto_output)["bit_equal"]
            if not (restored and input_intact and output_intact and metadata_restored) or changed_readonly:
                self._first_step_stop("shadow_restore_failure", layer=index)
        row = {
            "forward_id": self._first_step_forward_id,
            "layer": index,
            "tokens": hidden.shape[0],
            "actual_context": actual_context,
            "policy": call.policy.constexprs(),
            "use_index_cache": bool(impl.use_index_cache),
            "skip_topk": bool(impl.skip_topk),
            "output": bit_metrics(pto_output, native_cpu),
            "caches": {name: bit_metrics(after[name], native_after[name]) for name in CACHE_NAMES},
            "shared_topk": bit_metrics(after["shared_topk"], native_after["shared_topk"])
            if shared_topk is not None
            else None,
            "restored_exact": restored,
            "input_unchanged": input_intact,
            "output_unchanged": output_intact,
            "metadata_restored_exact": metadata_restored,
            "changed_readonly_inputs": changed_readonly,
        }
        self._first_step_report["layers"].append(row)
        self._first_step_save()
        if (
            not row["output"]["bit_equal"]
            or not all(m["bit_equal"] for m in row["caches"].values())
            or (row["shared_topk"] is not None and not row["shared_topk"]["bit_equal"])
        ):
            # Only the first failing layer is persisted; no checkpoint weights or unrelated KV banks.
            torch.save(
                {
                    "native_inputs": readonly,
                    "initial_caches": initial,
                    "pto_caches": after,
                    "native_caches": native_after,
                    "pto_output": pto_output,
                    "native_output": native_cpu,
                },
                self._first_step_root / f"first_difference_layer{index:02d}.pt",
            )
            self._first_step_stop("same_input_csa", layer=index, output=row["output"], caches=row["caches"])
        return result

    def offline_begin_observation(self):
        from vllm.distributed import get_tensor_model_parallel_world_size

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime
        from vllm_ascend.utils import enable_dsa_cp, olora_tp_enable, oproj_tp_enable

        result = super().offline_begin_observation()
        if self.vllm_config.parallel_config.data_parallel_rank != 0:
            return result
        assert get_tensor_model_parallel_world_size() == 1
        assert not (enable_dsa_cp() or olora_tp_enable() or oproj_tp_enable()), "shadow requires communication-free CSA"
        actual_pto = self.offline_pto_window_config()
        assert actual_pto["ring_dep_pool"] is None, "retain the default dependency pool"
        self._first_step_root = Path(os.environ["CSA_FIRST_STEP_OUTPUT"])
        self._first_step_indices = {
            id(runtime): index
            for index, layer in enumerate(self.model_runner.get_model().model.layers)
            if (runtime := getattr(layer.self_attn.dsa_attn, "_pto_csa_runtime", None)) is not None
        }
        assert len(self._first_step_indices) == 21
        self._first_step_report = {
            "status": "armed",
            "first_forward": None,
            "layers": [],
            "actual_pto_config": actual_pto,
            "cache_scope": "five physical-page views; excludes allocation alignment prefix/suffix",
            "metadata_scope": "request fields and direct tensors; nested objects/factory closure caches not covered",
        }
        self._first_step_lock = threading.Lock()
        self._first_step_claimed = self._first_step_terminal = False
        self._first_step_forward_id = 0
        self._first_step_active = False
        self._first_step_runtime = None
        original_call = CSAServiceRuntime.__call__
        original_prepared = PreparedDynamicCSA.__call__

        def prepared(call, **kwargs):
            runtime = self._first_step_runtime
            if self._first_step_active and runtime is not None:
                assert self._first_step_pending is None, "one actual CSA call per layer expected"
                assert len(call.buffers) == 55 and not call._diagnostic, "production 55-tensor entry required"
                size = sum(call.buffers[n].numel() * call.buffers[n].element_size() for n in CACHE_NAMES)
                if size > MAX_SNAPSHOT_BYTES:
                    self._first_step_stop("snapshot_budget_exceeded", bytes=size)
                torch.npu.synchronize()
                initial = snapshot({name: call.buffers[name] for name in CACHE_NAMES})
                shared = getattr(runtime.wrapper.dsa_attn.impl, "topk_indices_buffer", None)
                if shared is not None:
                    initial["shared_topk"] = cpu(shared)
                excluded = set(runtime.weights) | {
                    "x_out",
                    "kv_cache",
                    "cmp_kv",
                    "compress_state",
                    "inner_compress_state",
                    "idx_native_kv_cache",
                }
                readonly = {name: cpu(value) for name, value in call.native.items() if name not in excluded}
                self._first_step_pending = (call, initial, readonly)
            return original_prepared(call, **kwargs)

        def paired(runtime, context, hidden, positions, output, kv_cache):
            if not self._first_step_active:
                return original_call(runtime, context, hidden, positions, output, kv_cache)
            assert not torch.npu.is_current_stream_capturing(), "CPU shadow cannot run inside graph capture"
            impl = runtime.wrapper.dsa_attn.impl
            if impl.use_index_cache or impl.skip_topk:
                self._first_step_stop("unsupported_index_cache_shadow", layer=self._first_step_indices[id(runtime)])
            self._first_step_runtime = runtime
            try:
                return self._first_step_pair(runtime, context, hidden, positions, output, kv_cache, original_call)
            finally:
                self._first_step_runtime = None
                self._first_step_pending = None

        PreparedDynamicCSA.__call__ = prepared
        CSAServiceRuntime.__call__ = paired
        origin = self.model_runner._model_forward
        prefix = self._offline_csa_layer_names[0]

        def forward(tokens, *args, **kwargs):
            # Worker error handling may consume another queued execution before
            # the launcher kills the group. Never rerun or overwrite a claimed probe.
            self._first_step_guard()
            context = get_forward_context()
            metadata = (context.attn_metadata or {}).get(prefix)
            selected = (
                not context.capturing
                and not self._threeway_in_dummy_batch
                and not getattr(context, "is_draft_model", False)
                and metadata is not None
                and not metadata.num_prefills
                and metadata.decode is not None
                and metadata.num_actual_tokens > 0
            )
            if not selected:
                return origin(tokens, *args, **kwargs)
            if context.cudagraph_runtime_mode.name != "NONE":
                self._first_step_stop("first_forward_not_ordinary", mode=context.cudagraph_runtime_mode.name)
            descriptor = {
                "graph_mode": context.cudagraph_runtime_mode.name,
                "tokens": tokens,
                "actual_tokens": metadata.num_actual_tokens,
                "requests": metadata.decode.num_reqs_actual,
                "query_start_loc": metadata.decode.query_start_loc.cpu().tolist(),
                "seq_lens": metadata.decode.seq_lens.cpu().tolist(),
                "request_ids": list(self.model_runner.input_batch.req_ids),
            }
            self._first_step_claim(descriptor)
            self._first_step_active = True
            try:
                value = origin(tokens, *args, **kwargs)
            finally:
                self._first_step_active = False
            expected = list(self._first_step_indices.values())
            if not check_complete(self._first_step_report, expected):
                self._first_step_stop("first_step_coverage_failure", expected_layers=expected)
            self._first_step_stop("first_step_complete", layers=expected)
            return value

        self.model_runner._model_forward = forward
        return result
