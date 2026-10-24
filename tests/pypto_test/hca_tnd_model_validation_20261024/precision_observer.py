# SPDX-License-Identifier: Apache-2.0
"""首token精度诊断：同一模型壳切换Native/HCA，并在真实输入上配对HCA半层。

只记录生成首轮，不在warmup/capture中采集。所有rank仍跑相同TP1、DP/EP16模型；
rank0额外做同初态层对照，随后恢复实际PTO输出及写入页。诊断不用于性能计时。
大Tensor快照只写本地；小型JSON保存逐层差异、logits及恢复是否逐bit一致。
"""

import json
import os
from pathlib import Path

import torch

from .observer import HCAObserver
from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.graph_service import HCATNDGraphServiceRuntime
from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.native_adapter import NativeHCACall

_MODE = os.environ["HCA_PRECISION_MODE"]
if _MODE not in ("native", "pto"):
    raise ValueError("HCA_PRECISION_MODE只允许native/pto")
_ORIGINAL_CALL = HCATNDGraphServiceRuntime.__call__
_ORIGINAL_CORE = NativeHCACall.__call__
_OBSERVER = None


def cpu(value):
    return value.detach().cpu().clone()


def bit_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def metrics(a, b):
    from dsv4_csa_validation import compare_tensor

    result = compare_tensor(a, b, 0.01, 0.01)
    result["bit_equal"] = bit_equal(a, b)
    result["bit_different_elements"] = int(a.ne(b).sum())
    return result


def _core(call):
    observer = _OBSERVER
    if observer is not None and observer._precision_current is not None:
        initial = []
        for name, slot in (("ori_cache", "ori_slots"), ("cmp_cache", "cmp_slots"), ("state", "state_slots")):
            value = call.args[name]
            pages = call.args[slot].detach().cpu()[:, 0].to(torch.int64)
            pages = pages[pages >= 0].unique().to(value.device)
            if pages.numel() and int(pages.max()) >= value.shape[0]:
                raise ValueError("实际HCA slot页号越界")
            initial.append((name, value, pages, value.index_select(0, pages).clone()))
        observer._precision_initial = initial
        # 首个HCA层保存不含权重的实际根入参，沿用已有别名/物理布局快照工具。
        # 若存储跨度过大，只记描述符，后续按已定位的层做定向采样。
        if observer._precision_current == 3:
            from dsv4_csa_replay import capture_tensors
            values = {k: v for k, v in call.args.items() if k not in observer._precision_weights}
            groups = {}
            for name, v in values.items():
                key = (str(v.device), v.untyped_storage().data_ptr())
                lo = v.storage_offset() * v.element_size()
                hi = lo + (1 + sum((n-1)*s for n, s in zip(v.shape, v.stride()))) * v.element_size()
                groups.setdefault(key, []).append((lo, hi))
            size = sum(max(b for a,b in r)-min(a for a,b in r) for r in groups.values())
            observer._precision_report["layer3_snapshot_estimated_bytes"] = size
            if size <= 256*1024*1024:
                roles = {k: ("inout" if k in ("state", "ori_cache", "cmp_cache") else "out" if k == "x_out" else "in")
                         for k in values}
                meta, payload = capture_tensors(values, roles, {}, {"state_timing": "before_call", "layer": 3,
                                                                   "scope": "实际模型根入参；权重按正式checkpoint另加载"})
                torch.save(payload, observer._precision_dir / "layer3_inputs.pt")
                (observer._precision_dir / "layer3_inputs.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
            else:
                observer._precision_report["layer3_snapshot_skipped"] = "超过256MiB采集预算，未复制大存储"
    return _ORIGINAL_CORE(call)


def _call(runtime, context, hidden, positions, output, kv_cache):
    from vllm_ascend.ops.dsv4_hca import native_attention_half
    from vllm_ascend.attention.utils import wait_for_kv_layer_from_connector

    observer = _OBSERVER
    capture = observer is not None and observer._precision_active and observer._precision_rank == 0
    if _MODE == "native":
        if capture:
            before = cpu(hidden)
        native_attention_half(runtime.wrapper, hidden, positions, output)
        if capture:
            index = observer._precision_layers[id(runtime)]
            torch.save({"input": before, "output": cpu(output)}, observer._precision_dir / f"layer_{index:02d}.pt")
            observer._precision_report["layers"].append({"layer": index, "path": "native", "input_shape": list(hidden.shape)})
        return
    if not capture:
        return _ORIGINAL_CALL(runtime, context, hidden, positions, output, kv_cache)
    index = observer._precision_layers[id(runtime)]
    before = cpu(hidden)
    wait_for_kv_layer_from_connector(runtime.layer_name)
    observer._precision_current = index
    observer._precision_weights = set(runtime.weights)
    observer._precision_initial = []
    try:
        result = _ORIGINAL_CALL(runtime, context, hidden, positions, output, kv_cache)
    finally:
        observer._precision_current = None
    torch.npu.synchronize()
    pto_output = output.detach().clone()
    initial = observer._precision_initial
    if len(initial) != 3:
        raise RuntimeError("未捕获实际PTO调用的完整写入页初态")
    after = [(name, value, pages, value.index_select(0, pages).clone()) for name,value,pages,_ in initial]
    for name, value, pages, data in initial:
        value.index_copy_(0, pages, data)
    native_output = torch.empty_like(output)
    native_attention_half(runtime.wrapper, hidden, positions, native_output)
    torch.npu.synchronize()
    states = {}
    for name, value, pages, data in after:
        current = value.index_select(0, pages)
        states[name] = {"pages": pages.cpu().tolist(), "bit_equal": bit_equal(current, data),
                        "changed_bytes": int(current.contiguous().view(torch.uint8).ne(data.contiguous().view(torch.uint8)).sum())}
    native_cpu, pto_cpu = cpu(native_output), cpu(pto_output)
    for name, value, pages, data in after:
        value.index_copy_(0, pages, data)
    torch.npu.synchronize()
    restored = all(bit_equal(value.index_select(0, pages), data) for _,value,pages,data in after)
    output_intact = bit_equal(output, pto_output)
    input_intact = bit_equal(cpu(hidden), before)
    if not (restored and output_intact and input_intact):
        raise AssertionError("层对照改变了原PTO状态、输出或输入")
    torch.save({"input": before, "output": pto_cpu, "native_same_input_output": native_cpu},
               observer._precision_dir / f"layer_{index:02d}.pt")
    observer._precision_report["layers"].append({"layer": index, "path": "pto", "same_input_native": metrics(pto_cpu,native_cpu),
                                               "states": states, "restored_exact": restored,
                                               "output_unchanged": output_intact, "input_unchanged": input_intact})
    observer._precision_save()
    return result


NativeHCACall.__call__ = _core
HCATNDGraphServiceRuntime.__call__ = _call


class PrecisionObserver(HCAObserver):
    def _precision_save(self):
        (self._precision_dir / "summary.json").write_text(json.dumps(self._precision_report,ensure_ascii=False,indent=2)+'\n')

    def offline_begin_observation(self):
        global _OBSERVER
        result = super().offline_begin_observation()
        self._precision_rank = self.vllm_config.parallel_config.data_parallel_rank
        self._precision_dir = Path(os.environ["HCA_PRECISION_OUTPUT"]) / f"rank{self._precision_rank}"
        self._precision_dir.mkdir(parents=True, exist_ok=False)
        self._precision_active = False
        self._precision_started = False
        self._precision_current = None
        self._precision_layers = {id(runtime): index for index,runtime in self._hca_runtimes}
        self._precision_report = {"mode": _MODE, "rank": self._precision_rank, "layers": [], "first_forward": None,
                                  "sampling": "真实生成首轮，独立诊断，不用于性能", "logits_captured": False}
        _OBSERVER = self
        runner = self.model_runner
        self._precision_forward_origin = runner._model_forward
        model = runner.get_model()
        self._precision_logits_origin = model.compute_logits
        self._precision_model = model

        def forward(num_tokens, input_ids=None, positions=None, *args, **kwargs):
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            req = (context.attn_metadata or {}).get(self._offline_csa_layer_names[0])
            capture = (not self._precision_started and not context.capturing and req is not None
                       and req.decode is not None and req.decode.max_seq_lens >= 131072)
            if capture:
                self._precision_started = True
                self._precision_active = True
                self._precision_report["first_forward"] = {
                    "tokens": num_tokens, "actual_tokens": req.num_actual_tokens,
                    "requests": req.decode.num_reqs_actual, "graph_mode": context.cudagraph_runtime_mode.name,
                    "query_start_loc": req.decode.query_start_loc.cpu().tolist(),
                    "seq_lens": req.decode.seq_lens.cpu().tolist(), "input_ids": input_ids.cpu().tolist(),
                    "positions": positions.cpu().tolist(), "request_ids": list(runner.input_batch.req_ids),
                }
                self._precision_save()
            try:
                value = self._precision_forward_origin(num_tokens, input_ids, positions, *args, **kwargs)
                if capture and self._precision_rank == 0:
                    hidden = value[0] if runner.use_aux_hidden_state_outputs else value
                    torch.save(cpu(hidden), self._precision_dir / "final_hidden.pt")
                return value
            finally:
                self._precision_active = False

        def logits(hidden):
            value = self._precision_logits_origin(hidden)
            if self._precision_started and not self._precision_report["logits_captured"]:
                tensor = cpu(value)
                scores, ids = tensor.float().topk(10,dim=-1)
                self._precision_report.update(logits_captured=True, logits_shape=list(tensor.shape),
                                              top10_ids=ids.tolist(), top10_scores=scores.tolist())
                if self._precision_rank == 0:
                    torch.save(tensor,self._precision_dir / "logits.pt")
                self._precision_save()
            return value

        runner._model_forward = forward
        model.compute_logits = logits
        return result

    def offline_end_observation(self):
        global _OBSERVER
        self.model_runner._model_forward = self._precision_forward_origin
        self._precision_model.compute_logits = self._precision_logits_origin
        _OBSERVER = None
        result = super().offline_end_observation()
        self._precision_save()
        result["precision_capture"] = str(self._precision_dir / "summary.json")
        return result
