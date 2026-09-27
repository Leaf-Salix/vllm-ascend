# SPDX-License-Identifier: Apache-2.0
"""离线 D 的配置观测、数值诊断和显式性能测量窗口。"""

import json
import os
import time
from collections import Counter
from pathlib import Path


def _monotonic_ulp(actual, expected, torch, floor=0.0):
    """BF16 的 ULP 距离，按单调序计算；floor 用于排除近零元素。

    位模式直接作有符号整数相减是错的：负数的位模式高位为 1，跨零比较会得到约
    32768 的假差值。标准做法是把负数映射成 0x8000 - bits，使整个编码在数轴上单调。
    """
    if actual.dtype is not torch.bfloat16:
        return None
    a = actual.view(torch.int16).to(torch.int32) & 0xFFFF
    b = expected.view(torch.int16).to(torch.int32) & 0xFFFF
    mono = lambda v: torch.where(v & 0x8000 != 0, 0x8000 - v, v)
    distance = (mono(a) - mono(b)).abs()
    if floor > 0.0:
        keep = (actual.float().abs() >= floor) & (expected.float().abs() >= floor)
        distance = distance[keep]
        if distance.numel() == 0:
            return None
    return int(distance.max())


def _enable_swimlane_init():
    """按环境变量给本进程的 pypto.torch.init 补上 DFX 泳道参数。

    pypto 只在第一次 init 时绑定诊断配置，而生产代码在 process_weights_after_loading
    里调用它。worker extension 在 load_model 之前就被解析导入，所以在模块导入时打补丁
    即可采集泳道，不需要为了诊断去改生产实现。只有显式设置环境变量的进程会被改写。
    """
    directory = os.environ.get("OFFLINE_PTO_SWIMLANE_DIR")
    if not directory:
        return
    import pypto.torch

    original = pypto.torch.init

    def init(**kwargs):
        return original(**kwargs, enable_chip_swimlane=4, enable_dep_gen=True, output_dir=directory)

    pypto.torch.init = init


_enable_swimlane_init()


# 捕获期 PTO/Native 选择计数。图模式下这个选择由 eligible() 在**捕获时**定下，
# 之后每次重放都沿用；普通 forward hook 在图重放时不触发，不能用空字典表示有效观测。
_CSA_SELECTION = {}


def _enable_selection_counter():
    """在模型加载前给 CSAServiceRuntime.eligible 打补丁，统计其判定结果。

    worker extension 在 load_model 之前就被解析导入，所以只有在模块导入时打补丁
    才覆盖得到捕获期。Native 后端没有这个模块，import 失败即跳过。
    """
    try:
        # 钩子必须补到实际在跑的那一套算子上，否则 PTO_CSA_VARIANT 一切换就全落空。
        from vllm_ascend.ops.pypto.variant import variant_package
        CSAServiceRuntime = __import__(
            f"{variant_package()}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime
    except Exception:
        return

    origin = CSAServiceRuntime.eligible

    def counted(runtime, context, hidden, positions):
        verdict = origin(runtime, context, hidden, positions)
        layer = getattr(runtime, "layer_name", "?")
        key = f"{'pto' if verdict else 'native'}_tokens{int(hidden.shape[0])}"
        bucket = _CSA_SELECTION.setdefault(layer, {})
        bucket[key] = bucket.get(key, 0) + 1
        return verdict

    CSAServiceRuntime.eligible = counted


_enable_selection_counter()


class OfflineCSAObserver:
    # CSA 的五个 cache group：swa、compressed、state、indexer、indexer_state。
    _OFFLINE_PADDING_GROUPS = 5
    # dummy 步里取几次 slot mapping 样本。
    _OFFLINE_SLOT_SAMPLES = 4
    # dummy 步里复算几次 compact slot mapping。
    _OFFLINE_COMPACT_SAMPLES = 3

    def offline_cache_layout(self):
        """只记录真实缓存描述符，用于定位共享存储边界，不读取设备数据。"""
        from types import SimpleNamespace

        from vllm_ascend.ops.dsa import _build_kv_cache
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import indexer_storage, physical_pages

        def describe(tensor):
            return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype), "pointer": tensor.data_ptr(),
                    "storage_pointer": tensor.untyped_storage().data_ptr(),
                    "storage_offset": tensor.storage_offset(),
                    "storage_bytes": tensor.untyped_storage().nbytes(),
                    "view_bytes": tensor.numel() * tensor.element_size()}

        layers = {}
        for index, layer in enumerate(self.model_runner.get_model().model.layers):
            attention = layer.self_attn
            if attention.compress_ratio != 4:
                continue
            cmp_kv, swa, state, inner_state, key, scale = _build_kv_cache(
                attention.dsa_attn, SimpleNamespace(virtual_engine=None))
            tensors = {"cmp_kv": cmp_kv, "kv_cache": swa,
                       "compress_state": physical_pages(state),
                       "inner_compress_state": physical_pages(inner_state),
                       "idx_kv_cache": indexer_storage(key, scale)}
            views = {name: describe(tensor) for name, tensor in tensors.items()}
            overlaps = []
            names = list(views)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    a, b = views[left], views[right]
                    begin = max(a["pointer"], b["pointer"])
                    end = min(a["pointer"] + a["view_bytes"], b["pointer"] + b["view_bytes"])
                    if begin < end:
                        overlaps.append({"left": left, "right": right, "overlap_bytes": end - begin,
                                         "same_byte_range": (a["pointer"], a["view_bytes"]) ==
                                                            (b["pointer"], b["view_bytes"]),
                                         "same_descriptor": all(a[k] == b[k] for k in
                                                                ("pointer", "shape", "stride", "dtype"))})
            layers[str(index)] = {"views": views, "overlaps": overlaps}
        return {"scope": "仅采集描述符，无PTO计算或精度结论", "layers": layers}

    def offline_begin_observation(self):
        from vllm.forward_context import get_forward_context

        if getattr(self, "_offline_csa_handles", None):
            raise RuntimeError("CSA observation is already active")
        self._offline_csa_counts = {}
        self._offline_csa_handles = []
        model = self.model_runner.get_model()
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            if attention.compress_ratio != 4:
                continue
            counts = Counter()
            self._offline_csa_counts[str(index)] = counts

            def completed(module, args, kwargs, output, counts=counts):
                hidden = kwargs["hidden_states"]
                positions = kwargs["positions"]
                context = get_forward_context()
                runtime = getattr(module.dsa_attn, "_pto_csa_runtime", None)
                selected = runtime is not None and runtime.eligible(context, hidden, positions)
                counts[f"{'pto' if selected else 'native'}_tokens{hidden.shape[0]}"] += 1

            self._offline_csa_handles.append(attention.register_forward_hook(completed, with_kwargs=True))
        if len(self._offline_csa_counts) != 21:
            raise RuntimeError(f"Expected 21 target CSA layers, got {len(self._offline_csa_counts)}")
        return {"target_csa_layers": list(self._offline_csa_counts),
                "dp_rank": self.vllm_config.parallel_config.data_parallel_rank}

    def offline_end_observation(self):
        for handle in self._offline_csa_handles:
            handle.remove()
        self._offline_csa_handles = []
        # forward hook 在图重放下不触发，窗口内的计数通常全是空的；
        # per-layer 命中以捕获期的 capture_time_selection 为准。
        return {"forward_hook_counts": {layer: dict(counts)
                                        for layer, counts in self._offline_csa_counts.items()},
                "capture_time_selection": {layer: dict(counts)
                                           for layer, counts in _CSA_SELECTION.items()},
                "note": "图重放不触发 forward hook，per-layer 命中看 capture_time_selection"}

    def offline_begin_profile(self, directory, start_step, steps, expected_tokens, expected_requests,
                              level=1, forward_events=False):
        """从指定稳态 step 起采集连续 CPU+NPU 数据，不采 Python stack。

        只统计达到稳态构成的 step，窗口两端各做一次同步，确保设备任务完整落在窗口内。
        窗口结束就地关闭，让各 rank 同时进入后处理，避免单 rank 落后拖住集合通信。
        采集本身带同步和落盘开销，这个窗口只用于结构对照，不产出稳态性能结论。
        """
        import torch_npu

        if getattr(self, "_offline_profile", None) is not None:
            raise RuntimeError("Profiling is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        target = os.path.join(directory, f"rank{rank}")
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=(torch_npu.profiler.ProfilerLevel.Level0 if level == 0
                                else torch_npu.profiler.ProfilerLevel.Level1)),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(target),
        )
        runner = self.model_runner
        original = runner.execute_model
        original_forward = runner._model_forward if forward_events else None
        forward_records, active_forward = [], [None]
        state = {"dp_rank": rank, "trace_dir": target, "start_step": start_step,
                 "requested_steps": steps, "expected_tokens": expected_tokens,
                 "expected_requests": expected_requests, "seen_steady_steps": 0,
                 "profiled_steps": 0, "closed": False, "window": [], "observed": {},
                 "profiler_level": level, "forward_events_enabled": forward_events}

        def measured_forward(*args, **kwargs):
            import torch

            entry = active_forward[0]
            if entry is None:
                return original_forward(*args, **kwargs)
            entry["forward_calls"] += 1
            entry["positions_cpu"] = runner._dsa_positions_cpu_buf[:expected_tokens].tolist()
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            entry["begin"], entry["end"] = begin, end
            begin.record()
            try:
                return original_forward(*args, **kwargs)
            finally:
                end.record()

        def profiled(scheduler_output, *args, **kwargs):
            import torch

            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            # 把看到的 (tokens, requests) 分布记下来：判定失败时不必重跑就能诊断。
            key = f"{tokens}/{requests}"
            state["observed"][key] = state["observed"].get(key, 0) + 1
            # 部分 batch 或 prefill 不能冒充约定档位。开始后采连续步，
            # 若窗口中途形状变化，保留原始记录并在收尾判定为不可用于验收。
            steady = tokens == expected_tokens and requests == expected_requests
            index = state["seen_steady_steps"]
            if steady:
                state["seen_steady_steps"] += 1
            active = (state["profiled_steps"] < steps and
                      (state["profiled_steps"] > 0 or (steady and index >= start_step)))
            if active and state["profiled_steps"] == 0:
                torch.npu.synchronize()
                profiler.start()
            if active and forward_events:
                active_forward[0] = {"steady_step_index": index, "forward_calls": 0}
            try:
                return original(scheduler_output, *args, **kwargs)
            finally:
                if active and forward_events:
                    forward_records.append(active_forward[0])
                    active_forward[0] = None
                if active:
                    state["profiled_steps"] += 1
                    state["window"].append({"steady_step_index": index, "scheduled_tokens": tokens,
                                            "requests": requests})
                    if state["profiled_steps"] == steps:
                        torch.npu.synchronize()
                        profiler.stop()
                        state["closed"] = True

        runner.execute_model = profiled
        if forward_events:
            runner._model_forward = measured_forward
        self._offline_profile = (state, profiler, original, original_forward, forward_records)
        return {"dp_rank": rank, "trace_dir": target}

    def offline_end_profile(self):
        import torch

        state, profiler, original, original_forward, forward_records = self._offline_profile
        self.model_runner.execute_model = original
        if original_forward is not None:
            self.model_runner._model_forward = original_forward
        self._offline_profile = None
        if state["profiled_steps"] and not state["closed"]:
            torch.npu.synchronize()
            profiler.stop()
            state["closed"] = True
        if original_forward is not None:
            import math

            valid = [entry for entry in forward_records if entry["forward_calls"] == 1]
            values = [entry["begin"].elapsed_time(entry["end"]) * 1000 for entry in valid]
            timestamps = [entry["begin"].recorded_time() for entry in valid]
            state["profile_forward"] = {
                "scope": "与本trace同一次forward的设备事件；含profiler影响，只用于对齐计时边界，不作为正式成绩",
                "samples_us": values,
                "start_timestamps_raw": timestamps,
                "step_indices": [entry["steady_step_index"] for entry in valid],
                "positions_cpu": [entry["positions_cpu"] for entry in valid],
                "sufficient": (len(valid) == len(forward_records) == state["requested_steps"]
                               and all(math.isfinite(v) and v > 0 for v in values)
                               and all(b > a for a, b in zip(timestamps, timestamps[1:]))),
            }
        # 采不到足够样本时**不要抛异常**：异常会让整个 rank 的 json 落不了盘，
        # 连同 observed 里的诊断信息一起丢失——T2.1 前两轮就是这样，失败了却
        # 拿不到"实际每步调度了多少 token"，只能另想办法查。这里如实记进报告，
        # 由调用方按 profiled_steps 判断是否可用。
        state["sufficient"] = state["profiled_steps"] == state["requested_steps"] and all(
            item["scheduled_tokens"] == state["expected_tokens"] and
            item["requests"] == state["expected_requests"] for item in state["window"])
        return state

    def offline_begin_bitcompare(self, layer_index, expected_tokens, max_samples, mode):
        """在真实生产路径上逐 bit 比对 PTO 与 Native 的 CSA 层输出张量。

        在 CSAServiceRuntime.__call__ 上捕获真实模型的权重、输入、缓存和 metadata，
        补充单卡合成用例之外的层级数值证据。

        在真正调用 kernel 前保存 slot 声明会写的 cache/state 页和输出缓冲，
        两次执行之间恢复初态，不假定重复写天然幂等。只备份触及页，避免复制整份 cache。
        越界和保护区仍需独立验收，不能由这条数值诊断路径推出。
        比完把 output 留成 Native 的值，让本轮继续沿参考轨迹走，避免差异累积。
        """
        import torch
        from dsv4_csa_replay import capture_written_pages, restore_written_pages
        from dsv4_csa_validation import compare_tensor

        from vllm_ascend.ops.pypto.variant import variant_package

        CSAServiceRuntime = __import__(
            f"{variant_package()}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime
        NativeCSACall = __import__(
            f"{variant_package()}.native_adapter", fromlist=["NativeCSACall"]).NativeCSACall

        if getattr(self, "_offline_bitcompare", None) is not None:
            raise RuntimeError("Bit comparison is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        attention = self.model_runner.get_model().model.layers[layer_index].self_attn
        wanted = getattr(attention.dsa_attn, "_pto_csa_runtime", None)
        if wanted is None:
            raise ValueError(f"Layer {layer_index} has no PTO CSA runtime")
        # Native 回退用的是生产入口那个 prefix（pypto_deepseek_v4.csa_attention_forward
        # 传的就是它），不是 runtime.layer_name——后者解析到的是另一个对象，
        # 直接用会在 dsa_forward 里报 'DSAAttention' object has no attribute 'prefix'。
        native_prefix = attention.dsa_attn.prefix
        original = CSAServiceRuntime.__call__
        if mode not in ("native", "self"):
            raise ValueError(f"mode must be 'native' or 'self', got {mode!r}")
        state = {"dp_rank": rank, "layer_index": layer_index, "layer_name": wanted.layer_name,
                 "native_prefix": native_prefix, "mode": mode,
                 "expected_tokens": expected_tokens, "max_samples": max_samples,
                 "samples": [], "seen": 0}

        def compared(runtime, context, hidden, positions, output, kv_cache, *args, **kwargs):
            if runtime is not wanted or hidden.shape[0] != expected_tokens:
                return original(runtime, context, hidden, positions, output, kv_cache, *args, **kwargs)
            state["seen"] += 1
            if len(state["samples"]) >= max_samples:
                return original(runtime, context, hidden, positions, output, kv_cache, *args, **kwargs)
            initial = []
            original_core_call = NativeCSACall.__call__

            def capture_initial(call):
                initial.extend(capture_written_pages(call.args))
                return original_core_call(call)

            NativeCSACall.__call__ = capture_initial
            try:
                result = original(runtime, context, hidden, positions, output, kv_cache, *args, **kwargs)
            finally:
                NativeCSACall.__call__ = original_core_call
            if not initial:
                raise RuntimeError("数值诊断未捕获实际 PTO 调用初态")
            pto = output.detach().clone()
            restore_written_pages(initial)
            if mode == "self":
                # PTO 自比对：同一输入再跑一遍 PTO。这是跨实现比对的前提——kv_fp32 用
                # atomic=pl.AtomicType.Add 且 KV_OK=2，同一输出位置由两个 K 分片的块
                # 原子加，FP32 加法不结合，到达顺序不同低位就不同。自身都不可复现时，
                # 与 Native 的 bit 比对无从归因。注意 torch 的确定性开关管不到 PTO
                # kernel 内部的原子加。
                original(runtime, context, hidden, positions, output, kv_cache, *args, **kwargs)
            else:
                # PTO 现在接管的是整个 attention 半边（mHC pre + input_layernorm +
                # attention + mHC post），hidden 是层间的 [T, HC_MULT, D] 残差流，
                # 不再是归一化后的 [T, D]，所以对照侧也必须走同样范围的 Native 链路。
                from vllm_ascend.ops.dsv4_csa import _native_attention_half
                _native_attention_half(attention.dsa_attn, hidden, positions, output)
            native = output.detach().clone()
            comparison = compare_tensor(pto, native, 0, 0)
            equal = bool(torch.equal(pto.contiguous().view(torch.uint8), native.contiguous().view(torch.uint8)))
            record = {"step": state["seen"], "shape": list(pto.shape), "dtype": str(pto.dtype),
                      "mode": mode, "bit_equal": equal, "elementwise": comparison}
            if not equal and comparison.get("nonfinite") == 0:
                diff = (pto.float() - native.float()).abs()
                mismatch = pto.ne(native)
                scale = float(native.float().abs().max())
                record.update({
                    "mismatches": int(mismatch.sum()), "elements": int(pto.numel()),
                    "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                    # 量级上下文：光有绝对差看不出相对多大。
                    "native_max_abs": scale,
                    "max_abs_over_scale": float(diff.max()) / scale if scale else None,
                    # ULP 必须在单调序下算：BF16 位模式按有符号整数直接相减时，
                    # 正负跨零的两个数会得到约 32768 的巨大差值，那是假象不是差距
                    # （上一版就因此报出 max_ulp=32307）。近零值上 ULP 本身也不可靠，
                    # 所以同时给出只在"两侧同号且绝对值不低于 scale 的千分之一"的
                    # 元素上统计的 ULP，那部分才有解释力。
                    "max_ulp_monotonic": _monotonic_ulp(pto, native, torch),
                    "max_ulp_significant": _monotonic_ulp(pto, native, torch, floor=scale / 1000.0),
                    "first_mismatch": mismatch.nonzero()[:4].tolist(),
                })
            state["samples"].append(record)
            return result

        CSAServiceRuntime.__call__ = compared
        self._offline_bitcompare = (state, original, CSAServiceRuntime)
        return {"dp_rank": rank, "layer_index": layer_index}

    def offline_end_bitcompare(self):
        state, original, runtime_cls = self._offline_bitcompare
        runtime_cls.__call__ = original
        self._offline_bitcompare = None
        samples = state["samples"]
        state["compared"] = len(samples)
        state["all_bit_equal"] = bool(samples) and all(s["bit_equal"] for s in samples)
        # 样本为 0 要如实记录，不能让空集合的 all() 为真而误判通过。
        state["sufficient"] = bool(samples) and len(samples) >= state["max_samples"]
        state["status"] = "MEASURED" if state["sufficient"] and all(
            s["elementwise"].get("nonfinite") == 0 for s in samples
        ) else "FAIL"
        state["scope"] = "零容差逐元素诊断；算术差异另按精度合同验收，不代表独立整模型通过"
        return state

    def offline_begin_forward(self, warmup_steps, expected_tokens, expected_requests, steps=10):
        """只包围 Native _model_forward；execute_model 仅用于辨认实际 decode 档位。"""
        import torch

        if getattr(self, "_offline_forward", None) is not None or getattr(self, "_offline_steady", None) is not None:
            raise RuntimeError("Forward/steady measurement is already active")
        runner = self.model_runner
        original_execute, original_forward = runner.execute_model, runner._model_forward
        state = {"schema": 3, "kind": "model_forward",
                 "dp_rank": self.vllm_config.parallel_config.data_parallel_rank,
                 "warmup_steps": warmup_steps, "requested_steps": steps, "seen_steps": 0,
                 "all_execute_calls": 0, "observed": {}}
        events, active = [], [None]
        torch.npu.reset_peak_memory_stats()

        def forward(*args, **kwargs):
            entry = active[0]
            if entry is None:
                return original_forward(*args, **kwargs)
            entry["forward_calls"] += 1
            # CPU metadata already prepared by the production runner. Keep
            # the actual request positions for comparing work across runs;
            # no device copy, hash, or synchronization is added to timing.
            entry["request_positions"] = runner._dsa_positions_cpu_buf[:expected_tokens].tolist()
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            entry["begin"], entry["end"] = begin, end
            begin.record()
            try:
                return original_forward(*args, **kwargs)
            finally:
                end.record()

        def execute(scheduler_output, *args, **kwargs):
            call = state["all_execute_calls"]
            state["all_execute_calls"] += 1
            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            key = f"{tokens}/{requests}"
            state["observed"][key] = state["observed"].get(key, 0) + 1
            if tokens != expected_tokens or requests != expected_requests:
                return original_execute(scheduler_output, *args, **kwargs)
            index = state["seen_steps"]
            state["seen_steps"] += 1
            if index < warmup_steps or len(events) >= steps:
                return original_execute(scheduler_output, *args, **kwargs)
            entry = {"call": call, "index": index, "tokens": tokens, "requests": requests, "forward_calls": 0}
            active[0] = entry
            try:
                return original_execute(scheduler_output, *args, **kwargs)
            finally:
                active[0] = None
                events.append(entry)

        runner.execute_model, runner._model_forward = execute, forward
        self._offline_forward = (state, events, original_execute, original_forward)
        return {"dp_rank": state["dp_rank"], "kind": state["kind"], "warmup_steps": warmup_steps}

    def offline_end_forward(self):
        import math
        import statistics

        import torch

        state, events, original_execute, original_forward = self._offline_forward
        self.model_runner.execute_model, self.model_runner._model_forward = original_execute, original_forward
        self._offline_forward = None
        state["peak_allocated_bytes"] = int(torch.npu.max_memory_allocated())
        state["peak_reserved_bytes"] = int(torch.npu.max_memory_reserved())
        torch.npu.synchronize()
        valid = [entry for entry in events if entry["forward_calls"] == 1]
        values = [entry["begin"].elapsed_time(entry["end"]) * 1000 for entry in valid]
        stamps = [entry["begin"].recorded_time() for entry in valid]
        state.update(measured_steps=len(events), step_tokens=[entry["tokens"] for entry in events],
                     step_requests=[entry["requests"] for entry in events],
                     steady_step_indices=[entry["index"] for entry in events],
                     step_positions_cpu=[entry["request_positions"] for entry in events])
        state["forward"] = {"samples_us": values, "start_timestamps_raw": stamps,
                            "scope": "_model_forward 调用前后设备事件；不含 metadata 准备、logits、采样、"
                                     "DSpark 草稿或步间调度等待；不逐步同步"}
        if values:
            state["forward"].update(mean_us=statistics.mean(values), p50_us=statistics.median(values),
                                     p95_us=sorted(values)[math.ceil(len(values) * .95) - 1])
        state["sufficient"] = (
            len(valid) == len(events) == state["requested_steps"]
            and all(math.isfinite(v) and v > 0 for v in values)
            and all(b > a for a, b in zip(stamps, stamps[1:]))
            and all(b["call"] == a["call"] + 1 for a, b in zip(events, events[1:])))
        return state

    def offline_begin_steady(self, warmup_steps, expected_tokens, expected_requests, cycles=10):
        """记录 execute、采样/草稿完成点及连续步骤起点，不逐步增加同步。"""
        import torch

        if getattr(self, "_offline_steady", None) is not None:
            raise RuntimeError("Steady measurement is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        runner = self.model_runner
        original, original_sample = runner.execute_model, runner.sample_tokens
        state = {"dp_rank": rank, "schema": 2, "warmup_steps": warmup_steps, "seen_steps": 0,
                 "expected_tokens": expected_tokens, "expected_requests": expected_requests,
                 "requested_cycles": cycles,
                 "observed": {}, "step_seconds": [], "step_tokens": [], "step_requests": [],
                 "all_execute_calls": 0, "incomplete_sample_steps": 0}
        events = []
        pending = [None]
        torch.npu.reset_peak_memory_stats()

        def timed(scheduler_output, *args, **kwargs):
            if pending[0] is not None:
                state["incomplete_sample_steps"] += 1
                pending[0] = None
            call_index = state["all_execute_calls"]
            state["all_execute_calls"] += 1
            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            key = f"{tokens}/{requests}"
            state["observed"][key] = state["observed"].get(key, 0) + 1
            if tokens != expected_tokens or requests != expected_requests:
                return original(scheduler_output, *args, **kwargs)
            index = state["seen_steps"]
            state["seen_steps"] += 1
            # 固定前部窗口，避开请求结束时引擎对 sampled_token_ids 的长度裁剪。
            if index < warmup_steps or len(events) >= cycles + 1:
                return original(scheduler_output, *args, **kwargs)
            begin, end, sample_end = (torch.npu.Event(enable_timing=True) for _ in range(3))
            entry = {"begin": begin, "execute_end": end, "sample_end": sample_end,
                     "call_index": call_index, "steady_index": index, "sample_complete": False}
            begin.record()
            start = time.perf_counter()
            entry["host_start"] = start
            pending[0] = entry
            try:
                return original(scheduler_output, *args, **kwargs)
            finally:
                end.record()
                events.append(entry)
                state["step_seconds"].append(time.perf_counter() - start)
                state["step_tokens"].append(tokens)
                state["step_requests"].append(requests)

        def sampled(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            entry = pending[0]
            if entry is not None:
                entry["sample_end"].record()
                entry["sample_complete"] = True
                # 异步引擎随后填充同一个 CPU 输出对象；收尾时读，不在热路径等 D2H。
                entry["output"] = getattr(result, "_model_runner_output", result)
                pending[0] = None
            return result

        runner.execute_model, runner.sample_tokens = timed, sampled
        self._offline_steady = (state, original, original_sample, events, pending)
        return {"dp_rank": rank, "warmup_steps": warmup_steps, "schema": 2}

    def offline_end_steady(self):
        import math
        import statistics

        import torch

        state, original, original_sample, events, pending = self._offline_steady
        self.model_runner.execute_model, self.model_runner.sample_tokens = original, original_sample
        self._offline_steady = None
        if pending[0] is not None:
            state["incomplete_sample_steps"] += 1
        state["measured_steps"] = len(events)
        state["peak_allocated_bytes"] = int(torch.npu.max_memory_allocated())
        state["peak_reserved_bytes"] = int(torch.npu.max_memory_reserved())
        torch.npu.synchronize()
        stamps = [e["begin"].recorded_time() for e in events]
        device_us = [e["begin"].elapsed_time(e["execute_end"]) * 1000 for e in events]
        valid_events = bool(stamps) and all(b > a for a, b in zip(stamps, stamps[1:])) and all(
            math.isfinite(value) and value > 0 for value in device_us)

        def distribution(values):
            if not values:
                return {}
            ordered = sorted(values)
            return {"mean_us": statistics.mean(values), "p50_us": statistics.median(values),
                    "p95_us": ordered[math.ceil(len(values) * 0.95) - 1]}

        state["device"] = {"samples_us": device_us, "start_timestamps_raw": stamps,
                           "valid_events": valid_events,
                           "scope": "execute_model 设备区间；完整周期见 decode_cycle",
                           **distribution(device_us)}
        complete = [e for e in events if e["sample_complete"]]
        state["execute_sample_device"] = {
            "samples_us": [e["begin"].elapsed_time(e["sample_end"]) * 1000 for e in complete],
            "scope": "execute_model 起点至 sample_tokens 返回前，包含采样与 DSpark 草稿"}
        cycles, host_cycles, output_counts, indices = [], [], [], []
        for left, right in zip(events, events[1:]):
            if not (left["sample_complete"] and right["sample_complete"] and
                    right["call_index"] == left["call_index"] + 1):
                continue
            rows = getattr(left["output"], "sampled_token_ids", None)
            count = None
            if isinstance(rows, list) and len(rows) == state["expected_requests"] and all(
                    isinstance(row, list) and row and all(type(t) is int and t >= 0 for t in row) for row in rows):
                count = sum(map(len, rows))
            cycles.append(left["begin"].elapsed_time(right["begin"]) * 1000)
            host_cycles.append(right["host_start"] - left["host_start"])
            output_counts.append(count)
            indices.append(left["steady_index"])
        valid_cycles = bool(cycles) and all(math.isfinite(v) and v > 0 for v in cycles)
        cycle_sufficient = (len(cycles) == state["requested_cycles"] and valid_cycles and
                            all(v is not None for v in output_counts))
        state["decode_cycle"] = {
            "samples_us": cycles, "host_samples_seconds": host_cycles, "steady_step_indices": indices,
            "actual_output_tokens": output_counts, "sufficient": cycle_sufficient,
            "scope": "连续满档 execute_model 起点间隔，已确认中间 sample_tokens 完成；"
                     "含采样、DSpark 草稿与引擎调度间隙，不含加载/前缀恢复",
            **distribution(cycles)}
        if cycle_sufficient:
            state["decode_cycle"]["actual_output_tokens_per_second"] = sum(output_counts) / (sum(cycles) * 1e-6)
        samples = state["step_seconds"]
        if samples:
            ordered = sorted(samples)
            state["p50_seconds"] = statistics.median(samples)
            state["p95_seconds"] = ordered[math.ceil(len(samples) * 0.95) - 1]
            state["mean_seconds"] = sum(samples) / len(samples)
            state["total_seconds"] = sum(samples)
            state["total_tokens"] = sum(state["step_tokens"])
            state["scheduled_tokens_per_host_second"] = state["total_tokens"] / state["total_seconds"]
        state["sufficient"] = valid_events and cycle_sufficient and not state["incomplete_sample_steps"]
        return state

    def offline_begin_host_profile(self, directory, start_step, steps, expected_tokens, expected_requests):
        """在稳态 step 上开 cProfile，定位 CSA 调用里的主机侧耗时函数。

        cProfile 会放大 Python 调用开销，得到的是相对归因，不能与设备侧耗时直接相加，
        也不能当作稳态性能结论。窗口构成判定与 NPU 采集一致，只统计满批稳态 step。
        """
        import cProfile

        if getattr(self, "_offline_host_profile", None) is not None:
            raise RuntimeError("Host profiling is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        profiler = cProfile.Profile()
        runner = self.model_runner
        original = runner.execute_model
        state = {"dp_rank": rank, "start_step": start_step, "requested_steps": steps,
                 "expected_tokens": expected_tokens, "expected_requests": expected_requests,
                 "seen_steady_steps": 0, "profiled_steps": 0, "closed": False,
                 "path": os.path.join(directory, f"rank{rank}.prof")}

        def profiled(scheduler_output, *args, **kwargs):
            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            steady = tokens == expected_tokens and requests == expected_requests
            index = state["seen_steady_steps"]
            if steady:
                state["seen_steady_steps"] += 1
            active = steady and index >= start_step and state["profiled_steps"] < steps
            if active and state["profiled_steps"] == 0:
                profiler.enable()
            try:
                return original(scheduler_output, *args, **kwargs)
            finally:
                if active:
                    state["profiled_steps"] += 1
                    if state["profiled_steps"] == steps:
                        profiler.disable()
                        state["closed"] = True

        runner.execute_model = profiled
        self._offline_host_profile = (state, profiler, original)
        return {"dp_rank": rank, "path": state["path"]}

    def offline_end_host_profile(self):
        import pstats

        state, profiler, original = self._offline_host_profile
        self.model_runner.execute_model = original
        self._offline_host_profile = None
        if state["profiled_steps"] and not state["closed"]:
            profiler.disable()
            state["closed"] = True
        if state["profiled_steps"] != state["requested_steps"]:
            raise RuntimeError(f"Host-profiled {state['profiled_steps']} steady steps, "
                               f"requested {state['requested_steps']}")
        os.makedirs(os.path.dirname(state["path"]), exist_ok=True)
        profiler.dump_stats(state["path"])
        rows = []
        for func, (calls, primitive, total, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append({"function": f"{func[0]}:{func[1]}({func[2]})", "calls": calls,
                         "primitive_calls": primitive, "tottime_seconds": round(total, 4),
                         "cumtime_seconds": round(cumulative, 4)})
        state["top_by_tottime"] = sorted(rows, key=lambda r: r["tottime_seconds"], reverse=True)[:30]
        state["top_by_cumtime"] = sorted(rows, key=lambda r: r["cumtime_seconds"], reverse=True)[:30]
        return state

    def offline_begin_padding_capture(self, directory, layer_index):
        """在真实离线 D 上捕获一次带补位请求的 decode 步，落盘 Native metadata 与 compact 形状。

        挂在 Native 的 AscendDSAMetadataBuilder.build 上，而不是 CSAServiceRuntime：
        后者只在 PTO 后端存在。builder 两个后端都会走，看到的就是生产路径的 metadata。
        （T1.4 之前 can_replay_csa_graph 还会在补位时主动回退 eager，把 padding 自己
        消掉，这也是必须挂 builder 的原因之一；该闸门已于 T1.4 放开。）

        只读取并落盘小张量，额外只多调一次 Native 自己的 compressor_metadata
        生产器来量其真实形状；不改变本步的计算路径与结果。

        同时统计 can_replay_csa_graph 的判定结果，用来证明补位档位确实进入了图重放，
        而不是静默回退 Native/eager（T1.4 的完成判据之一）。
        """
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        if getattr(self, "_offline_padding", None) is not None:
            raise RuntimeError("Padding capture is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        original = AscendDSAMetadataBuilder.build
        state = {"dp_rank": rank, "layer_index": layer_index,
                 "captured": 0, "builds_seen": 0, "padded_builds": 0,
                 "dummy_runs": 0, "dummy_tokens": [],
                 "step_probes": 0, "step_probe_results": [],
                 "in_dummy": False, "slot_samples": 0, "slot_probe_results": [],
                 "recaptures": 0,
                 "compact_samples": 0, "compact_probe_results": [],
                 "replay_calls": 0, "replay_padded": 0,
                 "replay_padded_allowed": 0, "replay_rejected": 0,
                 "records": [], "directory": str(directory)}
        self._offline_replay_probe(state)
        self._offline_dummy_probe(state)
        self._offline_slot_probe(state)
        self._offline_recapture_probe(state)

        def probed(builder, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
            result = original(builder, common_prefix_len, common_attn_metadata, fast_build, **kwargs)
            state["builds_seen"] += 1
            # dummy 步里暂存 decode metadata，供 dummy 返回后复算 compact slot。
            # 注意**不能**挂在补位分支里：dummy run 的所有请求都是假的、
            # num_reqs_actual == num_reqs，根本不带补位，挂那儿永远采不到样本
            # （compact_slot_probe 那轮即因此 0 样本）。
            # 也不在这里直接算：算子虽只产出 metadata、不碰 KV cache，
            # 放在 build 内仍有扰动本步的风险。
            if state["in_dummy"] and state["compact_samples"] < self._OFFLINE_COMPACT_SAMPLES:
                ratio = int(getattr(builder, "compressor_ratio", 0))
                if ratio == 4 and getattr(result, "decode", None) is not None:
                    state["pending_compact"] = (ratio, result.decode)
            actual = kwargs.get("num_reqs_actual")
            padded = actual is not None and actual < common_attn_metadata.num_reqs
            if not padded:
                return result
            state["padded_builds"] += 1
            # 一步里每个 cache group 各 build 一次，采满一轮即停，不逐步累积。
            if state["captured"] < self._OFFLINE_PADDING_GROUPS:
                record = self._offline_padding_record(builder, common_attn_metadata, actual, result)
                if record is not None:
                    state["captured"] += 1
                    state["records"].append(record)
            return result

        AscendDSAMetadataBuilder.build = probed
        self._offline_padding = (state, original)
        return dict(state)

    def _offline_replay_probe(self, state):
        """统计 can_replay_csa_graph 的判定，只计数不改判定结果。

        Native 后端不注入 PyptoCSADeepseekV4ForCausalLM，is_csa_model 为假，
        该函数根本不会被调用，计数保持为 0——这本身就是两个后端的区分证据。
        """
        from vllm_ascend.worker import model_runner_v1

        origin = getattr(model_runner_v1, "can_replay_csa_graph", None)
        if origin is None:
            state["replay_probe"] = "absent"
            return

        def counted(*, num_tokens, num_reqs, uniform_decode, padded_tokens):
            verdict = origin(num_tokens=num_tokens, num_reqs=num_reqs,
                             uniform_decode=uniform_decode, padded_tokens=padded_tokens)
            state["replay_calls"] += 1
            if num_tokens < padded_tokens:
                state["replay_padded"] += 1
                if verdict:
                    state["replay_padded_allowed"] += 1
            if not verdict:
                state["replay_rejected"] += 1
            return verdict

        model_runner_v1.can_replay_csa_graph = counted
        self._offline_replay_origin = origin

    def _offline_compact_sample(self, state, runner, num_tokens):
        """复算 dummy 步的 compact slot mapping，判断它会不会写进 cache。

        compact slot（cmp_slot_mapping／idx_slot_mapping）由 compressor_metadata
        算子在图内从 start_pos 与 block_table 现算，**不来自**被 fill 成 -1 的那个
        slot_mapping 缓冲，所以主 slot 那轮测量覆盖不到它。图内产出的张量 Python
        侧读不到，但算子是纯 metadata 生产者（只产出 cos/sin/slots，不碰 KV cache），
        用同一份 metadata 复算一次即可得到与图内一致的结果。

        必须取同一 compress_ratio 的 impl，按 builder 的 compressor_ratio 找到对应层。
        """
        pending = state.pop("pending_compact", None)
        if pending is None or state["compact_samples"] >= self._OFFLINE_COMPACT_SAMPLES:
            return
        ratio, decode = pending
        try:
            import torch

            impl = None
            for layer in runner.get_model().model.layers:
                attention = layer.self_attn
                if getattr(attention, "compress_ratio", 0) == ratio:
                    impl = attention.dsa_attn.dsa_attn.impl
                    break
            if impl is None:
                state.setdefault("compact_probe_error", f"no layer with compress_ratio={ratio}")
                return
            torch.npu.synchronize()
            _cos, _sin, slots = impl._compute_compressor_metadata(decode)
            flat = slots.reshape(-1, slots.shape[-1]) if slots.dim() > 1 else slots.reshape(-1, 1)
            pages = flat[:, 0]
            state["compact_samples"] += 1
            state["compact_probe_results"].append({
                "num_tokens": int(num_tokens),
                "rows": int(flat.shape[0]),
                "non_negative_pages": int((pages >= 0).sum().item()),
                "max_page": int(pages.max().item()),
                "min_page": int(pages.min().item()),
            })
        except Exception as error:
            state.setdefault("compact_probe_error", repr(error))

    def _offline_recapture_probe(self, state):
        """统计采集窗口内新建了多少个 NPUGraph，用来判定有没有 replay 期重新捕获。

        `ACLGraphWrapper.__call__` 里 `entry.aclgraph is None` 走捕获分支、
        否则重放，捕获时会 `torch.npu.NPUGraph()`。采集窗口开在预热与捕获之后，
        窗口内再出现新的 NPUGraph 就意味着重新捕获——这正是 T1.5 那条
        "无 replay 期重新编译"要排除的情形。计数为 0 即判定通过。
        """
        try:
            import torch

            origin = torch.npu.NPUGraph
        except Exception as error:
            state["recapture_probe"] = f"unavailable: {error!r}"
            return

        def counted(*args, **kwargs):
            state["recaptures"] += 1
            return origin(*args, **kwargs)

        torch.npu.NPUGraph = counted
        self._offline_recapture_origin = origin

    def _offline_slot_probe(self, state):
        """记录 dummy 步里 PTO 实际拿到的 slot mapping。

        整份缓存视图对比在这套布局下不可能成立——`decode_cache_layout_v1` 实测
        `cmp_kv` 与 `compress_state` 共用同一块 678MB 分配、重叠 676MB，写一处
        会让多个视图一起"变化"。改为直接看**输入**：若某个 slot mapping 全为
        -1，kernel 的 `page >= 0` 守卫就必然挡住，写不进去，与存储布局无关。
        """
        try:
            # 钩子必须补到实际在跑的那一套算子上，否则 PTO_CSA_VARIANT 一切换就全落空。
            from vllm_ascend.ops.pypto.variant import variant_package
            CSAServiceRuntime = __import__(
                f"{variant_package()}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime
        except Exception as error:
            state["slot_probe"] = f"unavailable: {error!r}"
            return

        origin = CSAServiceRuntime.__call__

        def probed(runtime, context, hidden, positions, output, kv_cache):
            if state["in_dummy"] and state["slot_samples"] < self._OFFLINE_SLOT_SAMPLES:
                try:
                    metadata = context.attn_metadata
                    record = {"dummy_index": state["dummy_runs"], "tokens": int(hidden.shape[0])}
                    for name, prefix in runtime.prefixes.items():
                        decode = metadata[prefix].decode
                        slots = getattr(decode, "slot_mapping", None)
                        if slots is None:
                            continue
                        flat = slots.reshape(-1)
                        record[name] = {"count": int(flat.numel()),
                                        "non_negative": int((flat >= 0).sum().item()),
                                        "max": int(flat.max().item())}
                    state["slot_samples"] += 1
                    state["slot_probe_results"].append(record)
                except Exception as error:
                    state.setdefault("slot_probe_error", repr(error))
            return origin(runtime, context, hidden, positions, output, kv_cache)

        CSAServiceRuntime.__call__ = probed
        self._offline_slot_origin = origin

    def _offline_dummy_body(self, state, origin, runner, num_tokens, args, kwargs):
        """dummy 的实际执行体；顺便读一次常驻 slot mapping 缓冲。

        图重放不跑 Python 前向闸门，所以 hook CSAServiceRuntime.__call__ 在
        dummy 步上一次都不会触发（实测 dummy_runs=26 而 slot_samples=0）。
        但重放读的就是这些固定地址的常驻缓冲，从 Python 侧可以直接读到。
        `model_runner_v1.py` 在非图捕获时会把它们填成 -1，这里核实是否属实：
        若确为全 -1，kernel 的 `page >= 0` 守卫必然挡住主 slot 那条写入路径。
        """
        result = origin(runner, num_tokens, *args, **kwargs)
        self._offline_compact_sample(state, runner, num_tokens)
        if state["slot_samples"] >= self._OFFLINE_SLOT_SAMPLES:
            return result
        try:
            import torch

            torch.npu.synchronize()
            record = {"dummy_index": state["dummy_runs"], "num_tokens": int(num_tokens),
                      "groups": {}}
            for gid in range(len(runner.kv_cache_config.kv_cache_groups)):
                table = runner.input_batch.block_table[gid]
                flat = table.slot_mapping.gpu.reshape(-1)
                entry = {
                    "count": int(flat.numel()),
                    "non_negative": int((flat >= 0).sum().item()),
                    "max": int(flat.max().item()),
                }
                # compact slot mapping 由算子在图内从 start_pos 与 block_table 现算，
                # 不来自上面那个被 fill 成 -1 的缓冲。但它的**输入**同样是常驻的，
                # 这里一并读：若 block_table 全为 0，算出的 compact slot 必然指向
                # 0 号页，而 0 号页是 vLLM 保留的 null block（永不分配给任何请求），
                # 那条写入路径就是无害的。
                blocks = getattr(table, "block_table", None)
                gpu = getattr(blocks, "gpu", blocks)
                if gpu is not None and hasattr(gpu, "reshape"):
                    bt = gpu.reshape(-1)
                    entry["block_table"] = {
                        "count": int(bt.numel()),
                        "non_zero": int((bt != 0).sum().item()),
                        "max": int(bt.max().item()),
                    }
                record["groups"][str(gid)] = entry
            state["slot_samples"] += 1
            state["slot_probe_results"].append(record)
        except Exception as error:
            state.setdefault("slot_probe_error", repr(error))
        return result

    def _offline_dummy_probe(self, state):
        """统计本 rank 跑了多少次 dummy batch，以及其中有多少次是空调度。

        空 rank（D04 / T1.6）不能靠耗时推断：rank0 与 rank1 测量耗时相同只能说明
        两者被 DP 锁步，说明不了引擎到底有没有真的空转。这里直接数
        NPUModelRunner._dummy_run 的调用次数，并按 with_prefill / num_reqs 分类。
        """
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        origin = getattr(NPUModelRunner, "_dummy_run", None)
        if origin is None:
            state["dummy_probe"] = "absent"
            return

        def counted(runner, num_tokens, *args, **kwargs):
            state["dummy_runs"] += 1
            state["dummy_tokens"].append(int(num_tokens))
            state["in_dummy"] = True
            try:
                return self._offline_dummy_body(state, origin, runner, num_tokens, args, kwargs)
            finally:
                state["in_dummy"] = False

        NPUModelRunner._dummy_run = counted
        self._offline_dummy_origin = origin

    def _offline_padding_record(self, builder, common, num_reqs_actual, result):
        """落盘这一份补位 metadata 的关键量；拿不到 decode 段时返回 None。"""
        import torch

        decode = getattr(result, "decode", None)
        if decode is None:
            return None
        seq_lens = decode.seq_lens
        entry = {
            "prefix": list(getattr(builder, "layer_names", []) or [])[:1],
            "compressor_ratio": int(getattr(builder, "compressor_ratio", 0)),
            "num_reqs_padded": int(common.num_reqs),
            "num_reqs_actual": int(num_reqs_actual),
            "num_actual_tokens": int(common.num_actual_tokens),
            "num_input_tokens": int(getattr(common, "num_input_tokens", 0)),
            "num_decodes": int(result.num_decodes),
            # Native 归一化的三个量，用于在计划第 4 节的方案 C 与 D 之间定夺。
            "seq_lens": seq_lens.tolist(),
            "seq_lens_zero_count": int((seq_lens == 0).sum().item()),
            "query_start_loc": decode.query_start_loc.tolist(),
            "block_table_shape": list(decode.block_table.shape),
            "block_table_nonzero_per_row": torch.count_nonzero(
                decode.block_table.reshape(decode.block_table.shape[0], -1), dim=-1).tolist(),
        }
        start_pos = getattr(decode, "start_pos", None)
        if start_pos is not None:
            entry["start_pos"] = start_pos.tolist()
        positions = getattr(common, "positions", None)
        if positions is not None:
            entry["positions"] = positions[: int(getattr(common, "num_input_tokens", 0)) or None].tolist()

        # compact 行数就是 Native 自己算好的 num_compressed_tokens
        # （dsa_v1.py:606 _num_compressor_metadata_rows 的结果），直接读取即可，
        # 不额外调用 compressor_metadata 算子：那需要与本 builder 同一层的 impl，
        # 取错层会因 compress_ratio 不匹配而报错，也会扰动本步。
        rows = getattr(decode, "num_compressed_tokens", None)
        if rows is not None:
            entry["num_compressed_tokens"] = int(rows)
        return entry


    def _offline_padding_write(self, directory, rank, records):
        import pathlib

        target = pathlib.Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"padding_capture_rank{rank}.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def offline_end_padding_capture(self):
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        if getattr(self, "_offline_padding", None) is None:
            return {"dp_rank": self.vllm_config.parallel_config.data_parallel_rank, "captured": 0}
        state, original = self._offline_padding
        AscendDSAMetadataBuilder.build = original
        origin = getattr(self, "_offline_replay_origin", None)
        if origin is not None:
            from vllm_ascend.worker import model_runner_v1
            model_runner_v1.can_replay_csa_graph = origin
            self._offline_replay_origin = None
        recapture_origin = getattr(self, "_offline_recapture_origin", None)
        if recapture_origin is not None:
            import torch
            torch.npu.NPUGraph = recapture_origin
            self._offline_recapture_origin = None
        slot_origin = getattr(self, "_offline_slot_origin", None)
        if slot_origin is not None:
            # 钩子必须补到实际在跑的那一套算子上，否则 PTO_CSA_VARIANT 一切换就全落空。
            from vllm_ascend.ops.pypto.variant import variant_package
            CSAServiceRuntime = __import__(
                f"{variant_package()}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime
            CSAServiceRuntime.__call__ = slot_origin
            self._offline_slot_origin = None
        dummy_origin = getattr(self, "_offline_dummy_origin", None)
        if dummy_origin is not None:
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
            NPUModelRunner._dummy_run = dummy_origin
            self._offline_dummy_origin = None
        self._offline_padding = None
        tokens = state.pop("dummy_tokens", [])
        if tokens:
            histogram = {}
            for value in tokens:
                histogram[value] = histogram.get(value, 0) + 1
            state["dummy_token_histogram"] = dict(sorted(histogram.items()))
        directory, rank = state.pop("directory"), state["dp_rank"]
        records = state.pop("records")
        if records:
            self._offline_padding_write(directory, rank, records)
        return dict(state)

    def offline_begin_argdump(self, layer_index, out_dir, expected_tokens):
        """保存调用前初态与调用后参考，供保留布局和别名的单卡回放。"""
        import importlib

        from dsv4_csa_replay import argument_roles, capture_tensors, save_snapshot

        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.nz_mode import WEIGHT_NZ_MODE, root_weight_layouts
        from vllm_ascend.ops.pypto.variant import selected_variant, variant_package

        package = variant_package()
        adapter = importlib.import_module(f"{package}.native_adapter")
        roots = importlib.import_module(f"{package}.decode_csa")
        NativeCSACall = adapter.NativeCSACall
        kernel = roots.decode_csa_tp1_layer_test
        root = roots._decode_csa_tp1_layer
        roles = argument_roles(root)
        if getattr(self, "_offline_argdump", None) is not None:
            raise RuntimeError("Argument dump is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        attention = self.model_runner.get_model().model.layers[layer_index].self_attn
        wanted = getattr(attention.dsa_attn, "_pto_csa_runtime", None)
        if wanted is None:
            raise ValueError(f"Layer {layer_index} has no PTO CSA runtime")
        state = {"dp_rank": rank, "layer_index": layer_index, "layer_name": wanted.layer_name,
                 "expected_tokens": expected_tokens, "enabled": rank == 0, "dumped": 0, "path": None,
                 "seen": {}}
        original_init, original_call = NativeCSACall.__init__, NativeCSACall.__call__
        selected = {}

        def traced_init(call, *args, **kwargs):
            original_init(call, *args, **kwargs)
            if not state["enabled"] or kwargs.get("layer_name") != wanted.layer_name:
                return
            tokens = call.args["x_hc"].shape[0]
            state["seen"][str(tokens)] = state["seen"].get(str(tokens), 0) + 1
            if selected or tokens != expected_tokens:
                return
            values = {name: call.args[name] for name in kernel.param_names}
            source = {"variant": selected_variant(), "package": package, "weight_nz_mode": WEIGHT_NZ_MODE,
                      "state_timing": "before_call", "root": kernel.__name__}
            meta, payload = capture_tensors(values, roles, root_weight_layouts(root), source)
            meta.update(layer_index=layer_index, layer_name=wanted.layer_name, tokens=tokens,
                        reference={"file": "csa_reference.pt", "kind": "source_PTO_after_one_call",
                                   "outputs": list(kernel.output_param_names)})
            save_snapshot(out_dir, meta, payload)
            selected["call"] = call

        def traced_call(call):
            result = original_call(call)
            if call is selected.get("call") and not state["dumped"]:
                import torch

                # 参考值只用于逐元素对照；回放初态与别名来自独立的原始存储快照。
                reference = {name: call.args[name].detach().to("cpu", copy=True)
                             for name in kernel.output_param_names}
                torch.save(reference, Path(out_dir) / "csa_reference.pt")
                state.update(dumped=1, path=str(Path(out_dir) / "csa_args.pt"))
            return result

        NativeCSACall.__init__, NativeCSACall.__call__ = traced_init, traced_call
        self._offline_argdump = (state, original_init, original_call, NativeCSACall)
        return dict(state)

    def offline_end_argdump(self):
        entry = getattr(self, "_offline_argdump", None)
        if entry is None:
            raise RuntimeError("Argument dump was not started")
        state, original_init, original_call, cls = entry
        cls.__init__, cls.__call__ = original_init, original_call
        self._offline_argdump = None
        return dict(state)

    def offline_begin_swimlane(self, layer_index, expected_tokens):
        """只给一层、一次达到稳态构成的 CSA 调用开 DFX 窗口，其余调用保持原路径。"""
        import pypto.torch

        # 钩子必须补到实际在跑的那一套算子上，否则 PTO_CSA_VARIANT 一切换就全落空。
        from vllm_ascend.ops.pypto.variant import variant_package
        CSAServiceRuntime = __import__(
            f"{variant_package()}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime

        if getattr(self, "_offline_swimlane", None) is not None:
            raise RuntimeError("Swimlane capture is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        if not os.environ.get("OFFLINE_PTO_SWIMLANE_DIR"):
            self._offline_swimlane = None
            return {"dp_rank": rank, "enabled": False}
        attention = self.model_runner.get_model().model.layers[layer_index].self_attn
        wanted = getattr(attention.dsa_attn, "_pto_csa_runtime", None)
        if wanted is None:
            raise ValueError(f"Layer {layer_index} has no PTO CSA runtime")
        original = CSAServiceRuntime.__call__
        state = {"dp_rank": rank, "enabled": True, "layer_index": layer_index,
                 "layer_name": wanted.layer_name, "expected_tokens": expected_tokens, "captured": 0}

        def traced(runtime, context, hidden, *args, **kwargs):
            if runtime is not wanted or state["captured"] or hidden.shape[0] != expected_tokens:
                return original(runtime, context, hidden, *args, **kwargs)
            pypto.torch.begin_dfx()
            try:
                return original(runtime, context, hidden, *args, **kwargs)
            finally:
                pypto.torch.end_dfx()
                state["captured"] += 1

        CSAServiceRuntime.__call__ = traced
        self._offline_swimlane = (state, original)
        return dict(state)

    def offline_end_swimlane(self):
        # 钩子必须补到实际在跑的那一套算子上，否则 PTO_CSA_VARIANT 一切换就全落空。
        from vllm_ascend.ops.pypto.variant import variant_package
        CSAServiceRuntime = __import__(
            f"{variant_package()}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime

        if self._offline_swimlane is None:
            return {"dp_rank": self.vllm_config.parallel_config.data_parallel_rank,
                    "enabled": False, "captured": 0}
        state, original = self._offline_swimlane
        CSAServiceRuntime.__call__ = original
        self._offline_swimlane = None
        if state["captured"] != 1:
            raise RuntimeError(f"Expected exactly one DFX window, captured {state['captured']}")
        return dict(state)
