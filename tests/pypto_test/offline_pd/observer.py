# SPDX-License-Identifier: Apache-2.0
"""Control-only worker extension for D integration evidence, not timing."""

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
# 之后每次重放都沿用；而 forward hook 在图重放时根本不触发——
# offline_begin_observation 用 register_forward_hook 采到的 21 个层全是空字典
# （release_csa_batch_sweep_20260924 的 b1/b4/b8 即如此），且它静默返回空，
# 看起来像有数据，比没有更糟。
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

    def offline_begin_profile(self, directory, start_step, steps, expected_tokens, expected_requests):
        """从指定 step 起采集整步 CPU+NPU 数据：Level1、带 device kernel、不采 Python stack。

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
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(target),
        )
        runner = self.model_runner
        original = runner.execute_model
        state = {"dp_rank": rank, "trace_dir": target, "start_step": start_step,
                 "requested_steps": steps, "expected_tokens": expected_tokens,
                 "expected_requests": expected_requests, "seen_steady_steps": 0,
                 "profiled_steps": 0, "closed": False, "window": [], "observed": {}}

        def profiled(scheduler_output, *args, **kwargs):
            import torch

            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            # 把看到的 (tokens, requests) 分布记下来：判定失败时不必重跑就能诊断。
            key = f"{tokens}/{requests}"
            state["observed"][key] = state["observed"].get(key, 0) + 1
            # 不再用形状判定"稳态"，改为按 step 序号采样。
            # 实测（native_v3／pto_v3 的 observed）：batch 32 下从来没有 32 个请求
            # 同时在跑，而是 21 个请求 126 token 与 11 个请求 66 token 交替——
            # 32 条被调度器拆成了两批。所以"所有请求都在跑"这个条件不成立，
            # 上一版判据同样 0 次命中。
            # 两个后端的 observed 分布完全一致（23/22/1），说明调度行为与后端无关，
            # 因此按相同的 step 序号采样即可得到可比的窗口。每个被采样步的
            # (tokens, requests) 都记进 window，读者可自行核对可比性。
            steady = True
            index = state["seen_steady_steps"]
            if steady:
                state["seen_steady_steps"] += 1
            active = steady and index >= start_step and state["profiled_steps"] < steps
            if active and state["profiled_steps"] == 0:
                torch.npu.synchronize()
                profiler.start()
            try:
                return original(scheduler_output, *args, **kwargs)
            finally:
                if active:
                    state["profiled_steps"] += 1
                    state["window"].append({"steady_step_index": index, "scheduled_tokens": tokens,
                                            "requests": requests})
                    if state["profiled_steps"] == steps:
                        torch.npu.synchronize()
                        profiler.stop()
                        state["closed"] = True

        runner.execute_model = profiled
        self._offline_profile = (state, profiler, original)
        return {"dp_rank": rank, "trace_dir": target}

    def offline_end_profile(self):
        import torch

        state, profiler, original = self._offline_profile
        self.model_runner.execute_model = original
        self._offline_profile = None
        if state["profiled_steps"] and not state["closed"]:
            torch.npu.synchronize()
            profiler.stop()
            state["closed"] = True
        # 采不到足够样本时**不要抛异常**：异常会让整个 rank 的 json 落不了盘，
        # 连同 observed 里的诊断信息一起丢失——T2.1 前两轮就是这样，失败了却
        # 拿不到"实际每步调度了多少 token"，只能另想办法查。这里如实记进报告，
        # 由调用方按 profiled_steps 判断是否可用。
        state["sufficient"] = state["profiled_steps"] == state["requested_steps"]
        return state

    def offline_begin_bitcompare(self, layer_index, expected_tokens, max_samples, mode):
        """在真实生产路径上逐 bit 比对 PTO 与 Native 的 CSA 层输出张量。

        挂点选 CSAServiceRuntime.__call__ 而不是离线 fixture：那套 fixture
        （dsv4_csa_native_forward.py）依赖当前 release 已删除的
        DeviceMetadataExecutor，重建它等于复活旧接口；而这里比的是线上真实张量，
        权重、缓存、metadata 全都一致，不存在人造输入的偏差。

        同一步里先跑 PTO 再跑 Native，两者都写同一批 cache 槽位。这是安全的：
        compress_state 等状态是按计算出的行列直接赋值而非累加，重复写幂等；
        Native 在读之前先写当前 token 的槽位，所以它的结果不受 PTO 先写的影响。
        比完把 output 留成 Native 的值，让本轮继续沿参考轨迹走，避免差异累积。
        """
        import torch

        from vllm_ascend.ops.dsa import dsa_forward
        from vllm_ascend.ops.pypto.variant import variant_package

        CSAServiceRuntime = __import__(
            f"{variant_package()}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime

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
            result = original(runtime, context, hidden, positions, output, kv_cache, *args, **kwargs)
            pto = output.detach().clone()
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
            equal = bool(torch.equal(pto, native))
            record = {"step": state["seen"], "shape": list(pto.shape), "dtype": str(pto.dtype),
                      "mode": mode, "bit_equal": equal}
            if not equal:
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
        state["sufficient"] = bool(samples)
        return state

    def offline_begin_steady(self, warmup_steps):
        """测稳态每步耗时与峰值显存，不加任何额外同步，避免测量本身改变被测对象。

        与 profile 窗口的区别：profile 为了让设备任务完整落在窗口内，两端各做一次
        synchronize，那对结构对照没问题，但会把同步开销算进耗时，不能用来报稳态性能。
        这里只在 execute_model 两侧取 perf_counter，前 warmup_steps 步丢弃，用来排除
        加载、首次编译与首个恢复步骤。

        口径限制要随数据一起报：不加同步意味着某一步的耗时里可能含等待上一步设备任务
        完成的时间。稳态下主机本就被设备拖住，总和与吞吐是准的，单步 p50/p95 只是近似。
        """
        import torch

        if getattr(self, "_offline_steady", None) is not None:
            raise RuntimeError("Steady measurement is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        runner = self.model_runner
        original = runner.execute_model
        state = {"dp_rank": rank, "warmup_steps": warmup_steps, "seen_steps": 0,
                 "step_seconds": [], "step_tokens": [], "step_requests": []}
        # 峰值统计从窗口开始处重新计数，否则读到的是加载与预热留下的高水位。
        torch.npu.reset_peak_memory_stats()

        def timed(scheduler_output, *args, **kwargs):
            index = state["seen_steps"]
            state["seen_steps"] += 1
            if index < warmup_steps:
                return original(scheduler_output, *args, **kwargs)
            start = time.perf_counter()
            try:
                return original(scheduler_output, *args, **kwargs)
            finally:
                state["step_seconds"].append(time.perf_counter() - start)
                state["step_tokens"].append(scheduler_output.total_num_scheduled_tokens)
                state["step_requests"].append(len(scheduler_output.num_scheduled_tokens))

        runner.execute_model = timed
        self._offline_steady = (state, original)
        return {"dp_rank": rank, "warmup_steps": warmup_steps}

    def offline_end_steady(self):
        import torch

        state, original = self._offline_steady
        self.model_runner.execute_model = original
        self._offline_steady = None
        samples = state["step_seconds"]
        state["measured_steps"] = len(samples)
        state["peak_allocated_bytes"] = int(torch.npu.max_memory_allocated())
        state["peak_reserved_bytes"] = int(torch.npu.max_memory_reserved())
        if samples:
            ordered = sorted(samples)
            def at(q):
                # 取最近秩次，样本少时不做插值，免得报出没测到的数。
                return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]
            state["p50_seconds"], state["p95_seconds"] = at(0.50), at(0.95)
            state["mean_seconds"] = sum(samples) / len(samples)
            state["total_seconds"] = sum(samples)
            state["total_tokens"] = sum(state["step_tokens"])
            state["tokens_per_second"] = state["total_tokens"] / state["total_seconds"]
        # 样本不足如实记录，由调用方判断可用性，不在 worker 里抛异常丢掉整份报告。
        state["sufficient"] = state["measured_steps"] >= 20
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

        必须取**同一层**的 impl：compress_ratio 不匹配会直接报错
        （task_20260924_000518 即因此失败）。这里按 builder 的 compressor_ratio
        找到对应层。
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
        import json
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
        """把一次真实 CSA 调用的全部根入参落盘，供单卡回放。

        目的是把"改一次 kernel 就要起 16 卡整模型"这条链切断：dump 一次之后，
        `dsv4_csa_single_card_bench.py` 就能在单卡上反复回放同一组真实输入，
        既能立刻暴露 aicore 故障，也能采泳道量单次耗时。
        """
        from vllm_ascend.ops.pypto.variant import variant_package
        package = variant_package()
        CSAServiceRuntime = __import__(f"{package}.service", fromlist=["CSAServiceRuntime"]).CSAServiceRuntime
        native_adapter = __import__(f"{package}.native_adapter", fromlist=["NativeCSACall"])
        NativeCSACall = native_adapter.NativeCSACall

        if getattr(self, "_offline_argdump", None) is not None:
            raise RuntimeError("Argument dump is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        attention = self.model_runner.get_model().model.layers[layer_index].self_attn
        wanted = getattr(attention.dsa_attn, "_pto_csa_runtime", None)
        if wanted is None:
            raise ValueError(f"Layer {layer_index} has no PTO CSA runtime")
        state = {"dp_rank": rank, "layer_index": layer_index, "layer_name": wanted.layer_name,
                 "expected_tokens": expected_tokens, "dumped": 0, "path": None}
        # 只 dump rank0：16 个 rank 的输入形状一致，多存只是浪费磁盘。
        state["enabled"] = rank == 0
        original_call = CSAServiceRuntime.__call__
        original_init = NativeCSACall.__init__
        latest = {}

        def traced_init(call, *args, **kwargs):
            original_init(call, *args, **kwargs)
            latest["call"] = call

        def traced(runtime, context, hidden, *args, **kwargs):
            result = original_call(runtime, context, hidden, *args, **kwargs)
            # 诊断：落盘条件不满足时，这张直方图能说明到底差在哪一项
            # （是 runtime 不匹配、还是 token 数对不上）。
            seen = state.setdefault("seen", {})
            key = f"shape={tuple(hidden.shape)} same_runtime={runtime is wanted}"
            seen[key] = seen.get(key, 0) + 1
            if (state["enabled"] and runtime is wanted and not state["dumped"]
                    and hidden.shape[0] == expected_tokens):
                call = latest.get("call")
                if call is not None:
                    import torch
                    target = Path(out_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    # 整层融合（1dcadd85）后入口改名，旧名保留成回退。
                    _entry = "decode_csa_tp1_layer_test"
                    _mod = __import__(f"{package}.decode_csa", fromlist=[_entry])
                    kernel = getattr(_mod, _entry, None) or _mod.decode_csa_tp1_attention_test
                    names = list(kernel.param_names)
                    payload = {}
                    for name in names:
                        value = call.args[name]
                        if not isinstance(value, torch.Tensor):
                            raise TypeError(f"CSA argument {name} is not a tensor: {type(value)}")
                        # 输出缓冲也一并存：回放要的是完全相同的形状与 dtype。
                        # 用 torch.save 而不是 np.savez——numpy 不认 bfloat16。
                        payload[name] = value.detach().cpu().contiguous().clone()
                    torch.save(payload, target / "csa_args.pt")
                    # 记录 indexer 的 key/scale 原始视图元数据：页跨度到底是 4096
                    # （键整段连续）还是 4160（逐页与 scale 交错），静态读代码读不出来。
                    key_view, scale_view = call.views["indexer"]
                    storage_meta = {
                        name: {"shape": list(t.shape), "strides": list(t.stride()),
                               "storage_offset": t.storage_offset(), "dtype": str(t.dtype),
                               "data_ptr": t.data_ptr(),
                               "storage_data_ptr": t.untyped_storage().data_ptr(),
                               "storage_nbytes": t.untyped_storage().nbytes(),
                               "is_contiguous": bool(t.is_contiguous())}
                        for name, t in (("indexer_key", key_view), ("indexer_scale", scale_view))
                    }
                    meta = {"layer_index": layer_index, "layer_name": wanted.layer_name,
                            "indexer_storage": storage_meta,
                            "tokens": int(hidden.shape[0]), "param_names": names,
                            "dtypes": {k: str(call.args[k].dtype) for k in names},
                            "shapes": {k: list(call.args[k].shape) for k in names}}
                    (target / "csa_args_meta.json").write_text(json.dumps(meta, indent=2))
                    state["dumped"] += 1
                    state["path"] = str(target / "csa_args.pt")
            return result

        CSAServiceRuntime.__call__ = traced
        NativeCSACall.__init__ = traced_init
        self._offline_argdump = (state, original_call, original_init, CSAServiceRuntime, NativeCSACall)
        return dict(state)

    def offline_end_argdump(self):
        entry = getattr(self, "_offline_argdump", None)
        if entry is None:
            raise RuntimeError("Argument dump was not started")
        state, original_call, original_init, CSAServiceRuntime, NativeCSACall = entry
        CSAServiceRuntime.__call__ = original_call
        NativeCSACall.__init__ = original_init
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
