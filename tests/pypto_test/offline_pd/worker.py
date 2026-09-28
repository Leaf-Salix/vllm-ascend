# SPDX-License-Identifier: Apache-2.0
"""仅用于离线测试：在真实 worker 进程、图捕获之前设置确定性。"""

import os

import torch
import torch_npu

from offline_pd.event_mode import get_event_work_mode, set_event_work_mode
from vllm_ascend.worker.worker import NPUWorker


class OfflineNPUWorker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        level = vllm_config.additional_config["offline_deterministic_level"]
        if level not in (0, 1):
            raise ValueError("offline_deterministic_level must be 0 or 1")
        torch_npu.npu.set_deterministic_level(level)
        super().__init__(vllm_config, *args, **kwargs)
        self._offline_requested_deterministic_level = level
        self._offline_event_work_mode = vllm_config.additional_config.get("offline_event_work_mode")
        self._offline_moe_routing_tokens = vllm_config.additional_config.get("offline_moe_routing_tokens")
        print(f"OFFLINE_WORKER_DETERMINISTIC pid={os.getpid()} level={level}", flush=True)

    def init_device(self):
        result = super().init_device()
        if self._offline_event_work_mode is not None:
            set_event_work_mode(self._offline_event_work_mode)
            print(f"OFFLINE_CANN_EVENT_MODE pid={os.getpid()} mode={get_event_work_mode()}", flush=True)
        if self._offline_moe_routing_tokens is not None:
            from offline_pd.moe_routing import RoutingCapture

            self._offline_moe_routing = RoutingCapture(
                self._offline_moe_routing_tokens, self.vllm_config.model_config.hf_config.num_hidden_layers)
            self._offline_moe_routing.install()
        return result

    def offline_begin_moe_routing(self, *args):
        from offline_pd.moe_routing import begin

        return begin(self, *args)

    def offline_end_moe_routing(self):
        from offline_pd.moe_routing import end

        return end(self)

    def offline_batch_barrier(self):
        from vllm.distributed.parallel_state import get_dp_group

        # GroupCoordinator.barrier uses its CPU group. It runs once after
        # enqueue and before resuming scheduling, never in a timed forward.
        get_dp_group().barrier()

    def offline_runtime_config(self):
        from vllm_ascend.ascend_config import get_ascend_config

        ascend = get_ascend_config()
        engine = self.vllm_config
        return {
            "requested_deterministic_level": self._offline_requested_deterministic_level,
            "deterministic_level": torch_npu.npu._get_deterministic_level(),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "hccl_deterministic": os.environ.get("HCCL_DETERMINISTIC", "false"),
            "dynamic_eplb": self.model_runner.dynamic_eplb,
            "cann_event_work_mode": get_event_work_mode() if torch_npu.npu.is_initialized() else None,
            "decode_optimizations": {
                "ascend_compilation_config": {
                    name: getattr(ascend.ascend_compilation_config, name)
                    for name in ("enable_npugraph_ex", "enable_static_kernel", "fuse_norm_quant")
                },
                "enable_cpu_binding": ascend.enable_cpu_binding,
                "multistream_overlap_shared_expert": ascend.multistream_overlap_shared_expert,
                "recompute_scheduler_enable": ascend.scheduler_config.recompute_scheduler_enable,
            },
            "engine": {
                "async_scheduling": engine.scheduler_config.async_scheduling,
                "disable_hybrid_kv_cache_manager": engine.scheduler_config.disable_hybrid_kv_cache_manager,
                "enable_prefix_caching": engine.cache_config.enable_prefix_caching,
                "cudagraph_mode": str(engine.compilation_config.cudagraph_mode),
            },
            "runtime_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
                             "VLLM_BATCH_INVARIANT", "PYTORCH_NPU_ALLOC_CONF")
            },
            "scheduler": {
                name: getattr(self.model_runner.scheduler_config, name)
                for name in ("max_num_seqs", "max_num_batched_tokens", "max_num_scheduled_tokens")
            },
        }
