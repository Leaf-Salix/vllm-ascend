# SPDX-License-Identifier: Apache-2.0
"""仅用于离线测试：在真实 worker 进程、图捕获之前设置确定性。"""

import os

import torch
import torch_npu

from vllm_ascend.worker.worker import NPUWorker


class OfflineNPUWorker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        level = vllm_config.additional_config["offline_deterministic_level"]
        if level not in (0, 1):
            raise ValueError("offline_deterministic_level must be 0 or 1")
        torch_npu.npu.set_deterministic_level(level)
        super().__init__(vllm_config, *args, **kwargs)
        self._offline_requested_deterministic_level = level
        print(f"OFFLINE_WORKER_DETERMINISTIC pid={os.getpid()} level={level}", flush=True)

    def offline_runtime_config(self):
        return {
            "requested_deterministic_level": self._offline_requested_deterministic_level,
            "deterministic_level": torch_npu.npu._get_deterministic_level(),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "hccl_deterministic": os.environ.get("HCCL_DETERMINISTIC", "false"),
            "dynamic_eplb": self.model_runner.dynamic_eplb,
        }
