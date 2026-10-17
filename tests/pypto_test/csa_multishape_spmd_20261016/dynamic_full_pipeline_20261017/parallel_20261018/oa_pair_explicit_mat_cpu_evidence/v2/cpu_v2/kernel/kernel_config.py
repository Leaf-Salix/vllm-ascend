# Kernel and Orchestration Configuration

from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer.
# AICPU thread count 0 selects the runtime's architecture default (a2a3: 4; a5: 5).
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 0,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "entry.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.OUT],
}

KERNELS = [
	{"func_id": 0, "name": "oa_pair_four_k256", "source": str(_ROOT_DIR / "kernels" / "aic" / "oa_pair_four_k256.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
]
