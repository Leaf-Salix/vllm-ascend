"""Compile logical contiguous cache ABI with an existing CPU fixture."""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo / "tests/pypto_test"))
from dsv4_csa_env import activate

activate()
import torch
from pypto.runtime import RunConfig
from pypto.runtime.kernel_compiler import KernelCompiler
from dsv4_csa_replay import materialize
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa import decode_csa_tp1_layer_test

case = repo / "tests/pypto_test/results/csa_baseline_20260926/nz_native_b16_timing/mode2/case"
meta = json.loads((case / "csa_args_meta.json").read_text())
blob = torch.load(case / "csa_args.pt", map_location="cpu", weights_only=True)
tensors, backings = materialize(meta, blob, "cpu")
raw = tensors["idx_kv_cache"]
pages = raw.shape[0]
tensors["idx_kv_cache"] = raw[:, :4096].contiguous().view(pages, 32, 1, 128)
tensors["idx_kv_scale"] = raw[:, 4096:4160].contiguous().view(torch.float16).view(pages, 32, 1, 1)
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.indexer_cache import SplitIndexerCache
cache = SplitIndexerCache(tensors["idx_kv_cache"], tensors["idx_kv_scale"],
                          batch_capacity=tensors["kv_seq_lens"].numel(),
                          table_columns=tensors["idx_block_table"].shape[1])
cache.load(tensors["idx_block_table"], tensors["kv_seq_lens"])
tensors["idx_kv_cache"], tensors["idx_kv_scale"] = cache.views(tensors["kv_seq_lens"].numel())
kernel = decode_csa_tp1_layer_test
assert set(kernel.param_names) == set(tensors)
out = Path(sys.argv[1]).resolve()
# 与torch注册入口一致地检查KernelArtifact ABI，不初始化设备。
kernel._resolve_compiled(
    tuple(tensors[name] for name in kernel.param_names),
    {"config": RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(out))},
    _kernel=True,
)
# 继续编译生成的AICPU调度源码，捕获跨scope别名等纯CPU可判定的问题。
compiler = KernelCompiler(platform="a2a3")
sources = list((out / "orchestration").glob("*.cpp"))
if not sources:
    raise RuntimeError(f"未生成调度C++: {out}")
for source in sources:
    compiler.compile_orchestration("tensormap_and_ringbuffer", str(source))
print("SPLIT_CACHE_CPU_COMPILE_OK", out)
