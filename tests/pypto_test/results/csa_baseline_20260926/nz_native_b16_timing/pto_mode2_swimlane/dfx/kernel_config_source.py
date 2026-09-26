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
	"source": str(_ROOT_DIR / "orchestration" / "_decode_csa_tp1_layer.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT, _D.OUT],
}

KERNELS = [
	{"func_id": 0, "name": "hc_widen", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_widen.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.OUT, _D.IN]},
	{"func_id": 1, "name": "hc_pre_rms", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_rms.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 2, "name": "hc_pre_linear", "source": str(_ROOT_DIR / "kernels" / "aic" / "hc_pre_linear.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 3, "name": "hc_pre_linear_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_linear_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 4, "name": "split_pre_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "split_pre_post.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.INOUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 5, "name": "comb_sinkhorn", "source": str(_ROOT_DIR / "kernels" / "aiv" / "comb_sinkhorn.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT]},
	{"func_id": 6, "name": "mix_x_rms_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "mix_x_rms_norm.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.IN, _D.OUT, _D.INOUT, _D.IN]},
	{"func_id": 7, "name": "csa_row_offsets", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_row_offsets.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 8, "name": "csa_rope_sign", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_rope_sign.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 9, "name": "q_rope_prepare", "source": str(_ROOT_DIR / "kernels" / "aiv" / "q_rope_prepare.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN]},
	{"func_id": 10, "name": "qr_proj_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_proj_seed.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 11, "name": "qr_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qr_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 12, "name": "qr_rms_norm_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rms_norm_quant.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN]},
	{"func_id": 13, "name": "qproj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qproj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 14, "name": "qproj_dequant_rms_nope_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qproj_dequant_rms_nope_rope.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 15, "name": "kv_proj_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_proj_seed.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 16, "name": "kv_proj_native_240", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_proj_native_240.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 17, "name": "kv_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 18, "name": "kv_rms_norm_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_rms_norm_rope.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 19, "name": "csa_cache_writeback", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_cache_writeback.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 20, "name": "kv_score_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_score_proj.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 21, "name": "scatter_softmax_pool", "source": str(_ROOT_DIR / "kernels" / "aiv" / "scatter_softmax_pool.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 22, "name": "compress_state_commit", "source": str(_ROOT_DIR / "kernels" / "aiv" / "compress_state_commit.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 23, "name": "rmsnorm_rope_cache_write", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rmsnorm_rope_cache_write.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.OUT, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 24, "name": "kv_score_proj_0", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_score_proj_0.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 25, "name": "scatter_softmax_pool_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "scatter_softmax_pool_0.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 26, "name": "compress_state_commit_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "compress_state_commit_0.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 27, "name": "indexer_boundary_init", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_boundary_init.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 28, "name": "rmsnorm_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rmsnorm_rope.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 29, "name": "kv_hadamard", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_hadamard.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 30, "name": "kv_and_cache_write", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_and_cache_write.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.INOUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 31, "name": "idx_kv_scale_commit", "source": str(_ROOT_DIR / "kernels" / "aiv" / "idx_kv_scale_commit.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN]},
	{"func_id": 32, "name": "idx_qr_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "idx_qr_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 33, "name": "idx_qr_dequant_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "idx_qr_dequant_rope.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 34, "name": "qr_hadamard_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qr_hadamard_matmul.cpp"), "core_type": "aic", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 35, "name": "qr_hadamard_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_hadamard_quant.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN]},
	{"func_id": 36, "name": "weights_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "weights_proj.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 37, "name": "weights_proj_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "weights_proj_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 38, "name": "indexer_key_repack", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_key_repack.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 39, "name": "indexer_score_topk_leaf_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "indexer_score_topk_leaf_aic.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 40, "name": "indexer_score_topk_leaf_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_score_topk_leaf_aiv.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 41, "name": "indexer_topk_single_leaf_publish", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_topk_single_leaf_publish.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 42, "name": "indexer_topk_query_merge", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_topk_query_merge.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.OUT, _D.OUT]},
	{"func_id": 43, "name": "kv_touch", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_touch.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 44, "name": "csa_slots_build_valid_qk_plan", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_slots_build_valid_qk_plan.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 45, "name": "qk_pv_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "qk_pv_aic.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 46, "name": "qk_pv_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_pv_aiv.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 47, "name": "rope_cs", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_cs.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 48, "name": "merge_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "merge_norm.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 49, "name": "proj_a_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "proj_a_mm.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 50, "name": "quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "quant.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN]},
	{"func_id": 51, "name": "_proj_b_mm_nz_kernel", "source": str(_ROOT_DIR / "kernels" / "aic" / "_proj_b_mm_nz_kernel.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 52, "name": "proj_b_act", "source": str(_ROOT_DIR / "kernels" / "aiv" / "proj_b_act.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 53, "name": "hc_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_post.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
]
