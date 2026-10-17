# SPDX-License-Identifier: Apache-2.0
"""从包内数学源码生成调用合同，不读取历史 catalog、trace 或 binding。

逻辑签名顺序用于 Host 绑定；物理参数按名称核对效果；真正生成的调用使用
kernels.py 含显式 worker 参数的顺序。这三种顺序不能混用。
"""

import ast
import collections
from pathlib import Path
from types import MappingProxyType

from .ir import ExplicitWorker, functions_from_ir

# 唯一无法从 Inline 装饰器恢复的信息是硬件角色，故保留这份小型声明。
# 新增或删除数学函数必须同时明确角色，不能把未知函数默认为 Vector。
_KERNEL_ROLES = MappingProxyType(
    {
        "hc_widen_rms": "aiv",
        "hc_pre_linear": "aic",
        "hc_pre_linear_reduce": "aiv",
        "split_pre_post": "aiv",
        "comb_sinkhorn": "aiv",
        "mix_x_rms_norm": "aiv",
        "csa_row_offsets": "aiv",
        "csa_rope_sign": "aiv",
        "q_rope_prepare": "aiv",
        "qr_proj_matmul": "aic",
        "qr_proj_matmul_0": "aic",
        "qr_rms_norm_quant": "aiv",
        "qproj_matmul": "aic",
        "qproj_dequant_rms_nope_rope": "aiv",
        "kv_proj_matmul": "aic",
        "kv_proj_matmul_0": "aic",
        "kv_rms_norm_rope": "aiv",
        "csa_cache_writeback": "aiv",
        "kv_score_proj": "aic",
        "scatter_softmax_pool": "aiv",
        "compress_state_commit": "aiv",
        "rmsnorm_rope_cache_write": "aiv",
        "kv_score_proj_0": "aic",
        "scatter_softmax_pool_0": "aiv",
        "compress_state_commit_0": "aiv",
        "indexer_boundary_init": "aiv",
        "rmsnorm_rope": "aiv",
        "kv_hadamard": "aic",
        "kv_and_cache_write": "aiv",
        "idx_kv_scale_commit": "aiv",
        "idx_qr_proj_matmul": "aic",
        "idx_qr_dequant_rope": "aiv",
        "qr_hadamard_matmul": "aic",
        "qr_hadamard_quant": "aiv",
        "weights_proj": "aic",
        "weights_proj_reduce": "aiv",
        "indexer_head_coefficients": "aiv",
        "indexer_score_topk_native_pair_aic": "aic",
        "indexer_score_topk_native_pair_aiv": "aiv",
        "indexer_head_coefficients_0": "aiv",
        "indexer_score_topk_native_pair_0_aic": "aic",
        "indexer_score_topk_native_pair_0_aiv": "aiv",
        "indexer_head_coefficients_1": "aiv",
        "indexer_score_topk_native_pair_1_aic": "aic",
        "indexer_score_topk_native_pair_1_aiv": "aiv",
        "indexer_head_coefficients_2": "aiv",
        "indexer_score_topk_native_pair_2_aic": "aic",
        "indexer_score_topk_native_pair_2_aiv": "aiv",
        "indexer_score_topk_leaf_aic": "aic",
        "indexer_score_topk_leaf_aiv": "aiv",
        "indexer_topk_single_leaf_publish": "aiv",
        "indexer_topk_query_merge": "aiv",
        "indexer_topk_query_merge__2": "aiv",
        "kv_touch": "aiv",
        "csa_slots_build_valid_qk_plan": "aiv",
        "rope_cs": "aiv",
        "qk_pv_aic": "aic",
        "qk_pv_aiv": "aiv",
        "proj_a_mm": "aic",
        "quant": "aiv",
        "_proj_b_mm_nz_kernel": "aic",
        "proj_b_act_hc_post": "aiv",
        "proj_a_mm_0": "aic",
        "quant_0": "aiv",
        "_proj_b_mm_nz_kernel__2": "aic",
        "proj_b_act_hc_post_0": "aiv",
        "proj_a_mm_1": "aic",
        "quant_1": "aiv",
        "proj_b_act_hc_post_1": "aiv",
        "proj_a_mm_2": "aic",
        "quant_2": "aiv",
        "_proj_b_mm_nz_kernel__3": "aic",
        "proj_b_act_hc_post_2": "aiv",
    }
)


def catalog():
    """每次从当前受维护源码生成独立合同，函数编号只在本包生成结果内使用。"""
    package = Path(__file__).resolve().parent
    body_path = package / "kernels.py"
    logical_path = package / "orchestration.py"
    _, physical = functions_from_ir(body_path)
    _, logical = functions_from_ir(logical_path)
    if physical.keys() != _KERNEL_ROLES.keys():
        missing = sorted(_KERNEL_ROLES.keys() - physical.keys())
        unknown = sorted(physical.keys() - _KERNEL_ROLES.keys())
        raise ValueError(f"物理数学体与角色声明不一致：缺少={missing}，未知={unknown}")
    workers = set(ExplicitWorker.REPLACEMENTS.values())
    kernels = []
    for func_id, (name, function) in enumerate(physical.items()):
        parent = name
        if parent not in logical and parent.endswith(("_aic", "_aiv")):
            parent = parent[:-4]
        if parent not in logical or parent == "_decode_csa_tp1_layer":
            raise ValueError(f"物理函数没有逻辑合同：{name}")
        parameters = []
        explicit_workers = []
        for parameter in function.args.args:
            annotation = ast.unparse(parameter.annotation)
            if parameter.arg in workers:
                if annotation != "pl.Scalar[pl.INDEX]":
                    raise ValueError(f"逻辑 worker 必须为 INDEX：{name}/{parameter.arg}")
                explicit_workers.append(parameter.arg)
            else:
                parameters.append({"name": parameter.arg, "annotation": annotation})
        # 精确四项白名单之外的参数全部保留，绝不按 original_* 前缀删除。
        uses = collections.Counter(
            node.id
            for statement in function.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id in explicit_workers
        )
        if set(uses) != set(explicit_workers):
            raise ValueError(f"显式 worker 没有在数学体中使用：{name}")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        system_operations = [
            {
                "operation": ast.unparse(call.func),
                "arguments": [ast.unparse(value) for value in call.args],
                "keywords": {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords},
                "source_line": call.lineno,
            }
            for call in calls
            if ast.unparse(call.func).startswith("pl.system.")
            or ast.unparse(call.func) in ("pl.tile.tpush_to_aiv", "pl.tile.tpop_from_aic")
        ]
        kernels.append(
            {
                "name": name,
                "core_type": _KERNEL_ROLES[name],
                "func_id": func_id,
                "logical_task": parent,
                "parameters": parameters,
                "return_annotation": ast.unparse(function.returns) if function.returns else None,
                "implicit_worker_replacements": dict(uses),
                "inline_parameter_order": [argument.arg for argument in function.args.args],
                "generated_line": function.lineno,
                "system_operations": system_operations,
            }
        )
    declared_logical = set(logical) - {"_decode_csa_tp1_layer"}
    used_logical = {kernel["logical_task"] for kernel in kernels}
    if declared_logical != used_logical:
        raise ValueError("逻辑调用合同与全部物理分支没有一一覆盖")
    return {
        "source_ir": str(body_path),
        "body_bundle": str(body_path),
        "logical_source_ir": str(logical_path),
        "physical_kernel_bodies": len(kernels),
        "logical_compute_bodies": len(used_logical),
        "all_generated_kernel_bodies_covered": True,
        "kernel_count_by_role": dict(collections.Counter(k["core_type"] for k in kernels)),
        "kernels": kernels,
    }


build_catalog = catalog
