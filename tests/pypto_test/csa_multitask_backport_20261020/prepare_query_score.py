# SPDX-License-Identifier: Apache-2.0
"""将已验证Q Vector尾段接入长G6 Score MIX，保留其它多task阶段。"""

import argparse
import ast
import json
import shutil
import subprocess

from prepare import HERE, REPO
from prepare_query_ready import replace_once

FIELDS = (
    "q_stage_weight_scale",
    "q_stage_cos",
    "q_stage_sin",
    "q_stage_swap",
    "q_stage_out",
    "q_stage_scale",
    "q_stage_projected",
)


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


def excerpt(source, node):
    start = min(node.lineno, *(d.lineno for d in node.decorator_list))
    return "".join(source.splitlines(keepends=True)[start - 1 : node.end_lineno])


def append_parameters(source, name, parameters):
    node = function(source, name)
    lines = source.splitlines(keepends=True)
    close = next(i for i in range(node.lineno, node.end_lineno) if lines[i].startswith(")"))
    lines.insert(close, "".join("    " + p + ",\n" for p in parameters))
    return "".join(lines)


def append_call_arguments(source, names, arguments, *, choose_fusion=False):
    calls = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
    ]
    if not calls:
        raise ValueError(f"没有找到调用：{names}")
    lines = source.splitlines(keepends=True)
    fused = 0
    for node in sorted(calls, key=lambda n: n.end_lineno, reverse=True):
        args = list(arguments)
        if choose_fusion:
            if len(node.args) != 23 or node.keywords:
                raise ValueError("Score原参数合同不符")
            enabled = ast.literal_eval(node.args[20])
            if enabled and not ast.literal_eval(node.args[19]):
                raise ValueError("仅允许balanced长G6拆M融合Q")
            args.append(str(enabled))
            fused += enabled
        close = lines[node.end_lineno - 1]
        if close.strip() != ")":
            raise ValueError("调用需有独立闭括号行")
        indent = len(close) - len(close.lstrip()) + 4
        lines.insert(node.end_lineno - 1, "".join(" " * indent + a + ",\n" for a in args))
    if choose_fusion and (len(calls) != 6 or fused != 1):
        raise ValueError("只改六个Score调用中的一个长G6拆M分支")
    return "".join(lines)


def make_q_stage(donor):
    first = min(n.lineno for n in ast.parse(donor).body if isinstance(n, ast.FunctionDef)) - 1
    # 首个函数前还有decorator；按其实际行号截断。
    first_node = next(n for n in ast.parse(donor).body if isinstance(n, ast.FunctionDef))
    first = min(first, *(d.lineno - 1 for d in first_node.decorator_list))
    prefix = "".join(donor.splitlines(keepends=True)[:first])
    prefix = prefix.replace("from ..deepseek_v4_flash_csa.config import FLASH as M", "from .config import FLASH as M")
    doc = ast.parse(prefix).body[0]
    lines = prefix.splitlines(keepends=True)
    lines[doc.lineno - 1 : doc.end_lineno] = ['"""7061c112的Q反量化/RMS/RoPE数学原文；不创建独立SPMD或新信号。"""\n']
    return "".join(lines) + "\n" + excerpt(donor, function(donor, "q_dequant_rope_stage")) + "\n"


def make_q_prepare(source):
    node = function(source, "qkv_proj_rope_policy")
    lines = source.splitlines(keepends=True)
    start = min(node.lineno, *(d.lineno for d in node.decorator_list)) - 1
    close = next(i for i in range(node.lineno, node.end_lineno) if lines[i].startswith("):"))
    header = "".join(lines[start:close]).replace("def qkv_proj_rope_policy(", "def qkv_prepare_query_score(")
    header += "    defer_query: pl.Scalar[pl.INT32],\n):\n"
    body = '''
    """当前有界decode的一份Q_B缓冲；长G6由Score AIV消费，其他路径原地完成Q。"""
    t_dim = pl.tensor.dim(x, 0)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, q_rope_sin_signed, q_rope_swap_idx)
    qr_i8_matmul = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    qr_scale_pad_store = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
    qa_tid = q_proj_qr_policy(
        x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store, qa_m16_parallel,
    )
    # 保留原QB及其既有依赖标记，没有新增dummy。
    q_seq_dep = pl.system.task_dummy(deps=[])
    qproj_rows = (t_dim + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE * QPROJ_PIPE_M_TILE
    q_projected = pl.create_tensor([qproj_rows, H * HEAD_DIM], dtype=pl.INT32)
    q_projected, qb_tid = q_proj_q_matmul(wq_b, qr_i8_matmul, q_projected, t_dim, q_seq_dep)
    if defer_query == 0:
        q_proj_q_dequant(
            wq_b_scale, rope_cos, q_rope_sin_signed, q_rope_swap_idx,
            q, qr_scale_pad_store, q_projected, 0, t_dim,
        )
    kv_proj_rope(x, wkv, gamma_ckv, rope_cos, q_rope_sin_signed, q_rope_swap_idx, kv, late_dep)
    return q, qa_tid, qr_scale_pad_store, q_projected, q_rope_sin_signed, q_rope_swap_idx
'''
    return source + "\n\n" + header + body.lstrip("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="query_score_v1")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    donor = subprocess.check_output(
        ["git", "show", "7061c112:vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py"],
        cwd=REPO,
        text=True,
    )
    stage = make_q_stage(donor)
    if ast.dump(function(stage, "q_dequant_rope_stage")) != ast.dump(function(donor, "q_dequant_rope_stage")):
        raise ValueError("不得修改来源Q数学")
    qkv = make_q_prepare((base / "qkv_proj_rope.py").read_text())
    indexer = (base / "decode_indexer.py").read_text()
    indexer = replace_once(
        indexer,
        "from .topk_threshold_stages import threshold_merge_top512\n",
        "from .query_score_stages import q_dequant_rope_stage\n"
        "from .topk_threshold_stages import threshold_merge_top512\n",
    )
    for name in ("indexer_score_topk_native_cube", "indexer_score_topk_forest", "indexer_weights_score", "indexer"):
        params = [n + ": pl.Tensor" for n in FIELDS]
        if name == "indexer_score_topk_native_cube":
            params.append("fuse_query: pl.constexpr")
        indexer = append_parameters(indexer, name, params)
    indexer = append_call_arguments(indexer, {"indexer_weights_score", "indexer_score_topk_forest"}, FIELDS)
    indexer = append_call_arguments(indexer, {"indexer_score_topk_native_cube"}, FIELDS, choose_fusion=True)
    begin = "        for buf_score_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):\n"
    fused = (
        begin
        + "            if fuse_query:\n"
        + "                # QB/scale/RoPE的Tensor读取提供真实数据依赖；AIV局部先Q后Score。\n"
        + "                q_stage_out = q_dequant_rope_stage(\n"
        + "".join("                    " + field + ",\n" for field in FIELDS)
        + "                    buf_worker * 2 + buf_score_lane, 2 * TOPK_SCORE_WORKERS,\n"
        + "                )\n"
    )
    indexer = replace_once(indexer, begin, fused)
    entry = (base / "decode_csa.py").read_text()
    entry = replace_once(
        entry,
        "from .decode_indexer import LONG_S6_MIN_QUERY_ROWS,",
        "from .decode_indexer import SCORE_SPLIT_MIN_QUERY_ROWS, LONG_S6_MIN_QUERY_ROWS,",
    )
    entry = replace_once(
        entry,
        "from .qkv_proj_rope import QR_M_TILE, qkv_proj_rope_policy\n",
        "from .qkv_proj_rope import QR_M_TILE, PREFILL_DENSE_TILE, qkv_prepare_query_score\n",
    )
    entry = replace_once(
        entry,
        "\nT = B * S\n",
        "\nT = B * S\nassert T_PAD <= PREFILL_DENSE_TILE, 'Q/Score候选只支持原有界单dense tile decode'\n",
    )
    entry = replace_once(
        entry,
        "            q, qa_tid = qkv_proj_rope_policy(\n",
        "            defer_q_to_score = pl.cast(0, pl.INT32)\n"
        "            if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
        "                if t_dim >= SCORE_SPLIT_MIN_QUERY_ROWS:\n"
        "                    defer_q_to_score = pl.cast(1, pl.INT32)\n"
        "            q, qa_tid, fused_q_scale, fused_q_projected, "
        "fused_q_sin, fused_q_swap = qkv_prepare_query_score(\n",
    )
    entry = append_call_arguments(entry, {"qkv_prepare_query_score"}, ["defer_q_to_score"])
    entry = replace_once(
        entry,
        "            ori_block_num = pl.tensor.dim(kv_cache, 0)\n",
        "            fused_q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])\n"
        "            ori_block_num = pl.tensor.dim(kv_cache, 0)\n",
    )
    entry = append_call_arguments(
        entry,
        {"indexer"},
        [
            "wq_b_scale",
            "freqs_cos",
            "fused_q_sin",
            "fused_q_swap",
            "fused_q_flat",
            "fused_q_scale",
            "fused_q_projected",
        ],
    )
    files = {
        "decode_csa.py": entry,
        "decode_indexer.py": indexer,
        "qkv_proj_rope.py": qkv,
        "query_score_stages.py": stage,
    }
    for value in files.values():
        ast.parse(value)
    shutil.copytree(base, target)
    for name, value in files.items():
        (target / name).write_text(value)
    manifest = dict(
        candidate=args.candidate,
        base=args.base,
        donor="7061c112/13259f92",
        changed_files=list(files),
        mechanism="长G6拆M路径把Q Vector尾段接到现有Score MIX AIV前段；QB、KV、Compressor、TopK、Attention及输出仍独立",
        numerical_source="q_dequant_rope_stage的完整函数AST与已提交来源相同；QR/QB/KV复用现有数学",
        lifetime="Q_B输出与scale/RoPE缓冲提升到Score消费者可见范围，真实Tensor依赖必须另查",
        scope="T<=512的已有S6有界decode；非prefill泛化，HBG未测",
        caveats=[
            "Score AIV含Q尾段，不能和旧Score按等工作量相减",
            "QB可能成为整个MIX的前置，比来源的AIV局部等待更粗",
            "CORE_ARGS与UB峰值待CPU验证",
        ],
        context_switch_history=32765,
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
