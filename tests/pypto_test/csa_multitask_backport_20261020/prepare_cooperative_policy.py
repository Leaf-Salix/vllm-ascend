# SPDX-License-Identifier: Apache-2.0
"""把本轮两种已测前段保留为资源分支，数学helper共用，基线分支保留24核Hadamard。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE, section
from prepare_fused_topk import add_calls, add_parameter

POLICY = """
@pl.jit.inline(auto_scope=False)
def _select_cooperative_prefix(
    tokens: pl.Scalar[pl.INDEX],
    maximum: pl.Scalar[pl.INT32],
    enabled: pl.constexpr,
):
    selected = pl.cast(0, pl.INT32)
    if enabled:
        if S == 6:
            if maximum // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:
                if tokens <= PREFILL_DENSE_TILE:
                    # 小M32区间里KV已满三组，或请求组数超过Score的24组核。
                    # 对应本轮受益的资源类型，不枚举B16/B28等测试点。
                    if tokens < KV_FIXED_SMALL_ROWS:
                        if tokens // KV_FIXED_SMALL_M_TILE >= KV_OM:
                            selected = pl.cast(1, pl.INT32)
                    if tokens // S > TOPK_SCORE_WORKERS:
                        selected = pl.cast(1, pl.INT32)
    return selected


"""


def function_source(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"需要唯一函数：{name}")
    node = nodes[0]
    begin = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "".join(source.splitlines(keepends=True)[begin - 1 : node.end_lineno])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="plan_workers_v1")
    parser.add_argument("--optimized", default="cooperative_q_only_sync_v1")
    parser.add_argument("--candidate", default="cooperative_policy_v1")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.optimized, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    baseline = HERE / "sources" / args.base
    optimized = HERE / "sources" / args.optimized
    target = HERE / "sources" / args.candidate
    shutil.copytree(baseline, target)

    # 只引入仍被调用的依赖适配；旧QR/QKV公开入口和签名保持。
    qkv = (optimized / "qkv_proj_rope.py").read_text()
    unused = section(
        qkv,
        "@pl.jit.inline(auto_scope=False)\ndef cooperative_q_proj_q(",
        "@pl.jit.inline(auto_scope=False)\ndef cooperative_q_dequant_after_score(",
    )
    qkv = qkv.replace(unused, "", 1)
    ast.parse(qkv)
    (target / "qkv_proj_rope.py").write_text(qkv)

    # 两条前段只在Hadamard分工上不同；保留各自已测数学体。
    path = target / "decode_indexer.py"
    indexer = path.read_text()
    donor_indexer = (optimized / "decode_indexer.py").read_text()
    hadamard = function_source(donor_indexer, "indexer_qr_hadamard_mm")
    hadamard = hadamard.replace("def indexer_qr_hadamard_mm(", "def indexer_qr_hadamard_cooperative(", 1)
    indexer += "\n\nQH_COOPERATIVE_WORKERS = 4\n\n" + hadamard
    ast.parse(indexer)
    path.write_text(indexer)

    # sync_start只随协同前段开启；普通路径继续使用原设置。
    path = target / "decode_sparse_attn_csa.py"
    attention = path.read_text()
    for name in ("sparse_attn_csa", "sparse_attn_csa_tp1"):
        attention = add_parameter(attention, name, "    group_sync_start: pl.constexpr = False,\n")
    before = 'with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid, rope_tid], allow_early_resolve=True)'
    if attention.count(before) != 1:
        raise ValueError("需要原独立QK/PV入口")
    attention = attention.replace(before, before[:-1] + ", sync_start=group_sync_start)", 1)
    attention = add_calls(attention, ("sparse_attn_csa",), ("group_sync_start",))
    ast.parse(attention)
    path.write_text(attention)

    path = target / "decode_csa.py"
    entry = path.read_text()
    donor_entry = (optimized / "decode_csa.py").read_text()
    imports = (
        "from .q_projection import QPROJ_PIPE_M_TILE, q_proj_q_matmul\n"
        "from .qkv_proj_rope import (\n"
        "    KV_FIXED_SMALL_ROWS, KV_FIXED_SMALL_M_TILE, KV_OM,\n"
        "    QPROJ_T_PAD, PREFILL_DENSE_TILE, rope_prepare, q_proj_qr_policy,\n"
        "    cooperative_q_dequant_after_score, kv_proj_rope,\n"
        ")\n"
        "from .decode_indexer import (\n"
        "    TP1_WEIGHTS_WORKERS, TOPK_SCORE_WORKERS, indexer_qr_rope,\n"
        "    indexer_qr_hadamard_cooperative, indexer_weights_score,\n"
        ")\n"
    )
    entry = entry.replace("# Dynamic shape variables.\n", imports + "\n# Dynamic shape variables.\n", 1)
    marker = "def _make_decode_csa_entry(*, host_metadata: bool):\n"
    entry = entry.replace(marker, POLICY + marker, 1)
    entry = entry.replace(
        "    resolve_max_seq_len = _use_host_max_seq_len if host_metadata else _read_device_max_seq_len\n",
        "    resolve_max_seq_len = _use_host_max_seq_len if host_metadata else _read_device_max_seq_len\n"
        "    enable_cooperative_prefix = not host_metadata\n",
        1,
    )
    # 直接在编排层分支，HC和Attention/O输出仍共用。
    original_front = section(
        entry,
        "            q, qa_tid = qkv_proj_rope_policy(",
        "\n            # sparse_attn_csa folds",
    )
    cooperative_front = section(
        donor_entry,
        "            # QA/QR先独立发出",
        "\n            # sparse_attn_csa folds",
    )
    cooperative_front = cooperative_front.replace("indexer_qr_hadamard_mm(", "indexer_qr_hadamard_cooperative(", 1)
    branching = (
        "            use_cooperative_prefix = _select_cooperative_prefix(\n"
        "                t_dim, max_seq_len, enable_cooperative_prefix,\n"
        "            )\n"
        "            stable_topk_flat = pl.reshape(idx_topk, [t_dim * IDX_TOPK])\n"
        "            if use_cooperative_prefix == 1:\n"
        + textwrap.indent(cooperative_front.rstrip(), "    ")
        + "\n            else:\n"
        + textwrap.indent(original_front.rstrip(), "    ")
        + "\n            index_indices = pl.reshape(stable_topk_flat, [t_dim, IDX_TOPK])\n"
    )
    entry = entry.replace(original_front, branching, 1)
    attn_call = section(
        entry,
        "            o_packed_heads, heads_dep = sparse_attn_csa_tp1(",
        "            with pl.scope():\n                layer_output",
    )
    true_call = textwrap.indent(add_calls(textwrap.dedent(attn_call), ("sparse_attn_csa_tp1",), ("True",)), " " * 12)
    false_call = textwrap.indent(add_calls(textwrap.dedent(attn_call), ("sparse_attn_csa_tp1",), ("False",)), " " * 12)
    entry = entry.replace(
        attn_call,
        "            if use_cooperative_prefix == 1:\n"
        + textwrap.indent(true_call.rstrip(), "    ")
        + "\n            else:\n"
        + textwrap.indent(false_call.rstrip(), "    ")
        + "\n",
        1,
    )
    ast.parse(entry)
    path.write_text(entry)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                optimized=args.optimized,
                donor="13259f92协同前段的已测多task适配；按本轮七档反馈选择资源分支",
                policy=(
                    "仅TMR/S6/长历史/T<=PREFILL_DENSE_TILE；"
                    "(T<KV_FIXED_SMALL_ROWS且T//KV_FIXED_SMALL_M_TILE>=KV_OM)"
                    "或T//S>TOPK_SCORE_WORKERS"
                ),
                optimized_cases=["128K/B16", "128K/B28"],
                fallback_cases=["128K/B4", "128K/B8", "128K/B24", "8K/B16", "8K/B32"],
                context_switch_history=8192,
                preserved="普通分支24核Hadamard及原前段次序；共享HC/数学helper/cache状态/输出；HBG关闭新策略",
                pending="必须测真实分支程序的完整CSA，不拼接独立候选结果认领收益",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
