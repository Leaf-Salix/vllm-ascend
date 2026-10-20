# SPDX-License-Identifier: Apache-2.0
"""从已提交源码冻结三方，并迁移13259f92的Score数学体；不修改正式算子。"""

import argparse
import ast
import io
import json
import shutil
import subprocess
import tarfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE = "beaefa6f6788a9448d49293238f6818c246f24f5"
SPMD = "13259f92"
OPS = "vllm_ascend/ops/pypto"


def freeze(revision, package, destination):
    destination.mkdir(parents=True, exist_ok=False)
    prefix = f"{OPS}/{package}/"
    archive = subprocess.check_output(["git", "archive", revision, prefix], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                relative = Path(member.name).relative_to(prefix)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.extractfile(member).read())


def section(source, start, stop):
    begin = source.index(start)
    return source[begin : source.index(stop, begin)]


def choose(original, replacement):
    """两个constexpr分支在原语句缩进下展开，保持原路径与MIX生命周期。"""
    indent = len(original.splitlines()[0]) - len(original.splitlines()[0].lstrip())
    prefix = " " * indent
    return (
        prefix
        + "if split_score_rows:\n"
        + textwrap.indent(textwrap.dedent(replacement), prefix + "    ")
        + prefix
        + "else:\n"
        + textwrap.indent(textwrap.dedent(original), prefix + "    ")
    )


def migrate_score(source, spmd_source):
    start = source.index("def indexer_score_topk_native_cube(")
    end = source.index("@pl.jit.inline(auto_scope=False)\ndef indexer_score_topk_forest", start)
    kernel = source[start:end]
    original_cube = section(kernel, "        for buf_item in pl.range(", "        for buf_score_lane in pl.split_aiv(")
    donor = section(
        spmd_source,
        "def _indexer_score_cube_split_store(",
        "@pl.jit.inline(auto_scope=False)\ndef indexer_score_cube_stage(",
    )
    kernel = kernel.replace(
        "    balance_leaves: pl.constexpr,\n",
        "    balance_leaves: pl.constexpr,\n    split_score_rows: pl.constexpr,\n",
        1,
    )
    changes = [
        (
            "                buf_query_left =",
            "                for buf_score_step",
            "            split_query0 =",
            "            # 双槽最多",
        ),
        (
            "                    buf_previous_l1 =",
            "                    buf_current_key =",
            "                split_previous0 =",
            "                buf_current_key =",
        ),
        (
            "                        buf_scores_l1 =",
            "                        if key_prefetch_panels > 0:\n                            if key_prefetch_to_l0:",
            "                    split_current0 =",
            "                    if key_prefetch_panels > 0:\n                        if key_prefetch_to_l0:",
        ),
        (
            "                    buf_last_pair =",
            "                    pl.system.sync_set(",
            "                split_right0_last =",
            "                pl.system.sync_set(",
        ),
    ]
    for old_start, old_end, new_start, new_end in changes:
        original = section(kernel, old_start, old_end)
        replacement = section(donor, new_start, new_end)
        # parser会先检查未采用分支的tile形状；半组大小对G2也须合法。
        # 实际启用仅长G6，此表达式仍严格等于已验证的192行。
        replacement = replacement.replace("SPLIT_QK_ROWS", "(query_group_size // 2 * IDX_N_HEADS)")
        kernel = kernel.replace(original, choose(original, replacement), 1)
    # SSA在折叠constexpr之前检查作用域；跨多个if保留Tile会形成FREE_VAR。
    # 两种完整Cube流水各自封闭，AIV仍是下面唯一的一份，不复制握手消费者。
    edited_cube = section(kernel, "        for buf_item in pl.range(", "        for buf_score_lane in pl.split_aiv(")

    class SelectSplit(ast.NodeTransformer):
        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == "split_score_rows":
                return [self.visit(statement) for statement in node.body]
            return self.generic_visit(node)

    split_tree = SelectSplit().visit(ast.parse(textwrap.dedent(edited_cube)))
    split_cube = ast.unparse(ast.fix_missing_locations(split_tree)) + "\n"
    kernel = kernel.replace(
        edited_cube,
        "        if split_score_rows:\n"
        + textwrap.indent(split_cube, " " * 12)
        + "        else:\n"
        + textwrap.indent(original_cube, " " * 4)
        + "\n",
        1,
    )
    kernel = kernel.replace(
        '    """同组query共用Key：双query/M128/N128或S6/M384/N64，保持FP16/Cube策略。"""',
        '    """同组query共用Key；长G6可用双M192早FIX，保持单MIX双槽READY/ACK。"""',
    )
    source = source[:start] + kernel + source[end:]
    source = source.replace(
        "NATIVE_QLI_WEIGHT_ROWS = 16\n",
        "NATIVE_QLI_WEIGHT_ROWS = 16\n"
        "# 13259f92：G6按head行拆为两半，WS仍累计完整六query后单次写回。\n"
        "SCORE_SPLIT_MIN_QUERY_ROWS = 32\n",
        1,
    )
    # T是device侧动态维度；小G6仍保留原M384，不用T_PAD容量代替有效T。
    forest = source.index("def indexer_score_topk_forest(")
    call = section(source[forest:], "                score_tid = indexer_score_topk_native_cube(", "            else:")
    optimized = call.replace(
        "                    True,\n                )",
        "                    True,\n                    True,\n                )",
        1,
    )
    if optimized == call:
        raise AssertionError("没有唯一定位长G6调用尾参")
    replacement = (
        "                if pl.tensor.dim(position_ids, 0) >= SCORE_SPLIT_MIN_QUERY_ROWS:\n"
        + textwrap.indent(optimized, "    ")
        + "                else:\n"
        + textwrap.indent(call, "    ")
    )
    source = source[:forest] + source[forest:].replace(call, replacement, 1)
    # PyPTO依赖调用不读取constexpr默认值；每个旧策略调用都显式传False。
    tree = ast.parse(source)
    count = len(
        next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_cube"
        ).args.args
    )
    lines = source.splitlines(keepends=True)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "indexer_score_topk_native_cube"
    ]
    for node in sorted(calls, key=lambda n: n.lineno, reverse=True):
        if len(node.args) == count - 1:
            closing = lines[node.end_lineno - 1]
            indent = len(closing) - len(closing.lstrip())
            lines.insert(node.end_lineno - 1, " " * (indent + 4) + "False,\n")
        elif len(node.args) != count:
            raise AssertionError("Score调用参数数目不匹配")
    source = "".join(lines)
    ast.parse(source)
    return source


def migrate_ob(source):
    """b26e8e86：仅NZ大M路径裁剪实际T以外的行；消费者本来按真实T读partial。"""
    for name in ("_proj_b_mm_nz_kernel", "_proj_b_mm_nz", "_proj_b_mm_nd"):
        start = source.index("def " + name + "(")
        end = source.index("\n):", start)
        signature = source[start:end]
        updated = signature.replace(
            "    ROW_TILE: pl.constexpr,",
            "    tokens: pl.Scalar[pl.INDEX],\n    ROW_TILE: pl.constexpr,",
        )
        if updated == signature:
            raise AssertionError("缺少O_B的ROW_TILE参数")
        source = source[:start] + updated + source[end:]
    old = (
        "                b_act_full = o_r_i8_pad[t0 : t0 + ROW_TILE, col_g : col_g + O_LORA]\n"
        "                full_acc = pl.matmul(b_act_full, weight_resident, out_dtype=pl.INT32)\n"
        "                partials[t0 : t0 + ROW_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = full_acc\n"
    )
    new = (
        "                # b26e8e86：只裁掉真实T之外的padding；T内dummy行仍计算。\n"
        "                active_rows = pl.min(ROW_TILE, tokens - t0)\n"
        "                b_act_full = pl.slice(\n"
        "                    o_r_i8_pad, [ROW_TILE, O_LORA], [t0, col_g],\n"
        "                    valid_shape=[active_rows, O_LORA],\n"
        "                )\n"
        "                full_acc = pl.matmul(b_act_full, weight_resident, out_dtype=pl.INT32)\n"
        "                # 尾部partial不再初始化；HC_post仅按真实T逐行消费。\n"
        "                partials = pl.assemble(partials, full_acc, [t0, g * D + n0])\n"
    )
    if source.count(old) != 1:
        raise AssertionError("NZ大M路径不唯一")
    source = (
        source.replace(old, new, 1)
        .replace(
            "_proj_b_mm_nz_kernel(o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, ROW_TILE)",
            "_proj_b_mm_nz_kernel(o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, tokens, ROW_TILE)",
        )
        .replace(
            "                o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, q_tid, ROW_TILE,",
            "                o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, q_tid, t_dim, ROW_TILE,",
        )
    )
    ast.parse(source)
    return source


def migrate_local_stats(source, donor, entry):
    """f222a917：同MIX、同AIV的三槽mi/li放入UB，跨核READY/ACK保持。"""
    for name, argument in (
        ("sparse_attn_csa", "    LOCAL_ATTENTION_STATS: pl.constexpr,\n"),
        ("sparse_attn_csa_tp1", "    max_seq_len: pl.Scalar[pl.INT32],\n"),
    ):
        start = source.index("def " + name + "(")
        end = source.index("\n)", start)
        source = source[:end] + "\n" + argument.rstrip("\n") + source[end:]
    source = source.replace(
        "            for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(",
        "            # 三槽交错存mi/li，仅当前AIV消费；跨query保留环形编号。\n"
        "            local_stats_ring = pl.create_tile([QK_TRANSFER_SLOTS * 2, H // 2], dtype=pl.FP32)\n"
        "            for qk_tick, (m_iter, l_iter, left_iter, right_iter, stats_cache_iter) in pl.range(",
        1,
    ).replace(
        "                init_values=(running_m, running_l, running_left, running_right),",
        "                init_values=(running_m, running_l, running_left, running_right, local_stats_ring),",
        1,
    )
    old_store = section(
        source,
        "                        pl.store(qk_mi,",
        "                if t_dim < EARLY_KV_MIN_TOKENS:",
    )
    new_store = section(
        donor, "                if LOCAL_ATTENTION_STATS:", "        if t_dim < EARLY_KV_MIN_TOKENS:"
    ).replace("EVENT_BASE + ", "")
    source = source.replace(old_store, textwrap.indent(new_store, " " * 8), 1)
    old_load = section(source, "                        pv_m = pl.load(", "                        next_m =")
    new_load = section(donor, "                # 同一 work 的 mi/li", "                next_m =")
    source = source.replace(old_load, textwrap.indent(new_load, " " * 8), 1)
    source = source.replace(
        "                running_m, running_l, running_left, running_right = pl.yield_(\n"
        "                    m_after, l_after, left_after, right_after\n",
        "                running_m, running_l, running_left, running_right, local_stats_ring = pl.yield_(\n"
        "                    m_after, l_after, left_after, right_after, stats_cache_ready\n",
        1,
    )
    call = section(source, "    packed, ready = sparse_attn_csa(", "    return packed, ready")
    optimized = call.replace("freqs_sin, o_packed_heads,", "freqs_sin, o_packed_heads, True,")
    fallback = call.replace("freqs_sin, o_packed_heads,", "freqs_sin, o_packed_heads, False,")
    source = source.replace(
        call,
        "    # 对齐单SPMD已验证长G6且24<=T<=96的资源区间；有效T仍来自Tensor。\n"
        "    t_dim = pl.tensor.dim(q, 0)\n"
        "    use_local_stats = pl.cast(0, pl.INT32)\n"
        "    if max_seq_len // COMPRESS_RATIO > 8192:\n"
        "        if t_dim >= 24:\n"
        "            if t_dim <= 96:\n"
        "                use_local_stats = pl.cast(1, pl.INT32)\n"
        "    if use_local_stats == 1:\n"
        + textwrap.indent(optimized, " " * 4)
        + "    else:\n"
        + textwrap.indent(fallback, " " * 4),
        1,
    )
    old_arg = "                o_packed_heads,\n            )"
    if entry.count(old_arg) != 1:
        raise AssertionError("完整CSA的Attention调用不唯一")
    entry = entry.replace(old_arg, "                o_packed_heads,\n                max_seq_len,\n            )", 1)
    ast.parse(source)
    ast.parse(entry)
    return source, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=("score_split", "ob_active", "score_split_ob", "local_stats", "score_stats_ob"),
        default="score_split",
    )
    args = parser.parse_args()
    sources = HERE / "sources"
    for revision, package, label in (
        (BASELINE, "deepseek_v4_flash_csa", "baseline"),
        (SPMD, "deepseek_v4_flash_csa_spmd", "spmd"),
    ):
        if not (sources / label).exists():
            freeze(revision, package, sources / label)
    # 每个私有包只创建一次；禁止覆盖排队/运行中JIT正在读取的包。
    destination = sources / args.candidate
    shutil.copytree(sources / "baseline", destination)
    changed = []
    if args.candidate.startswith("score_"):
        target = destination / "decode_indexer.py"
        target.write_text(migrate_score(target.read_text(), (sources / "spmd/indexer_score_stages.py").read_text()))
        changed.append(target.name)
    if args.candidate in ("ob_active", "score_split_ob", "score_stats_ob"):
        target = destination / "decode_o_proj.py"
        target.write_text(migrate_ob(target.read_text()))
        changed.append(target.name)
    if args.candidate in ("local_stats", "score_stats_ob"):
        target = destination / "decode_sparse_attn_csa.py"
        entry = destination / "decode_csa.py"
        rewritten, entry_source = migrate_local_stats(
            target.read_text(), (sources / "spmd/attention_stream_stages.py").read_text(), entry.read_text()
        )
        target.write_text(rewritten)
        entry.write_text(entry_source)
        changed.extend((target.name, entry.name))
    provenance = {
        "baseline_commit": BASELINE,
        "spmd_algorithm_commit": subprocess.check_output(["git", "rev-parse", SPMD], cwd=REPO, text=True).strip(),
        "cases": [[131072, b] for b in (4, 8, 16, 24, 28)] + [[8192, b] for b in (16, 32)],
        "query_tokens": 6,
        "goal": "多task完整CSA L1优于同轮正式单SPMD",
        "status": "source_prepared_no_device_evidence",
    }
    if not (HERE / "PROVENANCE.json").exists():
        (HERE / "PROVENANCE.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    (destination / "candidate.json").write_text(
        json.dumps(
            dict(candidate=args.candidate, changed_files=changed, **provenance),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(candidate=args.candidate, changed_files=changed), ensure_ascii=False))


if __name__ == "__main__":
    main()
