"""冻结当前算子，组合Native的query分工、连续S2/UB排序和单根发布。"""

import ast
import difflib
import json
import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-hc-pre-priority-d93bba14-baseline"
PREFIX = WORKSPACE / ".cache/csa-score-single-root-d93bba14"
OLD_PACKAGE = "dsv4_csa_hc_pre_priority_d93bba14"
PACKAGE = "dsv4_csa_score_single_root_d93bba14"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"唯一替换锚点失效：{old[:100]}")
    return source.replace(old, new)


def transform(before):
    # Retain the previous helper for short/dual-query branches. Derive the
    # same tile sorting sizes and within-segment tie rule without GM staging.
    start = before.index("@pl.jit.inline\ndef indexer_topk_segment_pairs(")
    stop = before.index("@pl.jit.inline\ndef indexer_long_leaf_plan(", start)
    helper = before[start:stop]
    for label, size in (("tiny", 512), ("small", 1024), ("large", 2048)):
        old = (f"        {label}_raw = pl.load(score_arena, [score_row, score_offset], "
               f"[1, {size}], valid_shape=[1, valid_count])\n")
        new = (f"        {label}_tile = pl.tile.extract(scores, 0, 0, [1, {size}], "
               "target_memory=pl.MemorySpace.Vec)\n"
               f"        {label}_raw = pl.set_validshape({label}_tile, 1, valid_count)\n")
        helper = once(helper, old, new)
    # This JIT branch strips Tile parameter annotations while discovering an
    # inline dependency. Embed the existing small sorting body in the kernel;
    # no toolchain patch or extra task is needed for a UB-only argument.
    sort_body = textwrap.dedent(helper[helper.index("    logical_i32 ="):helper.index("    return root")])
    for old, new in (("scores", "single_segment_data"), ("logical_begin", "single_segment_begin"),
                     ("valid_count", "single_segment_valid"), ("root", "single_new_root")):
        sort_body = re.sub(r"\b" + old + r"\b", new, sort_body)
    wrapped = []
    for line in sort_body.splitlines():
        if len(line) <= 75:
            wrapped.append(line)
            continue
        assignment = ast.parse(line.strip()).body[0]
        if not isinstance(assignment, ast.Assign) or not isinstance(assignment.value, ast.Call):
            raise ValueError("排序模板出现非预期长表达式")
        call = assignment.value
        indent = line[:len(line) - len(line.lstrip())]
        wrapped.append(indent + ast.unparse(assignment.targets[0]) + " = " + ast.unparse(call.func) + "(")
        args = [ast.unparse(a) for a in call.args]
        args += [kw.arg + "=" + ast.unparse(kw.value) for kw in call.keywords]
        wrapped += [indent + "    " + a + "," for a in args]
        wrapped.append(indent + ")")
    sort_body = "\n".join(wrapped) + "\n"
    after = before
    anchor = "        half_count = leaf_count * 2\n"
    after = once(after, anchor, anchor + "        if multiway:\n"
                 "            if pl.tensor.dim(position_ids, 0) >= LONG_S6_MIN_QUERY_ROWS:\n"
                 "                half_count = leaf_count\n")
    # Only the balanced S6 long path changes Cube's candidate order; the
    # panel shape, FIXPIPE scaling, reduction and two-slot handshake stay fixed.
    aic_start = after.index("def indexer_score_topk_native_cube(")
    aiv_start = after.index("        for buf_score_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):", aic_start)
    aic = after[aic_start:aiv_start]
    anchor = "                    buf_score_begin = buf_score_step * (score_tile // 2)\n"
    aic = once(aic, anchor, anchor + "                    if balance_leaves:\n"
               "                        buf_score_begin = buf_score_step * score_tile\n")
    anchor = "                            buf_prime_safe_page = pl.min(\n"
    aic = once(aic, anchor, "                            if balance_leaves:\n"
               "                                buf_prime_key_row = (\n"
               "                                    buf_logical_begin + buf_score_begin\n"
               "                                    + buf_prime_col + buf_prime_key_page * BLOCK_SIZE\n"
               "                                )\n"
               + anchor)
    anchor = "                                buf_safe_page = pl.min(\n"
    aic = once(aic, anchor, "                                if balance_leaves:\n"
               "                                    buf_key_row = (buf_logical_begin + buf_score_begin\n"
               "                                                   + buf_load_col + buf_key_page * BLOCK_SIZE)\n"
               + anchor)
    after = after[:aic_start] + aic + after[aiv_start:]
    aiv_start = after.index("        for buf_score_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):", aic_start)
    body_start = after.index("\n", aiv_start) + 1
    end = after.index("    return buffered_leaf_tid", body_start)
    original = after[body_start:end]
    original = original.replace(
        "pl.min(pl.min(buf_cache_len, (buf_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES), 0",
        "pl.min(pl.min(buf_cache_len, (buf_position + 1) // COMPRESS_RATIO),\n"
        "                                   TOPK_MAX_CANDIDATES), 0",
    )
    snippet = (ROOT / "long_query_aiv.py.inc").read_text()
    sort_call = ("                        single_new_root = indexer_topk_tile_pairs(\n"
                 "                            single_segment_data, single_segment_begin, single_segment_valid\n"
                 "                        )\n")
    snippet = once(snippet, sort_call, textwrap.indent(sort_body, " " * 24))
    new = ("            if balance_leaves:\n"
           + textwrap.indent(snippet, " " * 16)
           + "            else:\n" + textwrap.indent(original, " " * 4))
    after = after[:body_start] + new + after[end:]
    ast.parse(after)
    return after


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("禁止修改已提交任务的源码")
    for side in ("baseline", "candidate"):
        dest = Path(f"{PREFIX}-{side}")
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_indexer.py"
    target = Path(f"{PREFIX}-candidate") / relative
    before = target.read_text()
    after = transform(before)
    target.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/decode_indexer.py", tofile="b/decode_indexer.py")))
    source = {"baseline": "d93bba14", "base_source": str(BASE), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "仅长S6：Cube连续候选、AIV按query分工、2048分数驻留UB排序、每leaf单根及consumer同步",
              "tie_rule": "每连续2048段内部沿原sort32/mrgsort；新段优先旧段，后leaf优先前leaf",
              "risk": "与原half切分的同分顺序可能不同；不能用保护区或排序结构替代误差/token验收",
              "inplace_pass": True, "scope": "单卡长B16/短B24筛选；Native不重测"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    origin = ROOT.parent / "csa_hc_pre_priority_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        value = (origin / name).read_text().replace(origin.name, ROOT.name)
        value = value.replace("csa-hc-pre-priority-d93bba14", "csa-score-single-root-d93bba14")
        value = value.replace(OLD_PACKAGE, PACKAGE)
        value = value.replace("# Only test the explicit pre/post -> Sinkhorn dependency; keep arithmetic fixed.",
                              "# Only test Native-style long-query streaming and single-root publication.")
        (ROOT / name).write_text(value)


if __name__ == "__main__":
    main()
