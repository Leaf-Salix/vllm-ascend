# SPDX-License-Identifier: Apache-2.0
"""Trial: N128 Score with K128 weighted reduction and two prefetched Right keys.

Unlike the earlier N128/K192 negative trial, 32 KiB WS Right + two 16 KiB
Key Right tiles fit the 64 KiB budget. One 96 KiB QK Acc and one 8 KiB WS
Acc fit L0C. The complete 384-row QK result stays in Mat; the middle K128
slice spans the two QK halves. Every query has one contiguous 64-head
coefficient block, so no query's nonzero terms cross a new K boundary.
Exact arithmetic and tie preservation still require actual device checks.
"""

import ast
import difflib
import json
import shutil
from pathlib import Path


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def qk(half, indent):
    pad = " " * indent
    return "\n".join(pad + line for line in (
        f"split_qk{half} = pl.tile.matmul_acc(",
        f"    split_qk_acc, split_query{half}, buf_key_panel, init_cond=True,",
        ")",
        "wide_current = pl.tile.assemble(",
        f"    wide_current, split_qk{half}, [{half} * SPLIT_QK_ROWS, 0],",
        "    pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,",
        ")",
    )) + "\n"


def ws(part, indent, suffix=""):
    pad = " " * indent
    init = ", init_cond=True" if part == 0 else ""
    previous = "split_ws_acc" if part == 0 else f"wide_ws{part - 1}{suffix}"
    return "\n".join(pad + line for line in (
        f"wide_right{part}{suffix} = pl.tile.extract(",
        f"    wide_previous, {part} * WIDE_WS_K_ROWS, 0, [WIDE_WS_K_ROWS, qk_panel_cols],",
        "    target_memory=pl.MemorySpace.Right,",
        ")",
        f"wide_ws{part}{suffix} = pl.tile.matmul_acc(",
        f"    {previous}, split_coefficient{part}, wide_right{part}{suffix}{init},",
        ")",
    )) + "\n"


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/current_bsh_native_tnd_candidate_v1"
    destination = here / "runs/score_wide_k128_native_candidate_v1"
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    previous_wide = (here.parent / "csa_critical_path_20261020/source_score_wide_pipeline/indexer_score_stages.py")
    body = function(previous_wide.read_text(), "_indexer_score_cube_wide_split_store")
    body = body.replace("_indexer_score_cube_wide_split_store", "_indexer_score_cube_split_store", 1)
    node = ast.parse(body).body[0]
    old_doc = ast.get_source_segment(body, node.body[0])
    body = body.replace(old_doc, '"""N128 with three K128 WS blocks and retained two-slot Key Right prefetch."""', 1)
    old_range = "            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count\n"
    assert body.count(old_range) == 1
    body = body.replace(old_range, old_range.rstrip("\n") + ", balance_leaves, query_group_size\n")
    for part in (0, 1):
        before = f'''            split_coefficient{part} = pl.tile.extract(
                buf_coefficients_l1, 0, {part} * SPLIT_QK_ROWS, [NATIVE_QLI_WEIGHT_ROWS, SPLIT_QK_ROWS],
                target_memory=pl.MemorySpace.Left,
            )
'''
        after = before.replace("SPLIT_QK_ROWS", "WIDE_WS_K_ROWS")
        assert body.count(before) == 1
        body = body.replace(before, after)
    marker = "            # 独立WS句柄避免FIX仍读QK时又被WS覆盖；实际地址必须审计。\n"
    assert body.count(marker) == 1
    body = body.replace(marker, '''            split_coefficient2 = pl.tile.extract(
                buf_coefficients_l1, 0, 2 * WIDE_WS_K_ROWS, [NATIVE_QLI_WEIGHT_ROWS, WIDE_WS_K_ROWS],
                target_memory=pl.MemorySpace.Left,
            )
''' + marker)
    first = body.index("                split_previous0 = pl.tile.create(")
    end = body.index("                buf_current_key =", first)
    body = body[:first] + '''                wide_previous = pl.tile.create(
                    [query_group_size * IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16,
                    target_memory=pl.MemorySpace.Mat,
                )
''' + body[end:]
    first = body.index("                    split_current0 = pl.tile.create(")
    end = body.index("                    if key_prefetch_panels > 0:\n", first)
    steady = '''                    wide_current = pl.tile.create(
                        [query_group_size * IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16,
                        target_memory=pl.MemorySpace.Mat,
                    )
                    if buf_qk_panel > 0:
'''
    steady += qk(0, 24) + ws(0, 24) + qk(1, 24) + ws(1, 24) + ws(2, 24)
    steady += '''                        pl.store(
                            pl.set_validshape(wide_ws2, query_group_size, qk_panel_cols),
                            [buf_transfer_row, buf_panel_col - qk_panel_cols], buf_score_transfer,
                        )
                    else:
'''
    steady += qk(0, 24) + qk(1, 24) + "                    wide_previous = wide_current\n"
    body = body[:first] + steady + body[end:]
    first = body.index("                split_right0_last = pl.tile.extract(")
    end = body.index("                pl.system.sync_set(SCORE_READY_EVENT", first)
    tail = "".join(ws(part, 16, "_last") for part in range(3))
    tail += '''                pl.store(
                    pl.set_validshape(wide_ws2_last, query_group_size, qk_panel_cols),
                    [buf_transfer_row, score_tile - qk_panel_cols], buf_score_transfer,
                )
'''
    body = body[:first] + tail + body[end:]
    ast.parse(body)
    path = destination / "indexer_score_stages.py"
    before = path.read_text()
    original_body = function(before, "_indexer_score_cube_split_store")
    after = before.replace(original_body, body, 1)
    marker = "SPLIT_QK_ROWS = 3 * IDX_N_HEADS"
    offset = after.index("\n", after.index(marker)) + 1
    after = after[:offset] + "WIDE_WS_K_ROWS = 2 * IDX_N_HEADS\n" + after[offset:]
    wrapper = function(after, "indexer_score_cube_stage")
    changed_wrapper = wrapper.replace("qk_panel_cols == 64", "qk_panel_cols == 128", 1)
    marker = "            64,  # 此helper固定的数学资源合同；不是T/B/历史实值。"
    assert changed_wrapper.count(marker) == 1
    changed_wrapper = changed_wrapper.replace(marker, marker.replace("64,", "128,"))
    after = after.replace(wrapper, changed_wrapper, 1)
    for name in ("_indexer_score_cube_original", "indexer_score_vector_lane_stage"):
        assert function(after, name) == function(before, name)
    ast.parse(after)
    path.write_text(after)
    changes = {path.name: (before, after)}
    path = destination / "policy.py"
    before = path.read_text()
    marker = "    return CSASPMDPolicy(\n"
    assert before.count(marker) == 1
    after = before.replace(
        marker, "    wide_score = long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS\n" + marker
    )
    after = after.replace(
        "qk_panel=64 if group == 6 else 128,", "qk_panel=128 if wide_score else 64 if group == 6 else 128,", 1
    )
    ast.parse(after)
    path.write_text(after)
    changes[path.name] = (before, after)
    patches = []
    for name, (before, after) in changes.items():
        patches.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                            fromfile="a/" + name, tofile="b/" + name))
    (here / "score_wide_k128_v1.patch").write_text("".join(patches))
    (here / "score_wide_k128_v1.json").write_text(json.dumps({
        "candidate": str(destination), "adopted": False,
        "actual_qk_panel_cols": 128, "weighted_reduction_k_blocks": [128, 128, 128],
        "expected_right_bytes": [16384, 16384, 32768], "expected_acc_bytes": [98304, 8192],
        "expected_left_bytes": [24576, 24576, 4096, 4096, 4096],
        "preserved": "B1 accepted math, AIV Score/TopK, original other Cube paths, leaf partition and FIFO protocol",
        "not_proven": "Physical resource reuse, exact FP32 accumulation and complete L1 performance",
    }, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
