# SPDX-License-Identifier: Apache-2.0
"""N128 Score trial with three M128 QKs and lifetime-based Acc reuse.

Current N64 already has two M192/48KiB QK accumulators. The prior N128/M192
trial needed a single96KiB QK accumulator and was slower. This candidate uses
two64KiB accumulators for three M128 QKs; WS reuses the retired QK1 accumulator
while QK2 remains live through the first WS multiply. Its delayed FIX then
releases the other accumulator. Actual allocation and FIX/M reuse dependencies
must prove this fits128KiB before any device run.

The three K128 WS products, rounding, ReLU, key prefetch, native paging, leaves,
TopK and FIFO contract follow the already exact-checked v39 trial. Only the
selected long G6 domain uses this path. No B1 or output-projection trial is added.
"""

import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare_score_wide_k128 import function


def fix(part, indent):
    return "\n".join(" " * indent + line for line in (
        "wide_current = pl.tile.assemble(",
        f"    wide_current, triple_qk{part}, [{part} * TRIPLE_QK_ROWS, 0],",
        "    pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,",
        ")",
    )) + "\n"


def ws(part, indent, suffix=""):
    lines = [
        f"wide_right{part}{suffix} = pl.tile.extract(",
        f"    wide_previous, {part} * WIDE_WS_K_ROWS, 0, [WIDE_WS_K_ROWS, qk_panel_cols],",
        "    target_memory=pl.MemorySpace.Right,",
        ")",
    ]
    if part == 0:
        lines.append(f"wide_ws0{suffix} = pl.tile.matmul(split_coefficient0, wide_right0{suffix})")
    else:
        lines += [
            f"wide_ws{part}{suffix} = pl.tile.matmul_acc(",
            f"    wide_ws{part - 1}{suffix}, split_coefficient{part}, wide_right{part}{suffix},",
            ")",
        ]
    return "\n".join(" " * indent + line for line in lines) + "\n"


def main():
    here = Path(__file__).resolve().parent
    original = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    destination = here / "runs/score_threeway_native_candidate_v1"
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    previous = here / "runs/score_wide_k128_native_candidate_v1/indexer_score_stages.py"
    body = function(previous.read_text(), "_indexer_score_cube_split_store")
    doc = ast.get_source_segment(body, ast.parse(body).body[0].body[0])
    body = body.replace(doc, '"""Three M128 QKs; retire QK1 before WS0, and QK2 FIX after WS0 issue."""', 1)
    start = body.index("            # 两份query")
    end = body.index("            # 双槽最多", start)
    preparation = ""
    for part in range(3):
        preparation += f'''            split_query{part} = pl.tile.extract(
                buf_query_vector, {part} * TRIPLE_QK_ROWS, 0, [TRIPLE_QK_ROWS, IDX_HEAD_DIM],
                target_memory=pl.MemorySpace.Left,
            )
            split_coefficient{part} = pl.tile.extract(
                buf_coefficients_l1, 0, {part} * WIDE_WS_K_ROWS, [NATIVE_QLI_WEIGHT_ROWS, WIDE_WS_K_ROWS],
                target_memory=pl.MemorySpace.Left,
            )
'''
    body = body[:start] + preparation + body[end:]
    start = body.index("                    wide_current = pl.tile.create(")
    end = body.index("                    if key_prefetch_panels > 0:\n", start)
    steady = '''                    wide_current = pl.tile.create(
                        [query_group_size * IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16,
                        target_memory=pl.MemorySpace.Mat,
                    )
                    triple_qk0 = pl.tile.matmul(split_query0, buf_key_panel)
                    triple_qk1 = pl.tile.matmul(split_query1, buf_key_panel)
'''
    steady += fix(0, 20)
    steady += "                    triple_qk2 = pl.tile.matmul(split_query2, buf_key_panel)\n"
    steady += fix(1, 20)
    steady += "                    if buf_qk_panel > 0:\n"
    # Keeping QK2 live across WS0 makes the earlier Acc the reusable candidate.
    # No barrier is removed; the backend must retain FIX->M protection on reuse.
    steady += ws(0, 24) + fix(2, 24) + ws(1, 24) + ws(2, 24)
    steady += '''                        pl.store(
                            pl.set_validshape(wide_ws2, query_group_size, qk_panel_cols),
                            [buf_transfer_row, buf_panel_col - qk_panel_cols], buf_score_transfer,
                        )
                    else:
'''
    steady += fix(2, 24)
    steady += "                    wide_previous = wide_current\n"
    body = body[:start] + steady + body[end:]
    start = body.index("                wide_right0_last = pl.tile.extract(")
    end = body.index("                pl.system.sync_set(SCORE_READY_EVENT", start)
    tail = "".join(ws(part, 16, "_last") for part in range(3))
    tail += '''                pl.store(
                    pl.set_validshape(wide_ws2_last, query_group_size, qk_panel_cols),
                    [buf_transfer_row, score_tile - qk_panel_cols], buf_score_transfer,
                )
'''
    body = body[:start] + tail + body[end:]
    assert "split_qk_acc" not in body and "split_ws_acc" not in body
    ast.parse(body)
    path = destination / "indexer_score_stages.py"
    before = path.read_text()
    after = before.replace(function(before, "_indexer_score_cube_split_store"), body, 1)
    marker = "SPLIT_QK_ROWS = 3 * IDX_N_HEADS"
    insertion = after.index("\n", after.index(marker)) + 1
    constants = "TRIPLE_QK_ROWS = 2 * IDX_N_HEADS\nWIDE_WS_K_ROWS = 2 * IDX_N_HEADS\n"
    after = after[:insertion] + constants + after[insertion:]
    wrapper = function(after, "indexer_score_cube_stage")
    updated = wrapper.replace("qk_panel_cols == 64", "qk_panel_cols == 128", 1)
    marker = "            64,  # 此helper固定的数学资源合同；不是T/B/历史实值。"
    assert updated.count(marker) == 1
    updated = updated.replace(marker, marker.replace("64,", "128,"), 1)
    after = after.replace(wrapper, updated, 1)
    for name in ("_indexer_score_cube_original", "indexer_score_vector_lane_stage"):
        assert function(after, name) == function(before, name)
    ast.parse(after)
    path.write_text(after)
    changes = {path.name: (before, after)}
    path = destination / "policy.py"
    before = path.read_text()
    marker = "    return CSASPMDPolicy(\n"
    assert before.count(marker) == 1
    after = before.replace(marker,
        "    wide_score = long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS\n" + marker, 1)
    after = after.replace("qk_panel=64 if group == 6 else 128,",
                          "qk_panel=128 if wide_score else 64 if group == 6 else 128,", 1)
    ast.parse(after)
    path.write_text(after)
    changes[path.name] = (before, after)
    patch = []
    for name, (before, after) in changes.items():
        patch.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                         fromfile="a/" + name, tofile="b/" + name))
    (here / "score_threeway_v1.patch").write_text("".join(patch))
    (here / "score_threeway_v1.json").write_text(json.dumps({
        "candidate": str(destination), "adopted": False, "qk_m_parts": [128, 128, 128], "qk_n": 128,
        "ws_k_parts": [128, 128, 128],
        "expected_acc": "Two64KiB QK; WS8KiB must reuse retired QK1 while QK2 stays live",
        "required_audit": "Actual Acc overlap/lifetime, FIX->M reuse, exact K128 WS, Key prefetch, input/output guards",
        "not_claimed": "No device correctness, engine overlap or performance yet",
    }, indent=2) + "\n")
    description = json.loads((here / "runs/cohort_prefix_a241_h131072_b16_s6_v41/buffer_descriptors.json").read_text())
    description["policy"]["QK_PANEL"] = 128
    (here / "runs/score_threeway_b16_descriptors.json").write_text(json.dumps(description, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
