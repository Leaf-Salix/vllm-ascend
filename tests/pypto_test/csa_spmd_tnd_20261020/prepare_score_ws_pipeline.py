# SPDX-License-Identifier: Apache-2.0
"""Three-way Score follow-up: keep early Key prefetch in L1 and double WS Right.

v61 was exact but slower. Its first WS Right load waited for the current Key
to retire, and all three WS Right loads were serial with their MMAs. Here the
first WS operand is loaded before the current Key enters Right; after QK, the
retired Key region can hold a second WS operand. No additional tensor, matrix
partition, numeric ordering, FIFO protocol or input layout is introduced.
"""

import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare_score_threeway import fix
from prepare_score_wide_k128 import function


def lines(text, indent):
    return "\n".join(" " * indent + line for line in text.splitlines()) + "\n"


def current_key(indent):
    return lines('''if key_prefetch_panels > 0:
    buf_key_panel = pl.tile.extract(
        pl.tile.transpose_view(buf_key_l1_pool), 0, buf_qk_panel % 2 * qk_panel_cols,
        [IDX_HEAD_DIM, qk_panel_cols], target_memory=pl.MemorySpace.Right,
    )
else:
    buf_key_panel = pl.tile.move(pl.tile.transpose_view(buf_panel_keys), target_memory=pl.MemorySpace.Right)''', indent)


def right(part, indent, suffix=""):
    return lines(f'''wide_right{part}{suffix} = pl.tile.extract(
    wide_previous, {part} * WIDE_WS_K_ROWS, 0, [WIDE_WS_K_ROWS, qk_panel_cols],
    target_memory=pl.MemorySpace.Right,
)''', indent)


def multiply(part, indent, suffix=""):
    if part == 0:
        return lines(f"wide_ws0{suffix} = pl.tile.matmul(split_coefficient0, wide_right0{suffix})", indent)
    return lines(f'''wide_ws{part}{suffix} = pl.tile.matmul_acc(
    wide_ws{part - 1}{suffix}, split_coefficient{part}, wide_right{part}{suffix},
)''', indent)


def qk(indent):
    text = lines('''triple_qk0 = pl.tile.matmul(split_query0, buf_key_panel)
triple_qk1 = pl.tile.matmul(split_query1, buf_key_panel)''', indent)
    text += fix(0, indent)
    text += lines("triple_qk2 = pl.tile.matmul(split_query2, buf_key_panel)", indent)
    return text + fix(1, indent)


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/score_threeway_native_candidate_v1"
    destination = here / "runs/score_ws_pipeline_native_candidate_v2"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = destination / "indexer_score_stages.py"
    before = path.read_text()
    body = function(before, "_indexer_score_cube_split_store")
    doc = ast.get_source_segment(body, ast.parse(body).body[0].body[0])
    body = body.replace(doc, '"""Key prefetch stays in L1; the current Key retires before two-slot WS Right."""', 1)
    start = body.index("                buf_current_key = pl.tile.create(")
    end = body.index("                buf_key_l1_pool =", start)
    body = body[:start] + body[end:]
    start = body.index("                    if key_prefetch_to_l0:\n")
    end = body.index("                # 固定panel顺序", start)
    body = body[:start] + body[end:]
    start = body.index("                    if key_prefetch_panels > 0:\n", body.index("for buf_qk_panel"))
    end = body.index("                    if buf_qk_panel + key_prefetch_panels", start)
    # Define the fallback Mat handle in both compile-time domains. It is dead
    # for the selected prefetch1 path, and the existing zero-prefetch math works.
    body = body[:start] + lines('''buf_panel_keys = pl.tile.create(
    [qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
)''', 20) + body[end:]
    duplicate = '''                        buf_panel_keys = pl.tile.create(
                            [qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
                        )
'''
    assert body.count(duplicate) == 1
    body = body.replace(duplicate, "", 1)
    early_key = (
        "                        if key_prefetch_panels > 0:\n                            if key_prefetch_to_l0:"
    )
    start = body.index(early_key)
    end = body.index("                    wide_current = pl.tile.create(", start)
    body = body[:start] + body[end:]
    start = body.index("                    triple_qk0 =")
    end = body.index("                wide_right0_last =", start)
    steady = lines("if buf_qk_panel > 0:", 20)
    steady += right(0, 24) + current_key(24) + qk(24)
    steady += right(1, 24) + multiply(0, 24) + fix(2, 24)
    steady += right(2, 24) + multiply(1, 24) + multiply(2, 24)
    steady += lines('''pl.store(
    pl.set_validshape(wide_ws2, query_group_size, qk_panel_cols),
    [buf_transfer_row, buf_panel_col - qk_panel_cols], buf_score_transfer,
)''', 24)
    steady += lines("else:", 20) + current_key(24) + qk(24) + fix(2, 24)
    steady += lines("wide_previous = wide_current", 20)
    body = body[:start] + steady + body[end:]
    start = body.index("                wide_right0_last =")
    end = body.index("                pl.store(\n                    pl.set_validshape(wide_ws2_last", start)
    tail = right(0, 16, "_last") + right(1, 16, "_last") + multiply(0, 16, "_last")
    tail += right(2, 16, "_last") + multiply(1, 16, "_last") + multiply(2, 16, "_last")
    body = body[:start] + tail + body[end:]
    assert "buf_current_key" not in body and "buf_prefetched_key" not in body
    ast.parse(body)
    after = before.replace(function(before, "_indexer_score_cube_split_store"), body, 1)
    for name in ("_indexer_score_cube_original", "indexer_score_vector_lane_stage"):
        assert function(after, name) == function(before, name)
    path.write_text(after)
    changes = {path.name: (before, after)}
    path = destination / "policy.py"
    before = path.read_text()
    marker = '            "TO_L0": True,'
    assert before.count(marker) == 1
    after = before.replace(marker, '            "TO_L0": not self.split_score_rows,', 1)
    path.write_text(after)
    changes[path.name] = (before, after)
    patch = []
    for name, (before, after) in changes.items():
        patch.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                         fromfile="v1/" + name, tofile="v2/" + name))
    (here / "score_ws_pipeline_v2.patch").write_text("".join(patch))
    (here / "score_ws_pipeline_v2.json").write_text(json.dumps({
        "candidate": str(destination), "adopted": False, "parent": str(source),
        "numeric_partitions_unchanged": "QK threeM128/N128/K128; WS threeK128; same FIX scale/ReLU and K order",
        "supply": "Next Key still prefetched into L1; current Key enters Right after first WS preload",
        "required_audit": "Two32KiB WS Right slots; Key16KiB only in a retired slot; retain M/MTE1 reuse",
        "no_device_claim": True,
    }, indent=2) + "\n")
    description = json.loads((here / "runs/score_threeway_b16_descriptors.json").read_text())
    description["policy"]["TO_L0"] = False
    (here / "runs/score_ws_pipeline_b16_descriptors.json").write_text(json.dumps(description, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
