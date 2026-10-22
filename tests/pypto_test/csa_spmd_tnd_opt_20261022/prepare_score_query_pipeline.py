# SPDX-License-Identifier: Apache-2.0
"""Trial M64/N256 Score with one-query QK and previous-query WS overlap.

Distinct from historical N128 trials: QK Acc is 64 KiB and separate WS Acc
is 16 KiB; one 32 KiB INT8 key and one 32 KiB FP16 WS Right coexist. Six
M64 queries and six K64 coefficient slices occupy the original 60 KiB Left.
Only L1 key prefetch is retained. Exact device arithmetic is not assumed.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def resources():
    source = "            # Six independent query/head blocks, all retained in Left for this leaf.\n"
    for q in range(6):
        source += f"""            query_left_{q} = pl.tile.extract(
                buf_query_vector, {q} * IDX_N_HEADS, 0, [IDX_N_HEADS, IDX_HEAD_DIM],
                target_memory=pl.MemorySpace.Left,
            )
            coefficient_left_{q} = pl.tile.extract(
                buf_coefficients_l1, 0, {q} * IDX_N_HEADS, [NATIVE_QLI_WEIGHT_ROWS, IDX_N_HEADS],
                target_memory=pl.MemorySpace.Left,
            )
"""
    source += """            query_qk_acc = pl.tile.create(
                [IDX_N_HEADS, qk_panel_cols], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc,
            )
            query_ws_acc = pl.tile.create(
                [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc,
            )
"""
    return source


def weighted_step(q):
    previous = "query_ws_acc" if q == 0 else f"query_ws_{q - 1}"
    init = ", init_cond=True" if q == 0 else ""
    return f"""                    query_ws_{q} = pl.tile.matmul_acc(
                        {previous}, coefficient_left_{q}, query_right_{q}{init},
                    )
"""


def right_step(q):
    return f"""                    query_right_{q} = pl.tile.extract(
                        query_current_{q}, 0, 0, [IDX_N_HEADS, qk_panel_cols],
                        target_memory=pl.MemorySpace.Right,
                    )
"""


def panel_math():
    source = ""
    for q in range(6):
        if q:
            source += right_step(q - 1)
        source += f"""                    query_current_{q} = pl.tile.create(
                        [IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat,
                    )
                    query_qk_{q} = pl.tile.matmul_acc(
                        query_qk_acc, query_left_{q}, buf_key_panel, init_cond=True,
                    )
                    query_current_{q} = pl.tile.assemble(
                        query_current_{q}, query_qk_{q}, [0, 0],
                        pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,
                    )
"""
        if q:
            # WS consumes the preceding query, so current QK's FIX can overlap
            # an independent Acc/Right operation. Generated dependencies decide
            # whether that overlap survives; this ordering alone proves no speedup.
            source += weighted_step(q - 1)
    source += right_step(5) + weighted_step(5)
    source += """                    pl.store(
                        pl.set_validshape(query_ws_5, query_group_size, qk_panel_cols),
                        [buf_transfer_row, buf_panel_col], buf_score_transfer,
                    )
"""
    return source


def transform_score(source):
    original = function(source, "_indexer_score_cube_split_store")
    body = original
    begin = body.index("            # 两份query共48KiB Left")
    end = body.index("            # 双槽最多允许两份未消费分数", begin)
    body = body[:begin] + resources() + body[end:]
    begin = body.index("                split_previous0 = pl.tile.create(")
    end = body.index("                buf_current_key =", begin)
    body = body[:begin] + body[end:]
    begin = body.index("                    split_current0 = pl.tile.create(")
    end = body.index("                    if key_prefetch_panels > 0:\n", begin)
    body = body[:begin] + panel_math() + body[end:]
    begin = body.index("                split_right0_last = pl.tile.extract(")
    end = body.index("                pl.system.sync_set(SCORE_READY_EVENT", begin)
    body = body[:begin] + body[end:]
    doc = ast.parse(body).body[0].body[0]
    body = body.replace(
        ast.get_source_segment(body, doc),
        '"""Six M64 queries, N256 panels, unchanged FIFO and logical leaf semantics."""',
        1,
    )
    assert "SPLIT_QK_ROWS" not in body
    ast.parse(body)
    changed = source.replace(original, body, 1)
    wrapper = function(changed, "indexer_score_cube_stage")
    replacement = wrapper.replace("qk_panel_cols == 64", "qk_panel_cols == 256", 1)
    marker = "            64,  # 此helper固定的数学资源合同；不是T/B/历史实值。"
    assert replacement.count(marker) == 1
    replacement = replacement.replace(marker, marker.replace("64,", "256,"), 1)
    changed = changed.replace(wrapper, replacement, 1)
    for name in ("_indexer_score_cube_original", "indexer_score_vector_lane_stage", "indexer_score_local_half_leaf"):
        assert function(changed, name) == function(source, name)
    ast.parse(changed)
    return changed


def transform_policy(source):
    marker = "    return CSASPMDPolicy(\n"
    assert source.count(marker) == 1
    result = source.replace(
        marker,
        "    query_panel_pipeline = long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS\n" + marker,
        1,
    )
    assert result.count("qk_panel=64 if group == 6 else 128,") == 1
    result = result.replace(
        "qk_panel=64 if group == 6 else 128,", "qk_panel=256 if query_panel_pipeline else 64 if group == 6 else 128,", 1
    )
    assert result.count('"TO_L0": True,') == 1
    result = result.replace('"TO_L0": True,', '"TO_L0": self.qk_panel != 256,', 1)
    ast.parse(result)
    return result


def main():
    here = Path(__file__).resolve().parent
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    changed = {
        "indexer_score_stages.py": transform_score((reference / "indexer_score_stages.py").read_text()),
        "policy.py": transform_policy((reference / "policy.py").read_text()),
    }
    target = here / "runs/score_query_pipeline"
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, text in changed.items():
        (target / name).write_text(text)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "scope": "existing long-G6 SPLIT_SCORE_ROWS domain; no frontend or Vector math changes",
        "planned_resources_bytes": {
            "qk_acc": 65536,
            "ws_acc": 16384,
            "left_queries": 49152,
            "left_coefficients": 12288,
            "right_key": 32768,
            "right_ws": 32768,
        },
        "planned_mma_per_1024": {"qk": 24, "ws": 24},
        "not_a_latency_prediction": True,
        "preserved": [
            "QK K128",
            "FIX scale and ReLU",
            "global WS K order",
            "Native page mapping",
            "logical leaves and TopK ties",
            "1024-column FIFO slots",
            "READY/ACK and drain",
        ],
        "numerical_risk": "WS uses six K64 instructions instead of two K192; full exact device proof is required",
        "changed_files": list(changed),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    with (here / "score_query_pipeline.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
