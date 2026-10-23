# SPDX-License-Identifier: Apache-2.0
"""Count logical Attention interface payloads, without claiming HBM traffic."""

import ast
import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    source = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/attention_stream_stages.py"
    wanted = {"H": 64, "HEAD_DIM": 512, "ATTN_K_TILE": 128, "SPARSE_BLOCKS": 5, "QK_TRANSFER_SLOTS": 3}
    constants = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    assert constants == wanted
    baseline = json.loads((here / "baseline.json").read_text())
    report = dict(
        measured=False,
        performance_eligible=False,
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        constants=constants,
        assumptions=[
            "All capacity queries active, all five sparse blocks valid and each contains 128 valid KV rows",
            "Count semantic tensor payload at each named interface; ignore DMA padding and extra implementation copies",
            "QK and PV reuse the existing L1 KV tile; do not count a second KV GM-to-L1 load",
        ],
        limitations=[
            "These are logical interface bytes, not measured HBM bytes or DRAM bandwidth",
            "L2 can serve repeated ring accesses; adding interfaces counts repeated traversal of the same payload",
            "Invalid masks/rows reduce actual traffic; "
            "mi/li, metadata, scalar reads and query preparation are excluded",
            "No speedup is inferred by dividing these bytes by a nominal bandwidth",
        ],
        cases={},
    )
    for case in ("C04", "C05"):
        tokens = baseline["cases"][case]["case"]["token_capacity"]
        kv = tokens * constants["SPARSE_BLOCKS"] * constants["ATTN_K_TILE"] * constants["HEAD_DIM"] * 2
        scores = tokens * constants["SPARSE_BLOCKS"] * constants["H"] * constants["ATTN_K_TILE"] * 4
        probabilities = scores // 2
        pv = tokens * constants["SPARSE_BLOCKS"] * constants["H"] * constants["HEAD_DIM"] * 4
        query = tokens * constants["H"] * constants["HEAD_DIM"] * 2
        channels = dict(
            kv_cache_to_vector=kv,
            kv_vector_to_gm_ring=kv,
            kv_gm_ring_to_cube_l1=kv,
            score_cube_fix_to_gm=scores,
            score_gm_to_vector=scores,
            probability_vector_to_gm=probabilities,
            probability_gm_to_cube_l1=probabilities,
            pv_cube_fix_to_gm=pv,
            pv_gm_to_vector=pv,
            query_gm_to_cube_l1=query,
            packed_output_to_gm=query,
        )
        report["cases"][case] = dict(
            tokens=tokens,
            channels_bytes=channels,
            channels_mib={key: value / 2**20 for key, value in channels.items()},
            total_interface_mib=sum(channels.values()) / 2**20,
            compressed_row_gather_count=tokens * (constants["SPARSE_BLOCKS"] - 1) * constants["ATTN_K_TILE"],
        )
    with (here / "attention_interface_traffic.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["cases"], indent=2))


if __name__ == "__main__":
    main()
