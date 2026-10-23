# SPDX-License-Identifier: Apache-2.0
"""Combine a current-plan validity proof with pure DMA then independent softmax."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_attention_overlap import indent, make_stream
from prepare_scale_gather import replace_once

PURE_DMA = """            for dma_half, (chunk_in,) in pl.range(2, init_values=(dma_kv_iter,)):
                for dma_row, (row_buf,) in pl.range(KV_DMA_ROWS, init_values=(chunk_in,)):
                    physical_row = pl.cast(
                        pl.read(physical_rows, [0, dma_half * KV_DMA_ROWS + dma_row]), pl.INDEX
                    )
                    dma_result = pl.gather_row(
                        row_buf, cmp_kv_flat, [dma_row, 0], [physical_row, 0], [1, HEAD_DIM]
                    )
                    dma_loaded = pl.yield_(dma_result)
                pl.store(dma_loaded, [qk_kv_row + qk_lane_kv + dma_half * KV_DMA_ROWS, 0], kv_transfer)
                dma_finished = pl.yield_(dma_loaded)
"""


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "merged_schedule_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    # Reuse only this exact, previously checked proof producer, not the rejected
    # MGATHER-Undefined consumer. Event predicates and geometry columns stay put.
    proof_manifest = json.loads((here / "plan_gather_valid_v1.json").read_text())
    proof = (Path(proof_manifest["source"]) / "attention_plan_stages.py").read_text()
    assert hashlib.sha256(proof.encode()).hexdigest() == proof_manifest["files"]["attention_plan_stages.py"]
    source["attention_plan_stages.py"] = proof
    stream = make_stream(source["attention_stream_stages.py"])
    stream = replace_once(
        stream,
        "    COMPRESSED_COLUMNS_DYN,\n",
        "    COMPRESSED_COLUMNS_DYN,\n    COMPRESSED_ALL_VALID_COL0,\n",
    )
    split = stream.index("@pl.jit.inline(auto_scope=False)\ndef attention_qk_pv_vector_overlap_stage(")
    prefix, fast = stream[:split], stream[split:]
    fast = replace_once(
        fast, "def attention_qk_pv_vector_overlap_stage(", "def attention_qk_pv_vector_valid_overlap_stage("
    )
    fast = replace_once(
        fast,
        "                    overlap_ready = pl.cast(1, pl.INT32)\n",
        "                    if pl.read(valid_block_mask, [qk_t, COMPRESSED_ALL_VALID_COL0 + qk_sb - 1]) == 1:\n"
        "                        overlap_ready = pl.cast(1, pl.INT32)\n",
    )
    fast = replace_once(fast, "    dma_zero_row = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)\n", "")
    math_begin = fast.index("            qk_masked = pl.col_expand_add(")
    math_end = fast.index("            qk_kv_row = (qk_core", math_begin)
    math = fast[math_begin:math_end]
    fast = fast[:math_begin] + fast[math_end:]
    dma_begin = fast.index("            for dma_half,")
    dma_end = fast.index("            if t_dim >= EARLY_KV_MIN_TOKENS:\n", dma_begin)
    fast = fast[:dma_begin] + PURE_DMA + fast[dma_end:]
    commit = fast.index("            pl.store(qk_probability,")
    fast = fast[:commit] + math + fast[commit:]
    source["attention_stream_stages.py"] = prefix + fast

    vector = source["vector_schedule.py"]
    vector = replace_once(
        vector,
        "from .attention_stream_stages import attention_qk_pv_vector_lane_stage\n",
        "from .attention_stream_stages import attention_qk_pv_vector_lane_stage, "
        "attention_qk_pv_vector_valid_overlap_stage\n",
    )
    call_name = "    kv_ring, probability_ring, mi_ring, li_ring, packed = attention_qk_pv_vector_lane_stage(\n"
    begin = vector.index(call_name)
    end = vector.index("    signal = publish_phase(signal, PHASE_ATTN_VECTOR", begin)
    call = vector[begin:end]
    selected = "    if PLAN_PHYSICAL_ROWS:\n"
    selected += indent(
        call.replace("attention_qk_pv_vector_lane_stage", "attention_qk_pv_vector_valid_overlap_stage"), 8
    )
    selected += "    else:\n" + indent(call, 8)
    source["vector_schedule.py"] = vector[:begin] + selected + vector[end:]
    for content in source.values():
        ast.parse(content)
    target = here / "runs/attention_valid_overlap_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        parent="merged_schedule_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="Prefetch Score/bias, issue pure row DMA/store loops, preserve original KV-before-softmax on large T",
        proof="Exact plan producer from plan_gather_valid_v1; "
        "complete 128-row validity is re-established every invocation",
        fallback="Original whole gather/softmax body whenever current proof or previous/current block validity fails",
        distinction="The rejected early-Vector candidate changes when math is issued; "
        "this candidate keeps math after KV",
        gate="Actual resource and synchronization audit; "
        "proof, event, math and fallback checks before any NPU submission",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "attention_valid_overlap_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
