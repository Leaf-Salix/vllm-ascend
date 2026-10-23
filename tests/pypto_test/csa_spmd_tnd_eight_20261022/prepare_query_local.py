# SPDX-License-Identifier: Apache-2.0
"""Freeze query-at-Attention scheduling, preserving all retained mechanisms."""

import argparse
import ast
import hashlib
import json
import runpy
from pathlib import Path

from prepare_scale_gather import replace_once


def prepare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v5", choices=("v5",))
    args = parser.parse_args()
    name_prefix = f"query_local_{args.version}"
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "merged_schedule_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    declaration = "    SCORE_GROUP_ORDER: pl.constexpr = False,\n"
    for name in ("entry.py", "cube_schedule.py", "vector_schedule.py"):
        assert declaration in source[name]
        source[name] = source[name].replace(declaration, declaration + "    LOCAL_QUERY_PREP: pl.constexpr = False,\n")
    argument = "            SCORE_GROUP_ORDER=SCORE_GROUP_ORDER,\n"
    assert source["entry.py"].count(argument) == 3
    source["entry.py"] = source["entry.py"].replace(
        argument, argument + "            LOCAL_QUERY_PREP=LOCAL_QUERY_PREP,\n"
    )
    vector_call = source["entry.py"].index(") = vector_schedule(\n")
    query_argument = source["entry.py"].index("            q_flat,\n", vector_call)
    source["entry.py"] = source["entry.py"][:query_argument] + source["entry.py"][query_argument:].replace(
        "            q_flat,\n", "            q_flat,\n            q_heads,\n", 1
    )
    annotation = "q_heads: pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16]"
    assert source["entry.py"].count(annotation) == 2
    source[("entry.py")] = source[("entry.py")].replace(
        annotation, ("q_heads: pl.InOut[pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16]]")
    )
    policy = source["policy.py"]
    policy = replace_once(
        policy,
        "    score_group_order: bool = False\n",
        "    score_group_order: bool = False\n    local_query_prep: bool = False\n",
    )
    policy = replace_once(
        policy,
        '            "SCORE_GROUP_ORDER": self.score_group_order,\n',
        (
            '            "SCORE_GROUP_ORDER": self.score_group_order,\n'
            '            "LOCAL_QUERY_PREP": self.local_query_prep,\n'
        ),
    )
    policy = replace_once(
        policy,
        "        cooperative_prefix=True,\n",
        (
            "        cooperative_prefix=True,\n"
            "        local_query_prep=long_history and group == 6 and 64 <= tokens <= "
            "192,\n"
        ),
    )
    source["policy.py"] = policy

    cube = source["cube_schedule.py"]
    acquire = (
        "    signal = wait_row_owners(signal, PHASE_QUERY, CORES, generation, 0, VECTORS, 1, VECTORS, GROUPS, 1)\n"
    )
    cube = replace_once(cube, acquire, "    if not LOCAL_QUERY_PREP:\n    " + acquire)
    begin = cube.index("    score_ring, pv_ring = attention_qk_pv_cube_stage(\n")
    end = cube.index("    signal = publish_phase(signal, PHASE_ATTN_CUBE", begin)
    call = cube[begin:end]
    call = replace_once(call, "        2,\n", "        2,\n        LOCAL_QUERY_PREP,\n")
    source["cube_schedule.py"] = cube[:begin] + call + cube[end:]

    vector = source["vector_schedule.py"]
    vector = replace_once(vector, "    QR_ROWS,\n", "    QR_ROWS,\n    QUERY_HEAD_ROWS,\n")
    vector = replace_once(
        vector,
        "    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],\n",
        "    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],\n"
        "    q_heads: pl.InOut[pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16]],\n",
    )
    begin = vector.index("    # 每个head占两个NZ N256列块。")
    end = vector.index("    if not COOPERATIVE_PREFIX:\n", begin)
    original = vector[begin:end].rstrip() + "\n"
    vector = (
        vector[:begin]
        + "    if not LOCAL_QUERY_PREP:\n"
        + "".join("    " + line if line.strip() else line for line in original.splitlines(keepends=True))
        + "\n"
        + vector[end:]
    )
    marker = "    # 事件2..5用于Attention三槽流水，和Indexer的0/1分开。"
    vector = replace_once(
        vector,
        marker,
        (
            "    if LOCAL_QUERY_PREP:\n"
            "        # Acquire the original Q_B payload before any local query is "
            "produced.\n"
            "        signal = wait_row_owners(signal, PHASE_QB, CORES, generation, 0, "
            "128, 1, GROUPS, 0, 1)\n"
        )
        + marker,
    )
    begin = vector.index(
        "    kv_ring, probability_ring, mi_ring, li_ring, packed = attention_qk_pv_vector_lane_stage(\n"
    )
    end = vector.index("    signal = publish_phase(signal, PHASE_ATTN_VECTOR", begin)
    call = vector[begin:end]
    call = replace_once(call, "li_ring, packed = attention", "li_ring, packed, q_heads = attention")
    call = replace_once(
        call,
        "        ffts,\n",
        (
            "        ffts,\n"
            "        wq_b_scale,\n"
            "        rope_signed,\n"
            "        rope_swap,\n"
            "        q_heads,\n"
            "        qr_scale_pad,\n"
            "        q_proj,\n"
        ),
    )
    call = replace_once(
        call, "        PLAN_PHYSICAL_ROWS,\n", "        PLAN_PHYSICAL_ROWS,\n        LOCAL_QUERY_PREP,\n"
    )
    call += (
        "    if LOCAL_QUERY_PREP:\n"
        "        # Retain the phase record; no consumer uses it as a global query "
        "gate.\n"
        "        signal = publish_phase(signal, PHASE_QUERY, core, CORES, generation)"
        "\n"
    )
    source["vector_schedule.py"] = vector[:begin] + call + vector[end:]

    attention = source["attention_stream_stages.py"]
    marker = 'Q_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_Q_ROWS")'
    attention = replace_once(
        attention,
        marker,
        (
            "from .query_local_stages import local_query_prepare_stage\n"
            "from .query_projection_stages import PAD_ROWS_DYN, QPROJ_ROWS_DYN\n"
            "\n"
        )
        + marker,
    )
    attention = replace_once(attention, "QK_PV_READY_EVENT = 3\n", "QK_PV_READY_EVENT = 3\nQUERY_READY_EVENT = 6\n")
    marker = "    EVENT_BASE: pl.constexpr = 0,\n"
    assert attention.count(marker) == 2
    attention = attention.replace(marker, marker + "    LOCAL_QUERY_PREP: pl.constexpr = False,\n", 1)
    marker = "            if qk_sb == 0:\n                qk_q = pl.gather_row"
    attention = replace_once(
        attention,
        marker,
        "            if qk_sb == 0:\n                if LOCAL_QUERY_PREP:\n"
        "                    pl.system.sync_wait(QUERY_READY_EVENT, pipe=pl.PipeType.MTE2, "
        "core_type=pl.KernelType.AIC)\n"
        "                qk_q = pl.gather_row",
    )
    begin = attention.index("def attention_qk_pv_vector_lane_stage(\n")
    head, body = attention[:begin], attention[begin:]
    marker = "    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],\n"
    body = replace_once(
        body,
        marker,
        marker
        + (
            "    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],\n"
            "    query_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],\n"
            "    query_swap: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],\n"
            "    q_flat: pl.InOut[pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16]],\n"
            "    qr_scale: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],\n"
            "    q_proj: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],\n"
        ),
    )
    body = replace_once(
        body,
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n    LOCAL_QUERY_PREP: pl.constexpr = False,\n",
    )
    marker = "        qk_sb = qk_tick % SPARSE_BLOCKS\n        if qk_tick < qk_block_count:\n"
    body = replace_once(
        body,
        marker,
        marker
        + (
            "            if LOCAL_QUERY_PREP and qk_sb == 0:\n"
            "                # Every capacity query has exactly one owning cluster, "
            "including dummy rows.\n"
            "                q_flat = local_query_prepare_stage(\n"
            "                    wq_b_scale, freqs_cos, query_sin_signed, query_swap, "
            "q_flat,\n"
            "                    qr_scale, q_proj, qk_t, qk_aiv,\n"
            "                )\n"
            "                pl.system.sync_set(\n"
            "                    QUERY_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, "
            "core_type=pl.KernelType.AIV\n"
            "                )\n"
        ),
    )
    body = replace_once(
        body,
        "    return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads\n",
        "    return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads, q_flat\n",
    )
    body = body.replace("q_flat", "query_heads").replace(
        "query_heads: pl.InOut[pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16]]",
        "query_heads: pl.InOut[pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16]]",
    )
    source["attention_stream_stages.py"] = head + body
    source["query_local_stages.py"] = (here / "query_local_stages_template.py").read_text()
    for code in source.values():
        ast.parse(code)
    target = here / "runs" / name_prefix
    target.mkdir(exist_ok=False)
    for name, code in source.items():
        (target / name).write_text(code)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    baseline = json.loads((here / "baseline.json").read_text())
    domains = {}
    for case_id, row in baseline["cases"].items():
        case = row["case"]
        descriptor = json.loads((here / f"descriptors/{case_id}_merged_schedule_v1.json").read_text())
        descriptor["policy"] = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        domains[case_id] = descriptor["policy"]["LOCAL_QUERY_PREP"]
        with (here / f"descriptors/{case_id}_{name_prefix}.json").open("x") as stream:
            stream.write(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        parent="merged_schedule_v1.json",
        adopted=False,
        device_execution=False,
        domains=domains,
        mechanism=(
            "Main Query preparation at the owning Attention cluster; Score starts without the unrelated query stage"
        ),
        dependencies=(
            "Retain Q_B acquire on AIV; replace all-48 PHASE_QUERY acquire on AIC with "
            "per-query FFTS event 6 from both lanes"
        ),
        limitations=[
            "Weight-scale traffic increases; extra FFTS cost must be measured",
            (
                "Eight head rows replace eight token rows, with unchanged 512-column "
                "reduction and original tail arithmetic"
            ),
            (
                "Existing q_heads alias is declared InOut because AIV now writes it; packed "
                "arguments and buffer ownership remain unchanged"
            ),
        ],
        files={name: hashlib.sha256(code.encode()).hexdigest() for name, code in source.items()},
    )
    with (here / f"{name_prefix}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    prepare()
