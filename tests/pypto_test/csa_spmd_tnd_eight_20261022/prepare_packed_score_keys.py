# SPDX-License-Identifier: Apache-2.0
"""Prototype coalesced AIV key supply while keeping original Score math."""

import argparse
import ast
import hashlib
import json
import runpy
import textwrap
from pathlib import Path

from prepare_fused_coefficients import forward_flag
from prepare_scale_gather import replace_once


def splice_function(source, name, edit):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [edit("".join(lines[node.lineno - 1 : node.end_lineno]))]
    return "".join(lines)


def add_flag(source, calls):
    # Reuse the AST-located call insertion without a second ad hoc parser.
    return forward_flag(source, calls).replace(
        "FUSED_INDEXER_COEFF=FUSED_INDEXER_COEFF", "PACKED_SCORE_KEYS=PACKED_SCORE_KEYS"
    )


def cube_function(source):
    source = replace_once(
        source,
        "    buf_worker: pl.Scalar[pl.INDEX],\n",
        "    packed_keys: pl.Tensor[[PACK_KEY_ROWS, IDX_HEAD_DIM], pl.INT8],\n    buf_worker: pl.Scalar[pl.INDEX],\n",
    )
    source = replace_once(
        source,
        "    SCORE_GROUP_ORDER: pl.constexpr = False,\n",
        "    SCORE_GROUP_ORDER: pl.constexpr = False,\n    PACKED_SCORE_KEYS: pl.constexpr = False,\n",
    )
    # The <=3-query path is left byte-for-byte unchanged. Only the full branch
    # receives a larger L1 key pool and the optional proven transfer path.
    start = source.index("            else:\n                buf_cache_len =")
    full = source[start:]
    full = replace_once(
        full,
        "                for buf_score_step in pl.range(buf_score_iters):\n",
        "                packed_direction = pl.cast(0, pl.INT32)\n"
        "                packed_first = pl.cast(0, pl.INDEX)\n"
        "                if PACKED_SCORE_KEYS and buf_valid_count % score_tile == 0:\n"
        "                    packed_first, packed_direction = packed_leaf_layout_scalar(\n"
        "                        idx_block_table, buf_batch_idx, buf_logical_begin // BLOCK_SIZE,\n"
        "                        buf_valid_count // BLOCK_SIZE, pl.tensor.dim(idx_native_kv_cache, 0))\n"
        "                for buf_score_step in pl.range(buf_score_iters):\n",
    )
    full = replace_once(
        full,
        "[2 * qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat",
        "[2 * qk_panel_cols + PACKED_SCORE_KEYS * (score_tile - 2 * qk_panel_cols), IDX_HEAD_DIM], "
        "dtype=pl.INT8, target_memory=pl.MemorySpace.Mat",
    )
    anchor = "                    if key_prefetch_panels > 0:\n                        buf_prime_col = 0\n"
    full = replace_once(
        full,
        anchor,
        "                    if PACKED_SCORE_KEYS and packed_direction != 0:\n"
        "                        pl.system.sync_wait(PACK_READY_EVENT, "
        "pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)\n"
        "                        buf_key_l1_pool = pl.gather_row(\n"
        "                            buf_key_l1_pool, packed_keys, [0, 0],\n"
        "                            [(buf_worker * 2 + buf_score_step % 2) * score_tile, 0],\n"
        "                            [score_tile, IDX_HEAD_DIM])\n" + anchor,
    )
    # Guard only original page DMAs; Right prefetch and all MMA/FIX remain in order.
    begin = full.index("                        for buf_prime_key_page in pl.unroll(")
    end = full.index("                        if key_prefetch_to_l0:", begin)
    full = (
        full[:begin]
        + "                        if packed_direction == 0:\n"
        + textwrap.indent(full[begin:end], "    ")
        + full[end:]
    )
    # The first match above is nested in the page loop; use indentation-sensitive
    # line matching for the following sibling (not a substring of nested spaces).
    lines = full.splitlines(keepends=True)
    first = next(
        i for i, line in enumerate(lines) if line.startswith("                            for buf_key_page in")
    )
    last = next(
        i
        for i in range(first + 1, len(lines))
        if lines[i].startswith("                            if key_prefetch_panels > 0:")
    )
    lines[first:last] = [
        "                            if packed_direction == 0:\n",
        textwrap.indent("".join(lines[first:last]), "    "),
    ]
    full = "".join(lines)
    full = replace_once(
        full,
        "                        buf_load_slot = (buf_qk_panel + key_prefetch_panels) % 2 * qk_panel_cols\n",
        "                        buf_load_slot = (buf_qk_panel + key_prefetch_panels) % 2 * qk_panel_cols\n"
        "                        if packed_direction != 0:\n"
        "                            buf_load_slot = (buf_qk_panel + key_prefetch_panels) * qk_panel_cols\n",
    )
    # This prototype is selected only with PREFETCH=1, TO_L0=True. The unselected
    # path retains its original two-slot offsets and original emitted behavior.
    return source[:start] + full


def vector_function(source):
    source = replace_once(
        source,
        "    buf_score_lane: pl.Scalar[pl.INDEX],\n",
        "    packed_keys: pl.InOut[pl.Tensor[[PACK_KEY_ROWS, IDX_HEAD_DIM], pl.INT8]],\n"
        "    buf_score_lane: pl.Scalar[pl.INDEX],\n",
    )
    source = replace_once(
        source,
        "    SCORE_GROUP_ORDER: pl.constexpr = False,\n",
        "    SCORE_GROUP_ORDER: pl.constexpr = False,\n    PACKED_SCORE_KEYS: pl.constexpr = False,\n",
    )
    anchor = "                for single_step in pl.range(single_steps):\n"
    prepare = (
        "                packed_enabled = pl.cast(0, pl.INT32)\n"
        "                if PACKED_SCORE_KEYS and buf_live > 3 and single_valid % score_tile == 0:\n"
        "                    if single_contiguous != 0:\n"
        "                        packed_enabled = pl.cast(1, pl.INT32)\n"
        "                if packed_enabled != 0:\n"
        "                    packed_keys = supply_packed_score_keys(\n"
        "                        idx_native_kv_cache, packed_keys, buf_worker, buf_score_lane,\n"
        "                                             0, single_first_page, single_contiguous)\n"
    )
    ahead = (
        "                    if packed_enabled != 0 and single_step + 1 < single_steps:\n"
        "                        packed_keys = supply_packed_score_keys(\n"
        "                            idx_native_kv_cache, packed_keys, buf_worker, buf_score_lane,\n"
        "                                                 single_step + 1, single_first_page, single_contiguous)\n"
    )
    return replace_once(source, anchor, prepare + anchor + ahead)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="packed_score_keys_v6")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["packed_score_keys.py"] = (here / "packed_score_keys_template.py").read_text()
    policy = replace_once(
        source["policy.py"],
        "    plan_table_columns: int = DEFAULT_PHYSICAL_PLAN_COLUMNS\n",
        "    plan_table_columns: int = DEFAULT_PHYSICAL_PLAN_COLUMNS\n    packed_score_keys: bool = False\n",
    )
    policy = replace_once(
        policy,
        '            "PLAN_TABLE_COLUMNS": self.plan_table_columns,\n',
        '            "PLAN_TABLE_COLUMNS": self.plan_table_columns,\n'
        '            "PACKED_SCORE_KEYS": self.packed_score_keys,\n',
    )
    policy = replace_once(
        policy,
        "        leaf_scale_layout=leaf_scale_layout,\n",
        "        leaf_scale_layout=leaf_scale_layout,\n        packed_score_keys=leaf_scale_layout,\n",
    )
    source["policy.py"] = policy
    for name, count in (("entry.py", 2), ("cube_schedule.py", 1), ("vector_schedule.py", 1)):
        marker = "    SCORE_GROUP_ORDER: pl.constexpr = False,\n"
        assert source[name].count(marker) == count
        source[name] = source[name].replace(marker, marker + "    PACKED_SCORE_KEYS: pl.constexpr = False,\n")
    source["entry.py"] = add_flag(source["entry.py"], {"cube_schedule", "vector_schedule", "_csa_core_group"})
    # Bind a zero-copy Torch view so the JIT receives concrete tensor metadata.
    # v1/v2 incore view lowering and v3 orchestration inference were rejected.
    for name in ("entry.py", "cube_schedule.py", "vector_schedule.py"):
        source[name] = replace_once(
            source[name],
            "import pypto.language as pl\n",
            "import pypto.language as pl\n\nfrom .packed_score_keys import PACK_KEY_ROWS\n",
        )
    signature = "    indexer_transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],\n"
    packed_signature = "    packed_keys: pl.InOut[pl.Tensor[[PACK_KEY_ROWS, 128], pl.INT8]],\n"
    for function in ("_csa_core_group", "decode_csa_spmd"):
        source["entry.py"] = splice_function(
            source["entry.py"], function, lambda text: replace_once(text, signature, signature + packed_signature)
        )
    source["binding.py"] = replace_once(
        source["binding.py"],
        '        cache = self.buffers["idx_cache"]\n',
        '        self.buffers["packed_keys"] = self.buffers["score_arena"].view(torch.int8).view(-1, 128)\n'
        '        cache = self.buffers["idx_cache"]\n',
    )
    for name in ("cube_schedule.py", "vector_schedule.py"):
        schedule_signature = next(
            line + "\n" for line in source[name].splitlines() if line.startswith("    indexer_transfer:")
        )
        role_signature = (
            "    packed_keys: pl.Tensor[[PACK_KEY_ROWS, 128], pl.INT8],\n"
            if name == "cube_schedule.py"
            else packed_signature
        )
        source[name] = replace_once(source[name], schedule_signature, schedule_signature + role_signature)
    entry = ast.parse(source["entry.py"])
    lines = source["entry.py"].splitlines(keepends=True)
    inserts = []
    for call in ast.walk(entry):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"cube_schedule", "vector_schedule", "_csa_core_group"}
        ):
            arg = next(arg for arg in call.args if isinstance(arg, ast.Name) and arg.id == "indexer_transfer")
            line = lines[arg.lineno - 1]
            inserts.append((arg.lineno, line[: len(line) - len(line.lstrip())] + "packed_keys,\n"))
    for line, text in sorted(inserts, reverse=True):
        lines.insert(line, text)
    source["entry.py"] = "".join(lines)
    cube = source["cube_schedule.py"]
    cube = replace_once(
        cube,
        "        indexer_transfer,\n        worker,\n",
        "        indexer_transfer,\n        packed_keys,\n        worker,\n",
    )
    source["cube_schedule.py"] = add_flag(cube, {"indexer_score_cube_stage"})
    source["vector_schedule.py"] = add_flag(source["vector_schedule.py"], {"indexer_score_vector_lane_stage"})
    source["vector_schedule.py"] = replace_once(
        source["vector_schedule.py"],
        "        pair_arena,\n        pl.tile.get_subblock_idx(),\n",
        "        pair_arena,\n        packed_keys,\n        pl.tile.get_subblock_idx(),\n",
    )
    score = replace_once(
        source["indexer_score_stages.py"],
        "from .request_geometry import",
        "from .packed_score_keys import PACK_KEY_ROWS, PACK_READY_EVENT, supply_packed_score_keys, "
        "packed_leaf_layout_scalar\n"
        "from .request_geometry import",
    )
    score = splice_function(score, "_indexer_score_cube_split_store", cube_function)

    def wrapper(text):
        text = replace_once(
            text,
            "    buf_worker: pl.Scalar[pl.INDEX],\n",
            "    packed_keys: pl.Tensor[[PACK_KEY_ROWS, IDX_HEAD_DIM], pl.INT8],\n"
            "    buf_worker: pl.Scalar[pl.INDEX],\n",
        )
        text = replace_once(
            text,
            "    SCORE_GROUP_ORDER: pl.constexpr = False,\n",
            "    SCORE_GROUP_ORDER: pl.constexpr = False,\n    PACKED_SCORE_KEYS: pl.constexpr = False,\n",
        )
        prefix, fallback = text.split("    else:\n", 1)
        prefix = replace_once(
            prefix,
            "            buf_score_transfer,\n            buf_worker,\n",
            "            buf_score_transfer,\n            packed_keys,\n            buf_worker,\n",
        )
        prefix = add_flag(prefix + "    else:\n        pass\n", {"_indexer_score_cube_split_store"}).split(
            "    else:\n", 1
        )[0]
        return prefix + "    else:\n" + fallback

    score = splice_function(score, "indexer_score_cube_stage", wrapper)
    source["indexer_score_stages.py"] = splice_function(score, "indexer_score_vector_lane_stage", vector_function)
    target = here / "runs" / args.variant
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    desc = json.loads((here / "descriptors/C04_output_rectangle_v1.json").read_text())
    desc["buffers"]["packed_keys"] = dict(
        shape=[desc["buffers"]["score_arena"]["shape"][0] * 256, 128], dtype="torch.int8"
    )
    desc["policy"] = select(
        tokens=96,
        requests=16,
        max_query_tokens=6,
        max_seq_len=131078,
        compressed_table_columns=desc["buffers"]["compressed_table"]["shape"][1],
    ).constexprs()
    assert desc["policy"]["PACKED_SCORE_KEYS"] and desc["policy"]["PREFETCH"] == 1 and desc["policy"]["TO_L0"]
    with (here / f"descriptors/C04_{args.variant}.json").open("x") as f:
        f.write(json.dumps(desc, indent=2) + "\n")
    manifest = dict(
        source=str(target),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        pilot="C04 only",
        root_abi_unchanged=False,
        root_alias="One internal INT8 view of existing FP32 score_arena; no additional allocation or business input",
        mechanism="Two AIVs coalesce proven full Native key pages; prefill and one-step-ahead GM packing",
        fallback="Original key page DMA for partial leaves, noncontiguous pages and <=3 live queries",
        workspace="First 6 MiB of existing 9 MiB balanced-G6 score_arena; no other G6 users",
        protocol="New KEY_READY event6; original Score READY/ACK proves previous key slot consumed",
        risks=[
            "Extra GM traffic",
            "Reverse-page UB gather cost",
            "Resource/lifetime/alias lowering",
            "Changed startup",
        ],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
    )
    with (here / f"{args.variant}.json").open("x") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
