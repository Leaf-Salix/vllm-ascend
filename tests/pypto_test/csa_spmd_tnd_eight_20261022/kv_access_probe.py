# SPDX-License-Identifier: Apache-2.0
"""Read already-produced attention metadata; never modify the device kernel."""

import hashlib
import json

import numpy as np


def export_access(entry, native, case, directory):
    assert entry.policy.plan_physical_rows
    tokens = case["token_capacity"]
    arrays = {
        name: entry.buffers[name].detach().cpu().numpy().copy()
        for name in ("positions", "idx_bounds", "idx_lengths", "compressed_table", "original_table")
    }
    arrays.update(
        logical_topk=native["idx_topk"].detach().cpu().numpy().copy(),
        topk_scores=native["idx_topk_scores"].detach().cpu().float().numpy().copy(),
        physical_indices=entry.buffers["attention_indices"][:tokens].detach().cpu().numpy().copy(),
        valid_blocks=entry.buffers["attention_mask"][:tokens, :5].detach().cpu().numpy().copy(),
    )
    assert arrays["logical_topk"].shape == arrays["physical_indices"].shape == (tokens, 512)
    assert arrays["valid_blocks"].shape == (tokens, 5)
    path = directory / "kv_access.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)
    info = dict(
        kind="single_execution_metadata_observation",
        performance_eligible=False,
        device_math_revalidated=False,
        unchanged_operator=(
            "merged_schedule_v1; bind actual emitted C++ to retained measurement before using this probe"
        ),
        arrays={name: dict(shape=list(a.shape), dtype=str(a.dtype)) for name, a in arrays.items()},
        archive=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cache_rows=dict(original=entry.buffers["cache_rows"].shape[0], compressed=entry.buffers["main_cache"].shape[0]),
        case=case,
        limitations=[
            "One seed/fixture observation, not a claim about serving traffic distributions",
            "No graph/L1 sampling or new mathematical-correctness claim; reuse the exact retained-code validation",
        ],
    )
    with (directory / "kv_access.json").open("x") as stream:
        stream.write(json.dumps(info, indent=2) + "\n")
    return info
