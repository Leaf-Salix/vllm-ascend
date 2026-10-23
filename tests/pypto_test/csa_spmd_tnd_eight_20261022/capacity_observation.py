# SPDX-License-Identifier: Apache-2.0
"""Read physical-plan outputs after all timing/checks; never change the kernel."""

import json

import numpy as np
from analyze_kv_access import analyze_arrays


def observe_capacity(entry, native, case, directory):
    assert case["id"] == "C02" and entry.policy.plan_physical_rows
    tokens = case["token_capacity"]
    arrays = {
        name: entry.buffers[name].detach().cpu().numpy().copy()
        for name in ("positions", "idx_bounds", "idx_lengths", "compressed_table")
    }
    arrays.update(
        logical_topk=native["idx_topk"].detach().cpu().numpy().copy(),
        physical_indices=entry.buffers["attention_indices"][:tokens].detach().cpu().numpy().copy(),
        valid_blocks=entry.buffers["attention_mask"][:tokens, :5].detach().cpu().numpy().copy(),
    )
    archive = directory / "physical_capacity_observation.npz"
    if archive.exists():
        raise FileExistsError(archive)
    np.savez_compressed(archive, **arrays)
    columns = arrays["compressed_table"].shape[1]
    compressed_rows = entry.buffers["main_cache"].shape[0]
    assert columns == 2050 and 0 < compressed_rows // 32 <= 16777216
    analysis = analyze_arrays(arrays, compressed_rows)
    assert analysis["host_plan_mapping_verified"]
    assert np.any(arrays["physical_indices"] != arrays["logical_topk"])
    valid_logical = arrays["logical_topk"][arrays["logical_topk"] >= 0]
    report = dict(
        performance_eligible=False,
        observation_after_all_L1_and_strict_checks=True,
        actual_table_columns=columns,
        old_capacity=2048,
        candidate_capacity=2080,
        physical_indices_match_independent_host_mapping=True,
        physical_representation_differs_from_logical_indices=True,
        largest_selected_logical_column=int(valid_logical.max() // 32) if valid_logical.size else None,
        selected_indices_in_columns_2048_or_later=int(np.sum(valid_logical // 32 >= 2048)),
        valid_compressed_rows=analysis["valid_compressed_rows"],
        archive=str(archive),
        limitation="Representative fixture observation; "
        "separate tail/invalid-page state coverage is required for adoption",
    )
    with (directory / "physical_capacity_observation.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report
