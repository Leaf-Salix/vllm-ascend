# SPDX-License-Identifier: Apache-2.0
"""Validate observed physical plans and measure locality without changing order."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    return (
        dict(
            count=int(data.size),
            minimum=float(data.min()),
            median=float(np.median(data)),
            mean=float(data.mean()),
            p95=float(np.quantile(data, 0.95)),
            maximum=float(data.max()),
        )
        if data.size
        else {}
    )


def row_set(rows):
    return set(int(x) for x in rows if x >= 0)


def analyze_arrays(arrays, compressed_rows):
    logical = arrays["logical_topk"].astype(np.int64)
    physical = arrays["physical_indices"].astype(np.int64)
    bounds = arrays["idx_bounds"].astype(np.int64)
    lengths = arrays["idx_lengths"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    table = arrays["compressed_table"].astype(np.int64)
    masks = arrays["valid_blocks"]
    tokens = len(positions)
    assert logical.shape == physical.shape == (tokens, 512)
    assert bounds[0] == 0 and bounds[-1] == tokens and np.all(np.diff(bounds) > 0)
    assert len(bounds) == len(lengths) + 1 and table.shape[0] == len(lengths)
    assert np.all((positions >= 0) & (positions < 2**24 - 1))
    owners = np.searchsorted(bounds[1:], np.arange(tokens), side="right")
    expected = np.full_like(physical, -1)
    for query, request in enumerate(owners):
        assert positions[query] == lengths[request] - (bounds[request + 1] - bounds[request]) + query - bounds[request]
        # Canonical FP32 upper bound is exact in this fixture's checked range.
        upper = int(np.trunc((np.float32(positions[query]) + np.float32(1)) * np.float32(0.25)))
        for slot, index in enumerate(logical[query]):
            if 0 <= index < upper and index // 32 < table.shape[1]:
                page = table[request, index // 32]
                if 0 <= page < compressed_rows // 32:
                    expected[query, slot] = page * 32 + index % 32
    np.testing.assert_array_equal(physical, expected)
    np.testing.assert_array_equal(masks[:, 1:] > 0, np.any(expected.reshape(tokens, 4, 128) >= 0, axis=2))

    segments = []
    for query in range(tokens):
        for segment in range(8):
            rows = physical[query, segment * 64 : (segment + 1) * 64]
            valid = rows >= 0
            joins = valid[1:] & valid[:-1] & (np.diff(rows) == 1)
            count = int(valid.sum())
            pages = len(set((rows[valid] // 32).tolist()))
            segments.append(
                dict(
                    query=query,
                    segment=segment,
                    valid_rows=count,
                    contiguous_runs=count - int(joins.sum()),
                    unique_pages=pages,
                )
            )
    pairs, requests = [], []
    for request, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        sets = [row_set(rows) for rows in physical[begin:end]]
        total = sum(len(s) for s in sets)
        union = set().union(*sets)
        requests.append(
            dict(
                request=request,
                query_count=int(end - begin),
                total_unique_rows_per_query=total,
                union_rows=len(union),
                repeated_row_fraction=1 - len(union) / total if total else 0,
            )
        )
        for query in range(begin + 1, end):
            previous, current = row_set(physical[query - 1]), row_set(physical[query])
            exact64 = sum(
                np.array_equal(physical[query, i : i + 64], physical[query - 1, i : i + 64]) for i in range(0, 512, 64)
            )
            previous_blocks = {tuple(physical[query - 1, i : i + 128]) for i in range(0, 512, 128)}
            exact128 = sum(tuple(physical[query, i : i + 128]) in previous_blocks for i in range(0, 512, 128))
            pairs.append(
                dict(
                    request=request,
                    previous_query=int(query - 1),
                    query=int(query),
                    row_set_reuse_fraction=len(previous & current) / len(current) if current else 0,
                    same_slot_rows=int(np.sum((physical[query] >= 0) & (physical[query] == physical[query - 1]))),
                    exact_same_slot_segments64=int(exact64),
                    exact_blocks128=int(exact128),
                )
            )

    valid_rows = sum(s["valid_rows"] for s in segments)
    runs = sum(s["contiguous_runs"] for s in segments)
    page_rows = sum(s["unique_pages"] * 32 for s in segments)
    waves = []
    for first in range(0, tokens, 24):
        sets = [row_set(rows) for rows in physical[first : first + 24]]
        waves.append(dict(first_query=first, queries=len(sets), unique_rows=len(set().union(*sets))))
    request_waves = []
    if len(lengths) <= 24:
        for step in range(int(np.diff(bounds).max())):
            queries = [int(begin + step) for begin, end in zip(bounds[:-1], bounds[1:]) if begin + step < end]
            request_waves.append(
                dict(
                    step=step,
                    queries=len(queries),
                    unique_rows=len(set().union(*(row_set(physical[q]) for q in queries))),
                )
            )
    result = dict(
        host_plan_mapping_verified=True,
        tokens=tokens,
        requests=len(lengths),
        valid_compressed_rows=valid_rows,
        completely_valid_128_blocks=int(np.all(physical.reshape(tokens, 4, 128) >= 0, axis=2).sum()),
        active_128_blocks=int(np.any(physical.reshape(tokens, 4, 128) >= 0, axis=2).sum()),
        contiguous_runs_within_existing_64_row_lane_segments=runs,
        upper_bound_row_request_reduction_fraction=1 - runs / valid_rows if valid_rows else 0,
        whole_page_overfetch_ratio_per_existing_lane_segment=page_rows / valid_rows if valid_rows else 0,
        segment_run_distribution=distribution([s["contiguous_runs"] for s in segments]),
        segment_page_distribution=distribution([s["unique_pages"] for s in segments]),
        consecutive_query_row_set_reuse=distribution([p["row_set_reuse_fraction"] for p in pairs]),
        consecutive_query_same_slot_fraction=distribution([p["same_slot_rows"] / 512 for p in pairs]),
        exact_64_segments=sum(p["exact_same_slot_segments64"] for p in pairs),
        exact_128_blocks=sum(p["exact_blocks128"] for p in pairs),
        repeated_row_fraction_within_request=distribution([r["repeated_row_fraction"] for r in requests]),
        current_logical_24_query_waves=waves,
        hypothetical_one_request_per_cluster_waves=request_waves,
        per_request=requests,
        consecutive_pairs=pairs,
        limitations=[
            "Rows remain in the actual TopK order; no sort or floating-point reduction order change is assumed",
            "Run merging is only an upper bound on fewer row requests, not a speedup or emitted DMA count",
            "Set overlap does not imply reusable ordered L1 tiles; retain the exact segment/block counts",
            "Logical waves approximate working sets, not a measured hardware schedule or L2-miss rate",
            "One fixture/seed cannot establish serving-wide reuse or justify a general fast path",
        ],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    probe = json.loads((args.run / "result.json").read_text())
    metadata = json.loads((args.run / "kv_access.json").read_text())
    assert probe["passed"] and probe["stage"] == "metadata_export_complete" and not probe["performance_eligible"]
    assert probe["sampling"] == {"eager_runs": 1, "valid_windows": 0}
    retained = json.loads((here / "merged_schedule_v1.json").read_text())
    assert probe["source_hashes"]["candidate"] == retained["files"]
    case = probe["case"]["id"]
    reference_path = here / "critical_prefix_v1_results.json"
    reference_info = json.loads(reference_path.read_text())["cases"][case]
    reference_path = here / reference_info["report"]
    reference = json.loads(reference_path.read_text())
    assert reference["passed"] and probe["case"] == reference["case"]
    assert probe["weights"] == reference["weights"] and all(probe["fixture_checks"].values())
    require_same_sdk(probe["shared_toolchain"], reference["shared_toolchain"])
    cpp_files = list((args.run / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp"))
    assert len(cpp_files) == 1
    original = executable(cpp_files[0].read_text())
    matches = [
        str(p)
        for p in (reference_path.parent / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp")
        if executable(p.read_text()) == original
    ]
    assert matches, "Observation does not execute the exact retained measured kernel"
    archive = Path(metadata["archive"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
    with np.load(archive, allow_pickle=False) as arrays:
        result = analyze_arrays(arrays, metadata["cache_rows"]["compressed"])
    result.update(
        source_report=str(args.run / "result.json"),
        archive=str(archive),
        performance_eligible=False,
        retained_measured_cpp=matches,
        case=probe["case"],
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("per_request", "consecutive_pairs", "case")}, indent=2)
    )


if __name__ == "__main__":
    main()
