# SPDX-License-Identifier: Apache-2.0
"""仅CPU准备完整CSA候选；不改默认生产入口。"""

import argparse
import json
from pathlib import Path

import shape_specialization as shapes
from rope_flat_gather import use_flat_gather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integrated", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = json.loads((root / "b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json").read_text())
    if not args.integrated:
        changed, audit = use_flat_gather(shapes.catalog(), args.output / "math")
        shapes.catalog = lambda: changed
    case, contract = shapes.build_variant(
        shapes.descriptors_from_roots(base["root_arguments"]), max_seq_len=131078,
        output_root=args.output, case_name="h131072_b16", balanced_schedule=True,
        schedule_policy="earliest_ready", cost_profile=root / "b16_dynamic_l1_cost.json",
        qk_vector_indices=True, row_dependencies=True, rope_flat_gather=args.integrated)
    if args.integrated:
        audit = contract["rope_flat_gather"]
    report = {"case": str(case), "audit": audit, "device_executed": False}
    (args.output / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
