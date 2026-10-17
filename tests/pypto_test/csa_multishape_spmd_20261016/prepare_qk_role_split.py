# SPDX-License-Identifier: Apache-2.0
"""只做CPU规划，暂保留完整父任务依赖；不套用旧packed行所有权证明。"""

import argparse
import json
from pathlib import Path

import shape_specialization as shapes
from qk_role_split import use_roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-dependencies", action="store_true")
    parser.add_argument("--head-tile", type=int, choices=(8, 16), default=8)
    parser.add_argument("--softmax-tile", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--publication-tile", type=int, choices=(8, 16))
    args = parser.parse_args()
    tiles = (args.head_tile, args.softmax_tile,
             args.head_tile if args.publication_tile is None else args.publication_tile)
    if args.row_dependencies and tiles not in ((8, 8, 8), (16, 32, 8)):
        raise ValueError("只把CPU规划通过的8/8/8和16/32/8候选接入行依赖")
    root = Path(__file__).resolve().parent
    base = json.loads((root / "b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json").read_text())
    output = args.output.resolve()
    if args.row_dependencies:
        case, contract = shapes.build_variant(
            shapes.descriptors_from_roots(base["root_arguments"]), max_seq_len=131078,
            output_root=output, case_name="h131072_b16", balanced_schedule=True,
            schedule_policy="earliest_ready", cost_profile=root / "b16_dynamic_l1_cost.json",
            qk_vector_indices=True, row_dependencies=True, qk_role_split=True,
            qk_role_wide_heads=tiles == (16, 32, 8))
        audit = contract["qk_role_split"]
    else:
        bodies, audit = use_roles(base, shapes.catalog(), output / "math_variants" / "qk_roles",
                                 head_tile=args.head_tile, softmax_tile=args.softmax_tile,
                                 publication_tile=args.publication_tile)
        original = shapes.catalog
        try:
            shapes.catalog = lambda: bodies
            case, _ = shapes.build_variant(
                shapes.descriptors_from_roots(base["root_arguments"]), max_seq_len=131078,
                output_root=output, case_name="h131072_b16", balanced_schedule=True,
                schedule_policy="earliest_ready", cost_profile=root / "b16_dynamic_l1_cost.json")
        finally:
            shapes.catalog = original
    report = {"case": str(case), "audit": audit, "device_executed": False,
              "row_dependencies_enabled": args.row_dependencies,
              "scope": "CPU准备；修改输出所有权，必须配套依赖与完整设备验证"}
    (output / "roles_preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
