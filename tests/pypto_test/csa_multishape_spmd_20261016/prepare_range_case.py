# SPDX-License-Identifier: Apache-2.0
"""CPU 区间模板生成；fixture 描述符不是实际设备输入证据。"""
import argparse
import json
from pathlib import Path

from prepare_cases import resize_descriptor_fixture
from range_policy import build_range_variant
from shape_specialization import STATIC_ROOT, descriptors_from_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--reference", default="h131072_b24")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("不覆盖已有生成记录")
    anchor = json.loads((STATIC_ROOT / "bindings" / f"{args.reference}_prepared.json").read_text())
    old_batch = anchor["root_arguments"]["kv_seq_lens"]["shape"][0]
    descriptors = resize_descriptor_fixture(descriptors_from_roots(anchor["root_arguments"]),
                                            old_batch, args.batch)
    case, contract = build_range_variant(descriptors, max_seq_len=args.history + 6,
                                         output_root=args.output)
    report = dict(case=str(case), anchor=contract["anchor"], range_policy=contract["range_policy"],
                  cpu_protocol=contract["cpu_protocol"], passed=True,
                  source_schedule=contract["source_schedule_audit"] is not None,
                  cpu_lowering_passed=False, device_validated=False, performance_measured=False)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(case=str(case), anchor=contract["anchor"], passed=True),ensure_ascii=False))


if __name__ == "__main__":
    main()
