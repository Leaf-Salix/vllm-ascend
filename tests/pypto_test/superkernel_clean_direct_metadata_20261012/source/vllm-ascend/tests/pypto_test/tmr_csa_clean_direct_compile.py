# SPDX-License-Identifier: Apache-2.0
"""Compare baseline, callback, mutable direct and metadata direct on one SDK."""

import argparse
import sys
from pathlib import Path

import pypto
import tmr_csa_full_128k_b16_compile as demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(pypto.__file__).resolve().parents[2] / "tests/st/runtime/scheduling"))
    from task_boundary_clean_direct_probe import clean_direct_notification  # noqa: PLC0415
    from task_boundary_direct_runtime_probe import direct_notification  # noqa: PLC0415

    for name in ("baseline", "mixed"):
        demo.compile_variant(name, args.output / name)
    for name, notification in (("direct", direct_notification), ("clean_direct", clean_direct_notification)):
        demo.PACKAGES[name] = demo.PACKAGES["mixed"]
        with notification():
            demo.compile_variant(name, args.output / name)


if __name__ == "__main__":
    main()
