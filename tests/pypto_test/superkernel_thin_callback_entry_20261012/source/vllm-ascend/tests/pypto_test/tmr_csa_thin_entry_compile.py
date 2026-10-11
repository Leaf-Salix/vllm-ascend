# SPDX-License-Identifier: Apache-2.0
"""Compare a small callback router while preserving the full task and runtime."""

import argparse
import json
import sys
from pathlib import Path

import pypto
import tmr_csa_full_128k_b16_compile as demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reused = {}
    for name in ("baseline", "mixed"):
        previous = (args.reuse / name).resolve(strict=True)
        destination = args.output / name
        if not destination.exists():
            destination.symlink_to(previous, target_is_directory=True)
        assert destination.resolve() == previous
        reused[name] = str(previous)
    (args.output / "reused_artifacts.json").write_text(json.dumps(reused, indent=2) + "\n")
    sys.path.insert(0, str(Path(pypto.__file__).resolve().parents[2] / "tests/st/runtime/scheduling"))
    from task_boundary_thin_entry_probe import thin_entry  # noqa: PLC0415

    demo.PACKAGES["thin"] = demo.PACKAGES["mixed"]
    with thin_entry():
        demo.compile_variant("thin", args.output / "thin")


if __name__ == "__main__":
    main()
