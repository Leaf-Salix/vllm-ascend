# SPDX-License-Identifier: Apache-2.0
"""Check the event-allocation compiler against the retained CSA notification binary."""

import argparse
import json
from pathlib import Path

import tmr_csa_full_128k_b16_compile as demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = args.previous.resolve(strict=True)
    target = args.output / "previous"
    if not target.exists():
        target.symlink_to(previous, target_is_directory=True)
    assert target.resolve() == previous
    (args.output / "reused_artifacts.json").write_text(json.dumps({"previous": str(previous)}, indent=2) + "\n")
    for variant in ("baseline", "mixed"):
        demo.compile_variant(variant, args.output / variant)


if __name__ == "__main__":
    main()
