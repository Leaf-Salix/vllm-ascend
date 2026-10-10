#!/usr/bin/env python3
"""Summarize the four Native SK mechanism controls without dropping samples."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for filename in (
        "01_graph.jsonl",
        "02_sk_early_off.jsonl",
        "03_sk_wait_first.jsonl",
        "04_sk_wait_after_scalar.jsonl",
    ):
        records = [
            json.loads(line) for line in (args.output / filename).read_text().splitlines() if line.startswith('{"sk":')
        ]
        if len(records) != 1:
            raise ValueError(f"{filename}: expected one result, got {len(records)}")
        record = records[0]
        trace = record["trace"]
        if record["mismatches"] or not record["checksum_ok"]:
            raise ValueError(f"{filename}: output/checksum mismatch")
        if record["sk"] and record["early_start"]:
            if trace[0] != trace[16] or not trace[1] & 16 or not trace[17] & 32:
                raise ValueError(f"{filename}: same-core Early-Start was not established")
        elif trace[1] or trace[17]:
            raise ValueError(f"{filename}: unexpected Early-Start configuration")
        row = {
            **record,
            "file": filename,
            "producer_core": trace[0],
            "consumer_core": trace[16],
            "producer_config": trace[1],
            "consumer_config": trace[17],
            "handoff_ticks": trace[18] - trace[3],
            "scalar_ticks": trace[20] - trace[19],
            "wait_ticks": (trace[19] - trace[18] if record["wait_first"] else trace[21] - trace[20]),
        }
        rows.append(row)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Control | min us | mean us | max us | A/B core | A/B config | Scalar/Wait ticks")
    for row in rows:
        print(
            f"{row['file']} | {row['min_us']:.3f} | {row['mean_us']:.3f} | "
            f"{row['max_us']:.3f} | {row['producer_core']}/{row['consumer_core']} | "
            f"{row['producer_config']}/{row['consumer_config']} | "
            f"{row['scalar_ticks']}/{row['wait_ticks']}"
        )


if __name__ == "__main__":
    main()
