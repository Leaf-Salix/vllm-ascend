"""Use the established official-parser checks and fixed-window path analysis."""

import argparse
import json

from collect import CASES, ROOT, load_previous

if __name__ == "__main__":
    # Reject retired shapes before the shared helper creates analysis files.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    history, batch = args.history, args.batch
    if (history, batch) not in CASES:
        raise ValueError("This shape is outside the current six-case comparison")
    load_previous("analyze_schedule").main()
    source = json.loads((ROOT / "source.json").read_text())
    path = ROOT / f"h{history}_b{batch}/schedule/README.md"
    path.write_text(path.read_text().replace("当前源码c93ec723", "当前源码" + source["operator_commit"]))
