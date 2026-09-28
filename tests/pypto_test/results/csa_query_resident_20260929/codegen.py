"""Record whether Query/coefficients really moved outside the worker leaf loop."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def inspect(side):
    path = ROOT / "compiled" / side / "kernels/aic/indexer_score_topk_native_pair_aic.cpp"
    text = path.read_text().split("static __aicore__ void indexer_score_topk_native_pair_aiv", 1)[0]
    lines = text.splitlines()
    # The first outer loop scans sequence lengths; the second enumerates leaves.
    outer_loops = [i + 1 for i, line in enumerate(lines) if re.match(r"^  for \(", line)]
    query_loads = [i + 1 for i, line in enumerate(lines) if re.search(r"\bTLOAD\(", line)][:2]
    left_moves = [i + 1 for i, line in enumerate(lines) if re.search(r"\bTMOV\(", line)][:2]
    if len(outer_loops) < 2 or len(query_loads) != 2 or len(left_moves) != 2:
        raise ValueError(f"Unexpected generated kernel structure: {side}")
    leaf_loop = outer_loops[1]
    outside = all(line < leaf_loop for line in query_loads + left_moves)
    if outside != (side == "candidate"):
        raise ValueError(f"Query/coefficient residency did not lower as intended: {side}")
    return {"source": str(path), "worker_leaf_loop_line": leaf_loop,
            "query_coefficient_TLOAD_lines": query_loads,
            "query_coefficient_TMOV_lines": left_moves,
            "outside_worker_leaf_loop": outside}


def main():
    evidence = {"scope": "Generated load/move placement, not a measured bandwidth or latency claim",
                "sides": {side: inspect(side) for side in ("baseline", "candidate")},
                "uniform_128K_B24": {"leaf_count": 5, "query_bytes_per_group": 49152,
                                    "coefficient_bytes_per_group": 12288,
                                    "source_level_saved_load_bytes_per_worker": 4 * (49152 + 12288)},
                "selection": "24 S6 query groups, identical last-query visible counts, more than one leaf"}
    (ROOT / "codegen.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
