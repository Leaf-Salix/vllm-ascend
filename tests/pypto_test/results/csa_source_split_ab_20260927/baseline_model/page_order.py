"""Summarize recorded CPU page IDs; never read or hash device cache payloads."""

import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    report = {"scope": "rank0每次bank恢复，逻辑4页对齐panel；按预热/计时/profile三轮分别统计", "cases": []}
    for history, batch in ((131072, 16), (8192, 40)):
        source = root / f"h{history}/pto/rank0.log"
        records = [json.loads(line.split("OFFLINE_INDEXER_PAGE_ORDER ", 1)[1])
                   for line in source.read_text().splitlines() if "OFFLINE_INDEXER_PAGE_ORDER " in line]
        if len(records) != 3 * batch:
            raise ValueError(f"{source}: expected {3 * batch} page records, got {len(records)}")
        case = {"history": history, "batch": batch, "source": str(source), "rounds": []}
        for index, phase in enumerate(("warmup", "steady", "profile")):
            selected = records[index * batch:(index + 1) * batch]
            row = {"phase": phase, "requests": len(selected),
                   **{key: sum(r[key] for r in selected) for key in ("pages", "panels", "ascending", "descending")},
                   "first_request_pages": selected[0]["first_pages"]}
            row["other"] = row["panels"] - row["ascending"] - row["descending"]
            case["rounds"].append(row)
        report["cases"].append(case)
    (root / "page_order.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
