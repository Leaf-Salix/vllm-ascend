"""Read only: compare all five rounds against a fixed source, without cherry-picking."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ROUNDS = [(11, "score_atomic_admission"), (12, "coeff_no_early"), (13, "score_48_workers"), (14, "score_leaf_major"), (15, "score_adaptive_admission")]
rows = []
for number, suffix in ROUNDS:
    directory = ROOT / f"round{number}_{suffix}"
    report = json.loads((directory / "report.json").read_text())
    for case in report["cases"]:
        if case["label"].startswith(f"round{number}_"):
            rows.append(dict(round=number, history=case["history"], batch=case["batch"], timing=case["timing"], decision=report["decision"], report=str(directory / "report.json")))
(ROOT / "five_rounds.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")
for row in rows:
    body=row["timing"]["body"]
    print(row["round"], row["history"], *(round(body[key],2) for key in ("mean_us","p50_us","p95_us")),row["decision"])
