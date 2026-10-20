# SPDX-License-Identifier: Apache-2.0
"""只迁移旧wave控制到当前数学基线，保留所有冲突，绝不写正式ops。"""

import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    old = REPO / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_current_20261019"
    reference = HERE.parent / "source_after_score_split"
    target, preview = output / "source_candidate", output / "merge_preview"
    target.mkdir(exist_ok=False)
    preview.mkdir(exist_ok=False)
    records = []
    for path in sorted((old / "source_candidate_v2").glob("*.py")):
        ancestor, current = old / "source_reference" / path.name, reference / path.name
        if not ancestor.exists():
            (target/path.name).write_bytes(path.read_bytes())
            status = "added_old_wave_module"
        elif path.read_bytes() == ancestor.read_bytes():
            (target/path.name).write_bytes(current.read_bytes())
            status = "kept_current"
        else:
            result = subprocess.run(
                ["git", "merge-file", "-p", "--diff3", str(current), str(ancestor), str(path)],
                capture_output=True, text=True,
            )
            if result.returncode < 0 or result.returncode >= 128:
                raise RuntimeError(result.stderr)
            (preview/(path.name+".merge")).write_text(result.stdout)
            if result.returncode == 0:
                (target/path.name).write_text(result.stdout)
                status = "merged"
            else:
                (target/path.name).write_bytes(current.read_bytes())
                status = "conflict_keep_current_pending"
        records.append(dict(file=path.name, status=status))
    report = dict(current_reference=str(reference), old_reference=str(old/"source_reference"),
                  old_candidate=str(old/"source_candidate_v2"), files=records,
                  formal_modified=False, compiled=False, device_executed=False)
    (output/"MIGRATION_PREVIEW.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print([r for r in records if r["status"] != "kept_current"])


if __name__ == "__main__":
    main()
