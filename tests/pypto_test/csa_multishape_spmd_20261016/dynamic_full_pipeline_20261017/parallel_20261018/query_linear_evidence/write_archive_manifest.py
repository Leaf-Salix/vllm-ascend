#!/usr/bin/env python3
"""仅列私有证据路径及大小，完整编译产物不复制。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name == "paths_and_sizes.json" or "__pycache__" in path.parts:
        continue
    relative = path.relative_to(ROOT)
    rows.append(dict(relative=str(relative), path=str(path), bytes=path.stat().st_size,
                     copy_for_archive=relative.parts[0] != "cpu_v1"))
(ROOT / "paths_and_sizes.json").write_text(json.dumps(
    dict(root=str(ROOT), files=rows,
         archive_files=sum(x["copy_for_archive"] for x in rows),
         archive_bytes=sum(x["bytes"] for x in rows if x["copy_for_archive"]),
         raw_cpu_files=sum(not x["copy_for_archive"] for x in rows)), ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(files=len(rows), archive_files=sum(x["copy_for_archive"] for x in rows)), ensure_ascii=False))
