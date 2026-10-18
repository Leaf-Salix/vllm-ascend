# SPDX-License-Identifier: Apache-2.0
"""归档本次最小复现小文本；大产物只列路径/大小，不hash、不删除原文件。"""

import json
import shutil
from pathlib import Path

from audit_evidence import FULL, HERE


def main():
    full_result = FULL / "cpu_long_v3/result.json"
    assert full_result.stat().st_size < 100_000
    data = json.loads(full_result.read_text())
    (HERE / "FULL_CHAIN_V3_LINK.json").write_text(json.dumps({
        "source": str(full_result), "bytes": full_result.stat().st_size,
        "passed": data["passed"], "symbolic_parse_passed": data["symbolic_parse_passed"],
        "error": data["error"],
        "ssa_passed_evidence": "主会话编译已越过SSA校验，到AllocateMemoryAddr才因Left上限失败",
        "other_source_fixes_present": True, "compiler_changed_for_scope_issue": False,
    }, ensure_ascii=False, indent=2) + "\n")
    audit = json.loads((HERE / "AUDIT.json").read_text())
    local_only = audit["large_files_unmodified"] + [
        {"path": entry["path"], "bytes": entry["bytes"]} for entry in audit["full_v2_passes"]
    ]
    (HERE / "LOCAL_ARTIFACTS.json").write_text(json.dumps({
        "private_evidence": str(HERE), "large_artifacts_not_copied": local_only,
        "dependency_sources_unmodified": "/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/jit_tile_parameter_overlay_20261018/pypto_overlay",
        "notes": "复用既有Tile注解overlay；没有新增compiler修复、二进制构建或整CSA编译。",
    }, ensure_ascii=False, indent=2) + "\n")
    files = sorted(path for path in HERE.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts
                   and path.name not in ("ARCHIVE_FILES.txt", "ARCHIVE_MANIFEST.json"))
    assert all(path.stat().st_size < 100_000 for path in files)
    relative = [str(path.relative_to(HERE)) for path in files]
    relative += ["ARCHIVE_FILES.txt", "ARCHIVE_MANIFEST.json"]
    (HERE / "ARCHIVE_FILES.txt").write_text("\n".join(relative) + "\n")
    (HERE / "ARCHIVE_MANIFEST.json").write_text(json.dumps({
        "source_directory": str(HERE),
        "destination": str(FULL / "unroll_iterarg_evidence"),
        "files": [{"relative_path": str(path.relative_to(HERE)), "bytes": path.stat().st_size} for path in files],
    }, ensure_ascii=False, indent=2) + "\n")
    destination = FULL / "unroll_iterarg_evidence"
    for name in relative:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (HERE / name).resolve() != target.resolve():
            shutil.copyfile(HERE / name, target)
        assert target.read_bytes() == (HERE / name).read_bytes()
    print(json.dumps({"destination": str(destination), "text_files": len(relative),
                      "bytes": sum((destination / name).stat().st_size for name in relative)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
