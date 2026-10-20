# SPDX-License-Identifier: Apache-2.0
"""列出本阶段可提交文件，并登记仅留本地的大编译产物；不自动commit/push。"""

import json
from pathlib import Path
import os


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT_NAMES = {
    "GOAL.md", "GOAL_BEFORE_10_PERCENT_20261021.md", "CHECKPOINT.json",
    "CURRENT_CHECKPOINT.md", "freeze_shared_sources.py", "run_current_wait_full.py",
    "SHARED_SOURCE_GOAL10.json", "SHARED_SOURCE_GOAL10_B1UPDATE.json",
    "SHARED_SCRATCH_DIAGNOSTIC_TASKS.json", "KNOWLEDGE_APPLICATION_20261021.md",
    "TOPK_PAIR_DIRECT_DESIGN.md", "run_address_audited_full.py",
    "run_b1_root_queries.py", "run_pair_direct_merge.py", "compile_pair_direct_helper.py",
    "paired_compact_capacity_probe.py", "collect_goal10_focus.py",
    "prepare_goal10_archive.py",
    "ptoas_tmov_subview_issue_followup.md",
}
PREFIXES = (
    "goal10", "b1", "pair_direct", "prepare_b1", "prepare_pair_direct", "controls_goal10",
    "score_pool_goal10", "prepare_score_pool", "ptoas_pair_direct",
)
SOURCE_NAMES = {
    "source_topk_pair_direct", "source_b1_leaf4k", "source_b1_root_resident",
    "source_b1_legacy_half", "source_b1_half_score_resident", "source_score_pool_goal10",
    "b1_root_query_regression",
}
EXCLUDED_DIRS = {
    "build_output", "compile_cache", "cache", "dfx", "passes_dump", "__pycache__",
    "kernel", "source", "reference_source", "codegen_probe", "spmd_reference_codegen",
    "spmd_candidate_codegen", "generated",
}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".jsonl", ".csv", ".log", ".txt", ".patch", ".diff"}


def main():
    roots = [
        p for p in HERE.iterdir()
        if p.name in ROOT_NAMES or p.name in SOURCE_NAMES or p.name.lower().startswith(PREFIXES)
    ]
    # 从任务清单取实际输出路径，避免遗漏同址A/A等没有候选名前缀的结果。
    for path in roots.copy():
        if path.is_file() and path.suffix == ".json" and "TASK" in path.name:
            data = json.loads(path.read_text())
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                if "output" in row:
                    output = Path(row["output"])
                    output = output if output.is_absolute() else REPO / output
                    if output.parent == HERE and output.exists():
                        roots.append(output)
    selected, excluded = {}, {}

    def add(path):
        if path.is_symlink():
            selected[str(path.relative_to(REPO))] = {
                "path": str(path.relative_to(REPO)), "symlink": os.readlink(path),
            }
        elif path.is_file():
            size = path.stat().st_size
            row = {"path": str(path.relative_to(REPO)), "bytes": size}
            small_repro = path.suffix in {".pto", ".cpp"} and size <= 128 * 1024
            if (path.suffix in TEXT_SUFFIXES and size <= 2 * 1024 * 1024) or small_repro:
                selected[row["path"]] = row
            else:
                row["reason"] = "大文本/生成物/非轻量文本；只保留本地路径，不做hash"
                excluded[row["path"]] = row

    for root in roots:
        if root.is_symlink() or root.is_file():
            add(root)
            continue
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root):
            current = Path(directory)
            for name in dirs.copy():
                path = current / name
                if path.is_symlink():
                    add(path)
                    dirs.remove(name)
                elif name in EXCLUDED_DIRS:
                    excluded[str(path.relative_to(REPO))] = {
                        "path": str(path.relative_to(REPO)),
                        "reason": "编译/cache/重复源快照目录；本地保留",
                    }
                    dirs.remove(name)
            for name in files:
                add(current / name)
    log = REPO / "vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md"
    add(log)
    record = HERE / "GOAL10_ARCHIVE_MANIFEST.json"
    manifest_path = str(record.relative_to(REPO))
    selected.pop(manifest_path, None)
    paths = sorted(selected) + [manifest_path]
    report = {
        "purpose": "仅本会话10%目标阶段源码、脚本、原始轻量证据及日志",
        "hashes_computed": False, "push_authorized": False,
        "frozen_public_sources": [
            "SHARED_SOURCE_GOAL10.json", "SHARED_SOURCE_GOAL10_B1UPDATE.json",
        ],
        "snapshot_recovery": "freeze_shared_sources.py --from-record --snapshot-name 对应目录 --record-name 对应记录",
        "files": [selected[k] for k in sorted(selected)],
        "local_only": [excluded[k] for k in sorted(excluded)],
    }
    record.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    output = Path("/tmp/csa_goal10_archive_paths.txt")
    output.write_text("\n".join(paths) + "\n")
    print(json.dumps({
        "files": len(paths),
        "bytes": sum(row.get("bytes", 0) for row in selected.values()),
        "local_only_entries": len(excluded), "path_list": str(output),
    }))


if __name__ == "__main__":
    main()
