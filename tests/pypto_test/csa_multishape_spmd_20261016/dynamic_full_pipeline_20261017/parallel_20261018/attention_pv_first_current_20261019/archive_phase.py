# SPDX-License-Identifier: Apache-2.0
"""归档本轮Score结果、全核时钟和Attention重排的源码和轻量证据，生成显式Git路径；不提交、不push。"""
import json
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=here, text=True,
    ).strip())
    directories = [
        "score_native_layout_acc_pool_20261019", "score_rolled_prefetch_current_20261019",
        "current_phase_clock_20261019", "attention_pv_first_current_20261019",
    ]
    selected = set()
    for name in directories:
        directory = here.parent / name
        chosen = set()
        for file in directory.iterdir():
            if (file.is_file() and file.name != "ARCHIVE_FILES.json"
                    and file.suffix in {".py", ".sh", ".md", ".json", ".txt", ".log"}
                    and file.stat().st_size < 1_500_000):
                chosen.add(file)
        for child in directory.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("source_"):
                chosen.update(file for file in child.rglob("*.py") if "__pycache__" not in file.parts)
            result = child / "result.json"
            if result.exists():
                content = json.loads(result.read_text())
                terminal = "stage" not in content or content.get("stage") == "complete" or "error" in content
                if terminal:
                    chosen.add(result)
                    for filename in ("metadata_replay.json", "padding_replay.json", "failure_snapshot.json",
                                     "failure_values.json", "failure_error_full.txt", "source_matrix/result.json"):
                        file = child / filename
                        if file.exists() and file.stat().st_size < 1_500_000:
                            chosen.add(file)
                    # 诊断全核时钟单独归档，禁止混作性能验收数据。
                    for file in (child / "phase_clock").glob("*.json"):
                        if file.stat().st_size < 1_500_000:
                            chosen.add(file)
                    # 保留性能原始时间戳；大C++/IR/权重/二进制仍只记录本地路径。
                    for file in (child / "source_matrix").glob("l1_*/*.json"):
                        if file.stat().st_size < 1_500_000:
                            chosen.add(file)
        local = []
        for file in directory.rglob("*"):
            if file.is_file() and not file.is_symlink() and file not in chosen and file.name != "ARCHIVE_FILES.json":
                local.append(dict(path=str(file.relative_to(directory)), bytes=file.stat().st_size))
        manifest = directory / "ARCHIVE_FILES.json"
        manifest.write_text(json.dumps(dict(
            scope="源码、编译失败、终态报告、原始L1记录入Git；大构建产物本地保留；排队结果不预记",
            hash_calculated=False,
            selected=[dict(path=str(f.relative_to(directory)), bytes=f.stat().st_size) for f in sorted(chosen)],
            local_only=local,
        ), ensure_ascii=False, indent=2) + "\n")
        chosen.add(manifest)
        selected.update(chosen)
        print(name, "files", len(chosen), "bytes", sum(f.stat().st_size for f in chosen))
    selected.add(repo / "vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md")
    paths = sorted(str(f.relative_to(repo)) for f in selected)
    target = Path("/tmp/csa_attention_phase_20261019_paths.nul")
    target.write_bytes(b"\0".join(path.encode() for path in paths) + b"\0")
    print("explicit_paths", len(paths), "file", target)


if __name__ == "__main__":
    main()
