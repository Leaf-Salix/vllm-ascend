# SPDX-License-Identifier: Apache-2.0
"""归档E34/E35/E36的轻量证据，生成显式Git路径清单；不提交、不push、不做hash。"""
import json
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=here, text=True).strip())
    names = ["score_attention_interleave_current_20261019",
             "attention_direct_l1_feasibility_20261019",
             "attention_direct_l1_full_pipeline_20261019"]
    selected = []
    for name in names:
        directory = here.parent / name
        chosen = set()
        for file in directory.iterdir():
            if (file.is_file() and file.name != "ARCHIVE_FILES.json"
                    and file.suffix in {".py", ".sh", ".md", ".json", ".txt", ".patch", ".log"}
                    and file.stat().st_size < 1_500_000):
                chosen.add(file)
        for child in directory.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("source_") or child.name in {"diagnostic_harness", "harness_revisions", "minimal_generated"}:
                for file in child.rglob("*"):
                    if (file.is_file() and "__pycache__" not in str(file)
                            and file.suffix in {".py", ".sh", ".json", ".cpp", ".pto", ".txt"}
                            and file.stat().st_size < 1_500_000):
                        chosen.add(file)
            result = child / "result.json"
            if result.exists():
                content = json.loads(result.read_text())
                terminal = not (isinstance(content, dict) and "stage" in content
                                and content.get("stage") != "complete" and "error" not in content)
                if terminal:
                    chosen.add(result)
                    for file in child.iterdir():
                        if (file.is_file() and
                                (file.name in {"failure_snapshot.json", "failure_values.json", "failure_error_full.txt", "failure_traceback_stack.txt"}
                                 or file.name.startswith("eager_")) and file.stat().st_size < 1_500_000):
                            chosen.add(file)
            for file in child.glob("*isa*/*.jsonl"):
                if file.is_file() and file.stat().st_size < 100_000:
                    chosen.add(file)
            if child.name.endswith("_isa_audit"):
                chosen.update(child.glob("*.jsonl"))
        local = [dict(path=str(file.relative_to(directory)), bytes=file.stat().st_size)
                 for file in directory.rglob("*")
                 if file.is_file() and file not in chosen and file.name != "ARCHIVE_FILES.json" and not file.is_symlink()]
        manifest = directory / "ARCHIVE_FILES.json"
        manifest.write_text(json.dumps(dict(
            scope="源码、轻量记录、终态结果和最小复现入Git；其余构建/私有头文件/二进制本地保留",
            hash_calculated=False,
            selected=[dict(path=str(file.relative_to(directory)), bytes=file.stat().st_size) for file in sorted(chosen)],
            local_only=local), ensure_ascii=False, indent=2) + "\n")
        chosen.add(manifest)
        selected.extend(chosen)
        print(name, len(chosen), sum(file.stat().st_size for file in chosen))
    selected.append(repo / "vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md")
    paths = sorted(str(file.relative_to(repo)) for file in selected)
    target = Path("/tmp/csa_phase_20261019_paths.nul")
    target.write_bytes(b"\0".join(path.encode() for path in paths) + b"\0")
    print("explicit_paths", len(paths), "file", target)


if __name__ == "__main__":
    main()
