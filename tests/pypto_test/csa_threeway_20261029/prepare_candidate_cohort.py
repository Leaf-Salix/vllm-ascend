# SPDX-License-Identifier: Apache-2.0
"""只复制已冻结八卡框架及指定CSA候选；保留金标、原快照和测试配置。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PACKAGE = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = args.root / "cohorts/eightcard-source-v5/ours"
    manifest_path = original / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    candidate = json.loads((args.candidate.parent / "manifest.json").read_text())
    for relative, digest in candidate["changed_files"].items():
        assert Path(relative).name == relative
        assert hashlib.sha256((args.candidate / PACKAGE / relative).read_bytes()).hexdigest() == digest, relative
    candidate_files = {
        str(path.relative_to(args.candidate)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((args.candidate / PACKAGE).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert manifest["decode_dp"] == manifest["request_wave_size"] == 8
    assert manifest["gpu_blocks_override"] == 4000 and manifest["precision_only"]
    assert not args.output.exists()
    # Copy exactly the frozen manifest, excluding later generated test files.
    for relative, digest in manifest["files"].items():
        data = (original / "source" / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, relative
    target = args.output / "ours/source"
    for relative in manifest["files"]:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / "source" / relative, destination)
    # The old run generated this packaging metadata outside its manifest.
    # Reuse those exact bytes rather than probing hardware or another install.
    build_info = original / "source/vllm_ascend/_build_info.py"
    shutil.copyfile(build_info, target / "vllm_ascend/_build_info.py")
    shutil.rmtree(target / PACKAGE)
    shutil.copytree(args.candidate / PACKAGE, target / PACKAGE, ignore=shutil.ignore_patterns("__pycache__"))
    manifest.update(
        csa_source=candidate["csa_source"],
        candidate_provenance=candidate,
        candidate_package_files_at_copy=candidate_files,
        original_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        copied_build_info_sha256=hashlib.sha256(build_info.read_bytes()).hexdigest(),
    )
    files = {
        str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    original_nonmath = {
        name: digest for name, digest in manifest["files"].items() if not name.startswith(PACKAGE + "/")
    }
    assert all(files[name] == digest for name, digest in original_nonmath.items())
    assert {name: digest for name, digest in files.items() if name.startswith(PACKAGE + "/")} == candidate_files
    manifest["files"] = files
    (target.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"source": str(target), "files": len(files), "nonmath_unchanged": True}))


if __name__ == "__main__":
    main()
