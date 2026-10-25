# SPDX-License-Identifier: Apache-2.0
"""冻结当前框架与候选，供已有服务图/CPU合同验证；不改生产或共享SDK。"""

import argparse
import ast
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from prepare import hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("name")
    args = parser.parse_args()
    assert args.name.isidentifier()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    workspace = next(p for p in repo.parents if (p / "env-dsv4-0251rc1.sh").exists())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    source = here / "frozen_service" / args.name / "source"
    source.mkdir(parents=True, exist_ok=False)
    paths = [
        "vllm_ascend",
        "tests/ut/conftest.py",
        "tests/ut/ops/test_dsv4_csa_spmd_tnd.py",
        "tests/ut/ops/test_dsv4_csa_service.py",
        "tests/pypto_test/csa_mpmd_tnd_20261021/scenarios.py",
        "tests/pypto_test/csa_tnd_model_validation_20261023/model_graph_probe.py",
        "tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py",
        "tests/pypto_test/csa_shared_toolchain",
    ]
    archive = subprocess.check_output(["git", "archive", revision, *paths], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile():
                target = source / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.extractfile(item).read())
    pending = {"dsv4_csa_env", "dsv4_csa_native_case", "dsv4_csa_single_layer", "dsv4_csa_validation"}
    copied = set()
    while pending:
        name = pending.pop()
        if name in copied:
            continue
        text = subprocess.check_output(["git", "show", f"{revision}:tests/pypto_test/{name}.py"], cwd=repo, text=True)
        (source / "tests/pypto_test" / (name + ".py")).write_text(text)
        copied.add(name)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dsv4_"):
                pending.add(node.module)
            elif isinstance(node, ast.Import):
                pending.update(alias.name for alias in node.names if alias.name.startswith("dsv4_"))
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    for module in args.candidate.glob("*.py"):
        shutil.copy2(module, source / package / module.name)
    assert hashes(source / package) == hashes(args.candidate)
    primary = workspace / "vllm-ascend-dsv4-pto-0251rc1"
    libraries = list((primary / ".cache/csa/native-install").glob("vllm_ascend_C*.so"))
    assert len(libraries) == 1
    (source / "vllm_ascend" / libraries[0].name).symlink_to(libraries[0])
    for name in ("_build_info.py", "_version.py"):
        original = repo / "vllm_ascend" / name
        if not original.exists():
            original = primary / "vllm_ascend" / name
        assert original.is_file(), original
        shutil.copy2(original, source / "vllm_ascend" / name)
    manifest = {
        "repository": str(repo),
        "framework_revision": revision,
        "candidate": str(args.candidate.resolve()),
        "package_hashes": hashes(args.candidate),
        "native_extension": str(libraries[0]),
        "source": str(source),
        "validation_pending": True,
        "scope": "既有15步真实服务图和CPU合同；不新增整模型EOS测试，不作性能收益",
        "shared_toolchain": json.loads((primary / "tests/pypto_test/csa_shared_toolchain/manifest.json").read_text()),
    }
    (source.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
