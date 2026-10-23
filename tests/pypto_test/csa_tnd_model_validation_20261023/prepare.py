# SPDX-License-Identifier: Apache-2.0
"""冻结当前框架和测试入口；沿用公共SDK及已安装Native二进制。"""

import argparse
import ast
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--workspace-fix", action="store_true")
    parser.add_argument("--redispatch-fix", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", args.revision], text=True).strip()
    root = here / "experiments" / args.name
    source = root / "source"
    source.mkdir(parents=True, exist_ok=False)
    paths = [
        "vllm_ascend",
        "tests/pypto_test/offline_pd",
        "tests/pypto_test/low_acceptance_20261001",
        "tests/pypto_test/csa_shared_toolchain",
        "tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py",
        "tests/pypto_test/dsv4_csa_env.py",
        "tests/pypto_test/dsv4_hca_prepare_opp.py",
    ]
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", revision, *paths])
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
        text = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:tests/pypto_test/{name}.py"], text=True
        )
        (source / "tests/pypto_test" / (name + ".py")).write_text(text)
        copied.add(name)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dsv4_"):
                pending.add(node.module)
            elif isinstance(node, ast.Import):
                pending.update(alias.name for alias in node.names if alias.name.startswith("dsv4_"))
    overlays = []
    if args.workspace_fix:
        overlays += [
            "vllm_ascend/models/pypto_deepseek_v4.py",
            "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/service.py",
            "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/workspace.py",
        ]
    if args.redispatch_fix:
        overlays += [
            "vllm_ascend/compilation/acl_graph.py",
            "vllm_ascend/ops/dsa.py",
            "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/graph_contract.py",
            "tests/pypto_test/dsv4_csa_native_case.py",
        ]
    for path in overlays:
        shutil.copy2(repo / path, source / path)
    if overlays:
        (root / "workspace.patch").write_text(
            subprocess.check_output(["git", "-C", str(repo), "diff", revision, "--", *overlays], text=True)
        )
    # 只覆盖本轮已知测试改动，其他会话未提交的HCA改动不进入本次验证。
    shutil.copy2(repo / "tests/pypto_test/offline_pd/run.py", source / "tests/pypto_test/offline_pd/run.py")
    harness = source / here.relative_to(repo)
    harness.mkdir(parents=True, exist_ok=True)
    for item in here.iterdir():
        if item.suffix in (".py", ".sh"):
            shutil.copy2(item, harness / item.name)
    binaries = list((repo / ".cache/csa/native-install").glob("vllm_ascend_C*.so"))
    if len(binaries) != 1:
        raise ValueError(binaries)
    (source / "vllm_ascend" / binaries[0].name).symlink_to(binaries[0])
    for name in ("_build_info.py", "_version.py"):
        generated = repo / "vllm_ascend" / name
        if generated.is_file():
            shutil.copy2(generated, source / "vllm_ascend" / name)
    runner = source / "tests/pypto_test/offline_pd/run.py"
    code = runner.read_text()
    old = '"worker_extension_cls": "offline_pd.observer.OfflineCSAObserver"'
    assert code.count(old) == 1
    runner.write_text(
        code.replace(old, '"worker_extension_cls": "csa_tnd_model_validation_20261023.observer.TNDObserver"')
    )
    delta = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--", "tests/pypto_test/offline_pd/run.py"], text=True
    )
    (root / "runner.patch").write_text(delta)
    manifest = {
        "repository": str(repo),
        "source_commit": revision,
        "source_overlays": overlays,
        "harness_delta": "关闭静态编译的独立CLI；只替换观测类，保留真实模型/调度/采样",
        "native_extension": str(binaries[0]),
        "toolchain": json.loads((repo / "tests/pypto_test/csa_shared_toolchain/manifest.json").read_text()),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
