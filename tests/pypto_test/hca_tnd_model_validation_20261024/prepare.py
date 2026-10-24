# SPDX-License-Identifier: Apache-2.0
"""冻结已提交框架及本轮HCA适配；共享SDK、权重和离线KV不复制。"""
import argparse
import ast
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from integrate import integrate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", args.revision], text=True).strip()
    experiment = HERE / "experiments" / args.name
    source = experiment / "source"
    source.mkdir(parents=True, exist_ok=False)
    paths = ["vllm_ascend", "tests/pypto_test/offline_pd", "tests/pypto_test/low_acceptance_20261001",
             "tests/pypto_test/csa_shared_toolchain", "tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py",
             "tests/pypto_test/csa_mpmd_tnd_20261021/scenarios.py", "tests/pypto_test/dsv4_hca_prepare_opp.py"]
    archive = subprocess.check_output(["git", "-C", str(REPO), "archive", revision, *paths])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream:
            if member.isfile():
                target = source / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.extractfile(member).read())
    pending = {"dsv4_csa_env", "dsv4_csa_native_case", "dsv4_csa_single_layer", "dsv4_csa_validation"}
    copied = set()
    while pending:
        name = pending.pop()
        if name in copied:
            continue
        text = subprocess.check_output(["git", "-C", str(REPO), "show", f"{revision}:tests/pypto_test/{name}.py"], text=True)
        (source / "tests/pypto_test" / (name + ".py")).write_text(text)
        copied.add(name)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dsv4_"):
                pending.add(node.module)
            elif isinstance(node, ast.Import):
                pending.update(alias.name for alias in node.names if alias.name.startswith("dsv4_"))
    harness = source / HERE.relative_to(REPO)
    harness.mkdir(parents=True)
    for path in HERE.glob("*.py"):
        shutil.copy2(path, harness / path.name)
    binaries = list((REPO / ".cache/csa/native-install").glob("vllm_ascend_C*.so"))
    assert len(binaries) == 1
    (source / "vllm_ascend" / binaries[0].name).symlink_to(binaries[0])
    for name in ("_build_info.py", "_version.py"):
        path = REPO / "vllm_ascend" / name
        if path.is_file():
            shutil.copy2(path, source / "vllm_ascend" / name)
    integrate(source)
    runner = source / "tests/pypto_test/offline_pd/run.py"
    text = runner.read_text()
    old = '"hca": "PyptoHCADeepseekV4ForCausalLM"'
    assert text.count(old) == 1
    text = text.replace(old, '"hca": "PyptoHCATNDDeepseekV4ForCausalLM"')
    old = '"worker_extension_cls": "offline_pd.observer.OfflineCSAObserver"'
    assert text.count(old) == 1
    runner.write_text(text.replace(old, '"worker_extension_cls": "hca_tnd_model_validation_20261024.observer.HCAObserver"'))
    manifest = {"repository": str(REPO), "source_commit": revision, "source": str(source),
                "toolchain": json.loads((REPO / "tests/pypto_test/csa_shared_toolchain/manifest.json").read_text()),
                "HCA": "deepseek_v4_flash_hca_tnd", "CSA": "native", "model_architecture": "PyptoHCATNDDeepseekV4ForCausalLM",
                "native_extension": str(binaries[0]), "graph_replay_required": True,
                "scope": "仅冻结包增加HCA模型及图适配；不修改算法、共享SDK或CSA会话未提交文件"}
    (experiment / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
