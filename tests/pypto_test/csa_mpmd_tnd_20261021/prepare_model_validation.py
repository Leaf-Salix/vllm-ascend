# SPDX-License-Identifier: Apache-2.0
"""冻结真实16卡测试所需源码；两侧只切TND variant，不复制SDK/权重/KV。"""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REFERENCE = "84751e34cb1f27eff9e4d3fdf03f8495982f1cdf"


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", sys.argv[2]], text=True).strip()
    destination = here / "experiments" / sys.argv[1] / "source"
    destination.mkdir(parents=True, exist_ok=False)
    packages = "vllm_ascend/ops/pypto"
    differences = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--name-only",
            REFERENCE,
            revision,
            "--",
            packages + "/deepseek_v4_flash_csa_spmd_tnd",
            packages + "/deepseek_v4_flash_csa",
        ],
        text=True,
    ).splitlines()
    allowed = {
        packages + "/deepseek_v4_flash_csa_spmd_tnd/service.py",
        packages + "/deepseek_v4_flash_csa_spmd_tnd/service_config.py",
        packages + "/deepseek_v4_flash_csa/nz_mode.py",
        packages + "/deepseek_v4_flash_csa/service_config.py",
    }
    if set(differences) - allowed:
        raise ValueError(f"SPMD核内或数学依赖相对冻结基线有新改动，不能自动更换分母：{differences}")
    mixed = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", revision, "tests/pypto_test/low_acceptance_20261001"],
        text=True,
    ).splitlines()
    paths = [
        "vllm_ascend",
        "tests/pypto_test/offline_pd",
        "tests/pypto_test/dsv4_hca_prepare_opp.py",
        "tests/pypto_test/csa_mpmd_tnd_20261021/model_observer.py",
        "tests/pypto_test/csa_mpmd_tnd_20261021/run_model_validation.sh",
        *[p for p in mixed if p.endswith(".py")],
    ]
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", revision, *paths])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile():
                path = destination / item.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(stream.extractfile(item).read())
    # 使用原环境已经验证的Native二进制，两侧共用；不触发任何额外构建。
    binaries = list((repo / ".cache/csa/native-install").glob("vllm_ascend_C*.so"))
    if len(binaries) != 1:
        raise ValueError(f"Native扩展不唯一：{binaries}")
    (destination / "vllm_ascend" / binaries[0].name).symlink_to(binaries[0])
    vendor = repo / "vllm_ascend/_cann_ops_custom"
    if vendor.exists():
        (destination / "vllm_ascend/_cann_ops_custom").symlink_to(vendor, target_is_directory=True)
    runner = destination / "tests/pypto_test/offline_pd/run.py"
    code = runner.read_text()
    needle = '"worker_extension_cls": "offline_pd.observer.OfflineCSAObserver"'
    assert code.count(needle) == 1
    # 只替换测试observer，保留真实调度、采样、接受逻辑及原图/编译配置。
    runner.write_text(
        code.replace(needle, '"worker_extension_cls": "csa_mpmd_tnd_20261021.model_observer.TNDModelObserver"')
    )
    manifest = {
        "source_commit": revision,
        "reference_math_commit": REFERENCE,
        "reference_service_differences": differences,
        "harness_delta": "offline_pd worker_extension_cls改为TNDModelObserver；计算及配置不改",
        "native_extension": str(binaries[0]),
        "toolchain": json.loads((repo / "tests/pypto_test/csa_shared_toolchain/manifest.json").read_text()),
        "repository": str(repo),
    }
    (destination.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
