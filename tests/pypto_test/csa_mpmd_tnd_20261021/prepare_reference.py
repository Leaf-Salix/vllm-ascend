# SPDX-License-Identifier: Apache-2.0
"""从 Git 冻结参考源码一次，不复制 SDK/权重，不扫描二进制 hash。"""

import io
import subprocess
import tarfile
from pathlib import Path


def prepare_reference():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    revision = "84751e34cb1f27eff9e4d3fdf03f8495982f1cdf"
    destination = here / "sources" / revision[:8]
    if destination.exists():
        return destination
    paths = [
        "vllm_ascend/ops/pypto/deepseek_v4_flash_csa",
        "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd",
        "tests/pypto_test/csa_spmd_tnd_20261020/native_toolchain_manifest.json",
        "tests/pypto_test/csa_spmd_tnd_20261020/dependency_patches",
    ]
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", revision, *paths])
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile():
                output = destination / item.name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(stream.extractfile(item).read())
    return destination


if __name__ == "__main__":
    print(prepare_reference())
