# SPDX-License-Identifier: Apache-2.0
"""冻结完整TND两种调度实现及共享权重适配；不修改公共SDK。"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PACKAGES = ("deepseek_v4_flash_csa", "deepseek_v4_flash_csa_spmd_tnd", "deepseek_v4_flash_csa_tnd")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("快照名称只能包含字母、数字和下划线")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    target = here / "snapshots" / args.name
    target.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in PACKAGES:
        source = repo / "vllm_ascend/ops/pypto" / name
        shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sources[name] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((target / name).glob("*.py"))
        }
    manifest = {
        "name": args.name,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "sources": sources,
        "scope": "TND-SPMD及TND-MPMD的wkv、idx_wq_b、weights_proj原生NZ权重支持",
        "SDK": "csa_shared_toolchain/manifest.json；仅通过公共bootstrap --shared-toolchain使用",
    }
    (here / (args.name + "_source.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
