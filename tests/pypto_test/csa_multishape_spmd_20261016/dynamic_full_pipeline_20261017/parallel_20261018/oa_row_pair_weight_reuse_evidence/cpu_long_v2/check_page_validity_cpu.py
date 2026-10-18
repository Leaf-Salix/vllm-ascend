# SPDX-License-Identifier: Apache-2.0
"""仅编译页前缀判定probe，先排除测试包装问题，不初始化设备。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import dsv4_csa_env

    dsv4_csa_env.activate()
    from plan_device_probe import page_validity_probe
    from pypto import passes
    from pypto.runtime import CompileOptions

    artifact = page_validity_probe._resolve_kernel_artifact(
        (),
        {"config": CompileOptions(
            platform="a2a3", output_dir=str(args.output / "kernel"),
            memory_planner=passes.MemoryPlanner.PYPTO,
        )},
        allow_signature_mode=True,
    )
    result = dict(passed=True, device_execution=False, artifact=str(artifact.output_dir))
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
