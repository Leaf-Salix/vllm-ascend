# SPDX-License-Identifier: Apache-2.0
"""在 CPU 导出冻结 CSA 的完整 IR，保留各计算体和 orchestration 调用现场。"""

import argparse
import json
import os
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

from pypto.runtime import RunConfig
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.decode_csa import decode_csa_tp1_layer_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source": os.environ["HC_PRE_TEST_OPERATOR_SOURCE"], "device_execution": False}
    try:
        compiled = decode_csa_tp1_layer_test.compile(config=RunConfig(
            platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "kernel"),
            dump_passes=True))
        report.update(code_generation_completed=True, output_dir=str(compiled.output_dir))
    except Exception as error:
        report.update(code_generation_completed=False, error=str(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
