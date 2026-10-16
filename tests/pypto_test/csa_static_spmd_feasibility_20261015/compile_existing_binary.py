# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""复用已有完整代码生成产物，仅编译和组装二进制；不创建 Worker、不运行 NPU。"""

import argparse
import json
import logging
import shutil
import time
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto
import simpler
from pypto.runtime.device_runner import _compile_and_assemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.artifact.resolve()
    target = args.output.resolve() / "kernel"
    target.mkdir()
    # kernel_config.py 使用相对自身的路径；副本避免污染已保存的代码生成证据。
    for directory in ("kernels", "orchestration"):
        for source in (artifact / directory).rglob("*"):
            if source.is_file() and source.suffix in {".cpp", ".h"}:
                output = target / source.relative_to(artifact)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output)
    for name in ("kernel_config.py", "compiled_meta.json"):
        shutil.copy2(artifact / name, target / name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("pypto.runtime.kernel_compiler").setLevel(logging.DEBUG)
    result = {
        "scope": "完整 B4 既有 program artifact 的 CCE/调度二进制编译与 callable 组装",
        "source_artifact": str(artifact),
        "output_artifact": str(target),
        "pypto": str(Path(pypto.__file__).resolve()),
        "simpler": str(Path(simpler.__file__).resolve()),
        "binary_compile_passed": False,
        "callable_assembled": False,
        "device_execution": False,
        "torch_registration_validated": False,
        "full_csa_validated": False,
    }
    started = time.monotonic()
    try:
        _, runtime_name, runtime_config = _compile_and_assemble(target, "a2a3")
        result.update(binary_compile_passed=True, callable_assembled=True,
                      runtime_name=runtime_name, runtime_config=runtime_config)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        result["binaries"] = [{"path": str(path), "bytes": path.stat().st_size}
                              for path in sorted(target.rglob("*"))
                              if path.is_file() and path.suffix in {".o", ".so", ".elf", ".bin"}]
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items()
                          if key != "traceback"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
