# SPDX-License-Identifier: Apache-2.0
"""CPU重建已测内核的ELF供反汇编；先逐字节比对原NPU执行payload，禁止修改SDK。"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from pypto.runtime.elf_parser import extract_text_section
from pypto.runtime.kernel_compiler import KernelCompiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--core", choices=("aic", "aiv"), default="aiv")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.artifact / "binary_manifest.json").read_text())
    kernel = next(
        k for k in manifest["kernels"] if k["name"] == ("cube_schedule" if args.core == "aic" else "vector_schedule")
    )
    source = args.artifact / "kernels" / args.core / (kernel["name"] + ".cpp")
    compiler = KernelCompiler("a2a3")
    isa = compiler.project_root / "build/pto-isa"
    binary = compiler.compile_incore(
        str(source), core_type=args.core, pto_isa_root=str(isa), runtime_name="tensormap_and_ringbuffer"
    )
    payload = extract_text_section(binary)
    original = (args.artifact / kernel["binary"]["path"]).read_bytes()
    assert payload == original, "重建payload与实际NPU已测二进制不同，不准拿来解释旧测量"
    elf = args.output / "kernel.elf"
    elf.write_bytes(binary)
    objdump = Path(compiler.sdk.ccec.cxx_path).with_name("llvm-objdump")
    command = [str(objdump), "--disassemble-aicore", "--demangle", "--line-numbers", str(elf)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    (args.output / "kernel.asm").write_text(result.stdout)
    (args.output / "objdump.stderr").write_text(result.stderr)
    record = {
        "CPU_only": True,
        "NPU_executed": False,
        "source": str(source),
        "core": args.core,
        "original_payload": str(args.artifact / kernel["binary"]["path"]),
        "payload_equal": True,
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "compiler": compiler.sdk.ccec.cxx_path,
        "objdump_command": command,
        "shared_toolchain": json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text()),
    }
    (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("core", "payload_equal", "payload_size")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
