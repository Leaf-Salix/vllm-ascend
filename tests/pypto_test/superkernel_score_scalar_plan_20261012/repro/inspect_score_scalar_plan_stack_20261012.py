"""Inspect Score stack usage without drawing a latency conclusion from it."""
import json
import subprocess
from pathlib import Path

from pypto.runtime.kernel_compiler import KernelCompiler
from pypto.runtime.pto_isa import ensure_pto_isa_root
from simpler_setup.elf_parser import extract_text_section

private = Path(__file__).resolve().parent.parent
root = private / "artifacts/csa_score_scalar_plan_128k_b16_20261012"
output = root / "stack_inspection"
output.mkdir(exist_ok=True)
compiler = KernelCompiler("a2a3")
toolchain = compiler.sdk.ccec
isa = Path(ensure_pto_isa_root())
includes = [str(isa / "include"), str(isa / "include/pto")]
includes += compiler.get_incore_include_dirs()
includes += compiler.get_kernel_include_dirs("tensormap_and_ringbuffer")
rows = []
for variant in ("current", "score_scalar", "score_early"):
    for core in ("aic", "aiv"):
        kernel = f"indexer_score_topk_native_pair_{core}"
        source = root / "compiled" / variant / "build/kernels" / core / f"{kernel}.cpp"
        obj = output / f"{variant}_{core}.o"
        command = [toolchain.cxx_path, *toolchain.get_compile_flags(core_type=core),
                   *compiler._include_flags(includes), "-fstack-usage", "-o", str(obj), str(source)]
        result = subprocess.run(command, cwd=compiler.project_root,
                                capture_output=True, text=True, timeout=120)
        obj.with_suffix(".log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        linked = obj.with_suffix(".elf")
        subprocess.run([toolchain.linker_path, "-e", "kernel_entry", "-o", str(linked), str(obj)],
                       capture_output=True, text=True, check=True, timeout=120)
        row = {"variant": variant, "core": core, "kernel": kernel,
               "source": str(source), "command": command,
               "stack_usage": obj.with_suffix(".su").read_text(),
               "linked_text_bytes": len(extract_text_section(linked))}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("command", "source")}), flush=True)
(output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
