"""Compare callback objects and stack frames without running NPU tasks."""
import json
import subprocess
import sys
from pathlib import Path

from pypto.runtime.kernel_compiler import KernelCompiler
from pypto.runtime.pto_isa import ensure_pto_isa_root
from simpler_setup.elf_parser import extract_text_section

root = Path(__file__).resolve().parent.parent
output = root / "artifacts/tail_call_pinned_objects_20261012"
output.mkdir(exist_ok=True)
compiler = KernelCompiler("a2a3")
toolchain = compiler.sdk.ccec
isa = Path(ensure_pto_isa_root())
sys.path.insert(0, str(root / "pypto/tests/st/runtime/scheduling"))
from task_boundary_thin_entry_probe import rewrite_entry
includes = [str(isa / "include"), str(isa / "include/pto")]
includes += compiler.get_incore_include_dirs()
includes += compiler.get_kernel_include_dirs("tensormap_and_ringbuffer")
source_root = root / "artifacts/csa_event_contract_128k_b16_20261011/compiled/mixed/build/kernels"
summary = []
linked_text = {}
for core, name in (
    ("aiv", "csa_cache_writeback"),
    ("aiv", "indexer_score_topk_native_pair_aiv"),
    ("aic", "_proj_b_mm_nz_kernel__2"),
):
    original = (source_root / core / f"{name}.cpp").read_text()
    old = "        pypto_poll_local_tail(tail_args);\n        return;"
    assert original.count(old) == 1
    for variant in ("original", "musttail", "thin"):
        source = output / f"{name}_{variant}.cpp"
        candidates = {
            "original": original,
            "musttail": original.replace(old, "        [[clang::musttail]] return pypto_poll_local_tail(tail_args);"),
            "thin": rewrite_entry(original),
        }
        source.write_text(candidates[variant])
        obj = source.with_suffix(".o")
        command = [toolchain.cxx_path,
                   *toolchain.get_compile_flags(core_type=core),
                   *compiler._include_flags(includes), "-fstack-usage", "-o", str(obj), str(source)]
        result = subprocess.run(command, cwd=compiler.project_root,
                                capture_output=True, text=True, timeout=120)
        source.with_suffix(".log").write_text(result.stdout + result.stderr)
        row = {"kernel": name, "variant": variant, "returncode": result.returncode, "command": command}
        if result.returncode:
            print(result.stderr[-2500:], flush=True)
        stack = source.with_suffix(".su")
        if stack.exists():
            row["stack_usage"] = stack.read_text()
        if obj.exists():
            symbols = subprocess.run(["readelf", "--wide", "--symbols", str(obj)],
                                     capture_output=True, text=True, check=True)
            source.with_suffix(".symbols.txt").write_text(symbols.stdout)
            row["symbols"] = [line.strip() for line in symbols.stdout.splitlines()
                              if any(name in line for name in ("pypto_poll_local_tail", "kernel_entry", "pypto_execute_local_task"))]
            linked = source.with_suffix(".elf")
            subprocess.run([toolchain.linker_path, "-e", "kernel_entry", "-o", str(linked), str(obj)],
                           capture_output=True, text=True, check=True, timeout=120)
            linked_text[name, variant] = extract_text_section(linked)
            row["linked_text_bytes"] = len(linked_text[name, variant])
        summary.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "command"}), flush=True)
(output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
assert all(row["returncode"] == 0 for row in summary)
comparison = [{
    "kernel": name,
    "musttail_linked_text_equal": linked_text[name, "original"] == linked_text[name, "musttail"],
    "linked_text_bytes": {variant: len(linked_text[name, variant]) for variant in ("original", "musttail", "thin")},
} for name in dict.fromkeys(row["kernel"] for row in summary)]
(output / "linked_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
