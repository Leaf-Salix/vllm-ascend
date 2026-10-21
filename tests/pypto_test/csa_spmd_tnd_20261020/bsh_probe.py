# SPDX-License-Identifier: Apache-2.0
"""Use the accepted BSH wait policy and audit actual CCEC source for both sides.

The callback is confined to this test process. Native backoff lowering is accepted
without rewriting it; the original BSH uses its frozen 150-tick source transform.
Production service code never installs this callback.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import runpy
import sys
from pathlib import Path


def check_toolchain(manifest_path):
    if manifest_path is None and os.environ.get("CSA_SHARED_TOOLCHAIN_MANIFEST"):
        manifest_path = Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"])
    if manifest_path is None:
        return {"passed": True, "scope": "Frozen BSH toolchain checked by launcher"}
    manifest = json.loads(manifest_path.read_text())
    if "release" in manifest:
        workspace = Path(os.environ["PTO_EAGER_ROOT"])
        expected = {
            "CSA_STATIC_PYPTO_ROOT": workspace / manifest["pypto"]["source"],
            "PTOAS_ROOT": workspace / manifest["ptoas"]["build"],
            "HCA_SIMPLER_ROOT": workspace / manifest["simpler"]["source"],
        }
        for name, root in expected.items():
            if not os.environ.get(name) or Path(os.environ[name]).resolve() != root.resolve():
                raise RuntimeError(f"公共 CSA 工具链路径不一致：{name}")
        return {"passed": True, "manifest": str(manifest_path), "release": manifest["release"],
                "scope": "shared source/build paths; loaded artifacts checked separately"}
    mismatches = [
        path
        for path, expected in manifest["files"].items()
        if not Path(path).is_file()
        or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
    ]
    if mismatches:
        raise RuntimeError(f"Native toolchain changed after freeze: {mismatches}")
    return {"passed": True, "manifest": str(manifest_path), "file_count": len(manifest["files"])}


def runtime_artifacts(*, require_loaded=False):
    from pypto.backend._ptoas_locate import find_ptoas_binary

    paths = {"ptoas": Path(find_ptoas_binary()).resolve()}
    pypto_dir = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]) / "python/pypto"
    simpler_dir = Path(os.environ["HCA_SIMPLER_ROOT"]) / ".venv/lib"
    expected = {
        "pypto.pypto_core": list(pypto_dir.glob("pypto_core*.so")),
        "pypto._torch_npu": list(pypto_dir.glob("_torch_npu*.so")),
        "_task_interface": list(simpler_dir.glob("python*/site-packages/_task_interface*.so")),
    }
    for name, matches in expected.items():
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely identify the configured artifact for {name}: {matches}")
        path = matches[0].resolve()
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != path:
            raise RuntimeError(f"{name} was loaded from a different toolchain")
        if require_loaded and loaded is None:
            raise RuntimeError(f"Expected runtime extension was not loaded: {name}")
        paths[name] = path
    return {
        name: {"path": str(path), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wait-policy-source", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolchain-manifest", type=Path)
    parser.add_argument("--wait-first-fast-polls", type=int, default=0)
    args, forwarded = parser.parse_known_args()
    if args.wait_first_fast_polls not in (0, 4):
        raise ValueError("This diagnostic supports only fixed150 or four initial fast polls")
    check_toolchain(args.toolchain_manifest)
    if "--raw-wait" in forwarded:
        raise ValueError("The accepted BSH performance baseline uses 150-tick backoff")
    import pypto

    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    loaded_artifacts = runtime_artifacts()
    parent_name = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd"
    parent = importlib.import_module(parent_name)
    name = parent_name + ".wait_codegen"
    spec = importlib.util.spec_from_file_location(name, args.wait_policy_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent.wait_codegen = module
    if module.WAIT_BACKOFF_TICKS != 150:
        raise ValueError("The frozen accepted BSH baseline must use 150 ticks")
    original = module.rewrite_wait_source

    def rewrite_or_accept_native(source):
        if "PTOAS__WAIT_GE_BACKOFF<" in source:
            ticks = set(re.findall(r"PTOAS__WAIT_GE_BACKOFF<(\d+)>", source))
            if ticks != {"150"} or "pto::comm::TWAIT(" in source:
                raise ValueError("Native TND source mixes unexpected wait policies")
            if args.wait_first_fast_polls:
                # Diagnostic only: preserve every DCCI/volatile read, the spin
                # bound and periodic barrier. Only the first four failed reads
                # skip the fixed delay. BSH's legacy helper is never changed.
                old = '''            const uint64_t wait_start = get_sys_cnt();
            while ((uint64_t)(get_sys_cnt() - wait_start) < BackoffTicks) {
                __asm__ __volatile__("" ::: "memory");
            }'''
                if source.count(old) != 1:
                    raise ValueError("Native wait helper changed; refuse an approximate rewrite")
                new = ("            // CSA_EXPERIMENT_FAST_POLLS=4\n"
                       "            if (spin >= 4) {\n"
                       + "\n".join("    " + line for line in old.splitlines())
                       + "\n            }")
                source = source.replace(old, new)
            return source, dict(
                native_lowering=True, backoff_syscnt_ticks=150,
                source_rewritten=bool(args.wait_first_fast_polls),
                initial_fast_polls=args.wait_first_fast_polls,
            )
        return original(source)

    module.rewrite_wait_source = rewrite_or_accept_native
    sys.argv = [str(args.probe_script), "--output", str(args.output), *forwarded]
    try:
        runpy.run_path(str(args.probe_script), run_name="__main__")
    finally:
        result_path = args.output / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            files = list((args.output / "build_output").glob("decode_csa_spmd_*/kernels/*/*.cpp"))
            audit = []
            for path in files:
                source = path.read_text()
                native = "PTOAS__WAIT_GE_BACKOFF<150>(" in source
                legacy = "CSA_WAIT_LOCAL_GM(" in source and "get_sys_cnt() - csa_wait_start) < 150)" in source
                fast_polls_present = "// CSA_EXPERIMENT_FAST_POLLS=4" in source and "if (spin >= 4)" in source
                audit.append(
                    dict(
                        path=str(path),
                        native_lowering=native,
                        legacy_rewrite=legacy,
                        initial_fast_polls=(4 if fast_polls_present else 0),
                        passed=(native or legacy)
                        and fast_polls_present == bool(native and args.wait_first_fast_polls)
                        and "pto::comm::TWAIT(" not in source
                        and "volatile __gm__ int32_t*" in source
                        and "dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);" in source,
                    )
                )
            result["bsh_wait_policy_audit"] = {
                "passed": bool(audit) and all(row["passed"] for row in audit),
                "ticks": 150,
                "actual_ccec_sources": audit,
                "scope": (
                    "BSH fixed150; native TND fixed150 unless the separately recorded fast-poll experiment is enabled"
                ),
            }
            result["wait_policy_experiment"] = (
                {"initial_fast_polls": args.wait_first_fast_polls,
                 "production_supported": False,
                 "scope": "Test-only native helper rewrite; not eligible for final eight-case acceptance"}
                if args.wait_first_fast_polls else None
            )
            result["toolchain_post_run_audit"] = check_toolchain(args.toolchain_manifest)
            result["runtime_artifacts"] = runtime_artifacts(require_loaded=True)
            if result["runtime_artifacts"] != loaded_artifacts:
                raise RuntimeError("Loaded compiler/runtime artifacts changed while the task was running")
            result["passed"] = result["passed"] and result["bsh_wait_policy_audit"]["passed"]
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            if not result["passed"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
