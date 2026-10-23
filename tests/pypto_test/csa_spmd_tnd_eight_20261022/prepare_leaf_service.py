# SPDX-License-Identifier: Apache-2.0
"""Preserve current Native query-bound Host fixes while checking the leaf candidate in service."""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    expected = json.loads((here / "leaf_scale_one_row_v1.json").read_text())["files"]
    source = {p.name: p.read_bytes() for p in (here / "runs/leaf_scale_one_row_v1").glob("*.py")}
    assert {n: hashlib.sha256(s).hexdigest() for n, s in source.items()} == expected
    production = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    for name in ("service.py", "service_config.py"):
        source[name] = (production / name).read_bytes()
    candidate = here / "runs/leaf_scale_one_row_service_v1"
    candidate.mkdir(exist_ok=False)
    for name, content in source.items():
        (candidate / name).write_bytes(content)
    assert all(
        hashlib.sha256(s).hexdigest() == expected[n]
        for n, s in source.items()
        if n not in ("service.py", "service_config.py")
    )
    target = here / "runs/priority_service_harness_v1"
    shutil.copytree(here / "runs/group_service_harness", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("measurement_provenance.py", "expected_sdk.json"):
        shutil.copy2(here / "runs/sdk_measurement_harness_v1" / name, target / name)
    main_source = (target / "run_pto.py").read_text()
    main_source = replace_once(
        main_source, "import types\n", "import types\n\nfrom measurement_provenance import capture_toolchain\n"
    )
    main_source = replace_once(
        main_source, 'args.case != "h131072_b16_mixed"', 'args.case not in ("h131072_b16_s6", "h131072_b24_s6")'
    )
    main_source = main_source.replace("Service-only C08 validation", "Service-only C04/C05 validation")
    main_source = replace_once(
        main_source,
        '        "performance_eligible": False,\n',
        '        "performance_eligible": False,\n'
        '        "shared_toolchain": capture_toolchain(\n'
        '            json.loads(Path(__file__).with_name("expected_sdk.json").read_text())),\n',
    )
    assert "pypto.torch.begin_dfx()" not in main_source
    (target / "run_pto.py").write_text(main_source)
    validator = (target / "service_validation.py").read_text()
    validator = replace_once(
        validator, "not prepared.policy.score_group_order", "not prepared.policy.scale_page_coalescing"
    )
    validator = validator.replace("mixed group-order tier", "continuous-scale leaf proof tier")
    (target / "service_validation.py").write_text(validator)
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    launcher = replace_once(
        (here / "launch_group_service.sh").read_text(),
        'harness="$here/runs/group_service_harness"',
        'harness="$here/runs/priority_service_harness_v1"',
    )
    (here / "launch_priority_service.sh").write_text(launcher)
    reports = {
        "leaf_scale_one_row_service_v1": dict(
            source=str(candidate),
            parent="leaf_scale_one_row_v1",
            adopted=False,
            host_source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            changes=["service.py", "service_config.py"],
            unchanged_kernel_modules=len(source) - 2,
            reason="Preserve current Native max_seqlen_q_actual handling; all kernel modules identical to measured v1",
            files={n: hashlib.sha256(s).hexdigest() for n, s in source.items()},
        ),
        "priority_service_harness_v1": dict(
            performance_eligible=False,
            parent="group_service_harness.json",
            variant="spmd_tnd",
            checks=[
                "canonical model/custom-op",
                "ACLGraphWrapper",
                "workspace reuse",
                "scale tier selected",
                "empty request handler",
                "Native fallback",
            ],
            shared_toolchain=json.loads((target / "expected_sdk.json").read_text()),
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        ),
    }
    for name, report in reports.items():
        with (here / (name + ".json")).open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(candidate)


if __name__ == "__main__":
    main()
