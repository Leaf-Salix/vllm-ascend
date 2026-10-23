# SPDX-License-Identifier: Apache-2.0
"""Exercise canonical spmd_tnd model preparation, custom-op and framework graph."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/group_service_harness"
    shutil.copytree(here / "runs/measurement_harness", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = (target / "run_pto.py").read_text()
    source = replace_once(source, "or args.state_validation or args.service_validation:", "or args.state_validation:")
    source = replace_once(
        source,
        "    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n"
        '    if not args.service_validation or args.case != "h131072_b16_mixed" or args.cpu or args.compile:\n'
        '        raise ValueError("Service-only C08 validation; performance sampling is disabled")\n',
    )
    source = replace_once(
        source, 'os.environ["PTO_CSA_VARIANT"] = "mpmd_tnd"', 'os.environ["PTO_CSA_VARIANT"] = "spmd_tnd"'
    )
    source = replace_once(
        source,
        '        packages = args.frozen / "pypto"\n',
        '        packages = args.frozen / "pypto"\n        reference_package = "deepseek_v4_flash_csa_spmd_tnd"\n',
    )
    source = replace_once(
        source,
        'candidate_name = mount_package(args.candidate, prefix + "deepseek_v4_flash_csa_tnd")',
        'candidate_name = mount_package(args.candidate, prefix + "deepseek_v4_flash_csa_spmd_tnd")',
    )
    source = replace_once(
        source,
        '        "device_execution": False,\n',
        '        "device_execution": False,\n        "performance_eligible": False,\n',
    )
    begin, end = source.index("            graphs = {}\n"), source.index("    except BaseException:\n")
    source = (
        source[:begin]
        + '            raise AssertionError("Service validation must return before measurement")\n'
        + source[end:]
    )
    assert "pypto.torch.begin_dfx()" not in source and 'report["statistics_us"]' not in source
    (target / "run_pto.py").write_text(source)
    source = (here.parent / "csa_mpmd_tnd_20261021/service_validation.py").read_text()
    source = source.replace('"mpmd_tnd"', '"spmd_tnd"').replace("MPMD", "SPMD").replace("mpmd_tnd", "spmd_tnd")
    source = replace_once(
        source,
        "    prepared = runtime.last_binding\n",
        "    prepared = runtime.last_binding\n"
        "    if not prepared.policy.score_group_order:\n"
        '        raise AssertionError("Canonical service did not select the mixed group-order tier")\n',
    )
    source = replace_once(
        source,
        '        "runtime_type": type(runtime).__module__ + "." + type(runtime).__name__,\n',
        '        "runtime_type": type(runtime).__module__ + "." + type(runtime).__name__,\n'
        '        "policy_constexprs": prepared.policy.constexprs(),\n',
    )
    (target / "service_validation.py").write_text(source)
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    launcher = (here / "launch.sh").read_text()
    launcher = replace_once(
        launcher, 'harness="$here/runs/measurement_harness"', 'harness="$here/runs/group_service_harness"'
    )
    launcher = replace_once(
        launcher, '"$harness/run_pto.py" --family tnd', '"$harness/run_pto.py" --service-validation --family tnd'
    )
    (here / "launch_group_service.sh").write_text(launcher)
    report = dict(
        performance_eligible=False,
        source_validator="../csa_mpmd_tnd_20261021/service_validation.py",
        variant="spmd_tnd",
        candidate="score_group_order_v4",
        checks=[
            "real model preparation",
            "canonical custom-op submission",
            "framework graph replay",
            "workspace reuse",
            "mixed tier selected",
            "empty request handler",
            "Native fallback",
        ],
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "group_service_harness.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
