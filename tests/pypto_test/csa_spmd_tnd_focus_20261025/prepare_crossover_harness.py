# SPDX-License-Identifier: Apache-2.0
"""Freeze preparation-order crossover without changing measurement boundaries."""

import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare import replace_once

ORDER_HELPER = '''
def preparation_sides(order):
    """Preparation order never selects the numerical golden."""
    if order == "candidate-first":
        return ("candidate", "reference")
    if order == "reference-first":
        return ("reference", "candidate")
    raise ValueError("Unknown preparation order: " + order)

'''


def transform(source):
    source = replace_once(source, "\ndef main():\n", ORDER_HELPER + "\ndef main():\n")
    source = replace_once(
        source,
        "    args = parser.parse_args()\n",
        '    parser.add_argument("--prepare-order", choices=("candidate-first", "reference-first"), '
        'default="candidate-first")\n'
        "    args = parser.parse_args()\n"
        "    prepare_sides = preparation_sides(args.prepare_order)\n",
    )
    source = replace_once(
        source,
        '        "candidate_init_layout": args.candidate_init_layout,\n',
        '        "candidate_init_layout": args.candidate_init_layout,\n'
        '        "prepare_order": args.prepare_order,\n'
        '        "golden_side": "reference",\n'
        '        "preparation_trace": {"package_mount": [], "binding": [], '
        '"compile_and_eager": [], "graph_capture": []},\n',
    )
    source = replace_once(
        source,
        '            baseline_name = mount_package(packages / reference_package, prefix + "_mpmd_tnd_reference")\n'
        '            candidate_name = mount_package(args.candidate, prefix + "_mpmd_tnd_candidate") '
        'if args.family == "tnd" else prefix + "deepseek_v4_flash_csa"\n',
        "            package_specs = {\n"
        '                "reference": (packages / reference_package, prefix + "_mpmd_tnd_reference"),\n'
        '                "candidate": (args.candidate, prefix + "_mpmd_tnd_candidate"),\n'
        "            }\n"
        "            mounted = {}\n"
        "            for side in prepare_sides:\n"
        "                mounted[side] = mount_package(*package_specs[side])\n"
        '                report["preparation_trace"]["package_mount"].append(side)\n'
        '            baseline_name, candidate_name = mounted["reference"], mounted["candidate"]\n',
    )
    cpu_start = source.index("        if args.cpu:\n")
    cpu_end = source.index("\n        import pypto.torch\n", cpu_start)
    cpu = """        if args.cpu:
            report["cpu_policies"], report["cpu_prepare_trace"] = {}, []
            package_names = {"reference": baseline_name, "candidate": candidate_name}
            for side in prepare_sides:
                module = importlib.import_module(package_names[side] + ".entry")
                policy = importlib.import_module(package_names[side] + ".policy").select_policy(
                    tokens=case["token_capacity"],
                    requests=case["request_capacity"],
                    max_query_tokens=max_query_tokens,
                    max_seq_len=max(case["seq_lens"]),
                )
                module.decode_csa_spmd._get_dep_graph()
                report["cpu_policies"][side] = policy.constexprs()
                save("source_dependencies_parsed_" + side)
                if args.compile:
                    module.decode_csa_spmd.compile(**policy.constexprs(), config=options)
                report["cpu_prepare_trace"].append(side)
            report["policy"] = report["cpu_policies"]["candidate"]
            report["passed"] = True
            save("cpu_backend_compiled" if args.compile else "cpu_parse_passed")
            return
"""
    source = source[:cpu_start] + cpu + source[cpu_end:]
    source = replace_once(
        source,
        '            variants.append(("candidate", candidate_name))\n',
        '            variants.append(("candidate", candidate_name))\n'
        "            variant_packages = dict(variants)\n"
        "            variants = [(side, variant_packages[side]) for side in prepare_sides]\n",
    )
    source = replace_once(
        source,
        '            report["policies"] = {name: entry.policy.constexprs() for name, entry in entries.items()}\n',
        '                report["preparation_trace"]["binding"].append(side)\n'
        '            report["policies"] = {name: entry.policy.constexprs() for name, entry in entries.items()}\n',
    )
    source = replace_once(
        source,
        '            expected, tables, report["checks"] = None, {}, {}\n',
        '            eager_snapshots, tables, report["checks"] = {}, {}, {}\n',
    )
    source = replace_once(
        source,
        "                state, check = snapshot(side)\n"
        "                if expected is None:\n"
        "                    expected = state\n"
        "                numerical = compare(state, expected)\n",
        "                eager_snapshots[side] = snapshot(side)\n"
        '                report["preparation_trace"]["compile_and_eager"].append(side)\n'
        "                save()\n"
        "            # Both snapshots are complete; preparation order cannot choose the golden.\n"
        '            expected = eager_snapshots["reference"][0]\n'
        '            for side in ("reference", "candidate"):\n'
        "                state, check = eager_snapshots[side]\n"
        "                numerical = compare(state, expected)\n",
    )
    source = replace_once(
        source,
        "                    graphs[side] = graph\n",
        "                    graphs[side] = graph\n"
        '                    report["preparation_trace"]["graph_capture"].append(side)\n',
    )
    ast.parse(source)
    return source


def launcher(source):
    source = source.replace('"$here/frozen_v2/run_pto.py"', '"$here/frozen_v3/run_pto.py"')
    source = source.replace('--frozen "$here/frozen_v2"', '--frozen "$here/frozen_v3"')
    start = source.index('candidate=$(realpath "${1:?')
    end = source.index('source "$workspace/env-dsv4-0251rc1.sh"', start)
    arguments = """candidate=$(realpath "${1:?冻结算子目录}")
case_name=${2:?固定场景}
attempt=${3:?新输出名称}
layout=${4:?初始化布局separate或fused}
shift 4
prepare_order=candidate-first
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare-order)
            [[ $# -ge 2 ]] || exit 2
            prepare_order=$2
            shift 2
            ;;
        --device)
            [[ $# -ge 2 && "$2" == "${TASK_DEVICE:?通过task-submit提交}" ]] || exit 2
            shift 2
            ;;
        *) exit 2 ;;
    esac
done
[[ "$attempt" =~ ^[a-zA-Z0-9_]+$ && "$case_name" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$layout" == separate || "$layout" == fused ]] || exit 2
[[ "$prepare_order" == candidate-first || "$prepare_order" == reference-first ]] || exit 2
: "${TASK_DEVICE:?通过task-submit提交}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || exit 2
"""
    source = source[:start] + arguments + source[end:]
    return replace_once(
        source,
        '--candidate-init-layout fused --device "$TASK_DEVICE"',
        '--candidate-init-layout "$layout" --prepare-order "$prepare_order" --device "$TASK_DEVICE"',
    )


def main():
    here = Path(__file__).resolve().parent
    old, new = here / "frozen_v2", here / "frozen_v3"
    shutil.copytree(old, new, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(here / "complete_l1_v2.py", new / "complete_l1_v2.py")
    before = (old / "run_pto.py").read_text()
    after = transform(before)
    (new / "run_pto.py").write_text(after)
    (here / "launch_crossover.sh").write_text(launcher((here / "launch_fused_init.sh").read_text()))
    (here / "crossover_harness_v3.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="frozen_v2/run_pto.py",
                tofile="frozen_v3/run_pto.py",
            )
        )
    )
    report = {
        "parent": str(old),
        "target": str(new),
        "default_prepare_order": "candidate-first",
        "supported_prepare_orders": ["candidate-first", "reference-first"],
        "golden": "always frozen reference eager snapshot; compare only after both sides are captured",
        "reversed_phases": ["package mount", "binding/allocation", "first compilation and eager", "graph capture"],
        "unchanged": [
            "fixture",
            "restore sites",
            "snapshot/bitwise/guard checks",
            "three alternating warmups",
            "ten alternating paired windows",
            "begin_dfx/replay/end_dfx boundaries",
        ],
        "raw_validator": "snapshot of current complete_l1_v2.py; separate reference, explicit candidate layout",
        "device_execution": False,
        "validation_status": "CPU_contracts_pending",
        "adopted": False,
    }
    (here / "crossover_harness_v3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(new)


if __name__ == "__main__":
    main()
