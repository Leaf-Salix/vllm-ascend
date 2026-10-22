# SPDX-License-Identifier: Apache-2.0
"""复用已验证单卡runner，显式限定新规范的场景/采样和两个BSH入口。"""

import ast
import shutil


def change(text, old, new, count=1):
    assert text.count(old) == count, old
    return text.replace(old, new)


def prepare_runner(here, repo):
    previous = repo / "tests/pypto_test/csa_mpmd_tnd_20261021"
    source = here / "sources"
    text = (previous / "run_case.py").read_text()
    text = change(
        text,
        'parser.add_argument("--case", required=True)',
        'parser.add_argument("--case", required=True)\n'
        '    parser.add_argument("--family", choices=("tnd", "bsh"), default="tnd")',
    )
    text = change(text, 'packages = args.frozen / "vllm_ascend/ops/pypto"', 'packages = args.frozen / "pypto"')
    text = change(
        text,
        'baseline_name = mount_package(packages / "deepseek_v4_flash_csa_spmd_tnd", prefix + "_mpmd_tnd_reference")',
        'reference_package = "deepseek_v4_flash_csa_spmd_tnd" '
        'if args.family == "tnd" else "deepseek_v4_flash_csa_spmd"\n'
        '            baseline_name = mount_package(packages / reference_package, prefix + "_mpmd_tnd_reference")',
    )
    text = change(
        text,
        'candidate_name = mount_package(args.candidate, prefix + "_mpmd_tnd_candidate")',
        'candidate_name = mount_package(args.candidate, prefix + "_mpmd_tnd_candidate") '
        'if args.family == "tnd" else prefix + "deepseek_v4_flash_csa"',
    )
    text = change(
        text,
        'native_bind = importlib.import_module(baseline_name + ".native_adapter").bind_native_inputs',
        'native_bind = importlib.import_module(prefix + "_input_contract.native_adapter").bind_native_inputs',
    )
    text = change(
        text,
        'report["sources"] = {',
        'mount_package(packages / "deepseek_v4_flash_csa_spmd_tnd", prefix + "_input_contract")\n'
        '        report["family"] = args.family\n'
        '        report["sources"] = {',
    )
    text = change(
        text, '"spmd": str(packages / "deepseek_v4_flash_csa_spmd_tnd"),', '"spmd": str(packages / reference_package),'
    )
    old = """                entries[side] = importlib.import_module(package + ".binding").PreparedDynamicCSA(
                    native,
                    max_query_tokens=max_query_tokens,
                    max_seq_len=max(case["seq_lens"]),
                    allocator=allocate,
                )"""
    new = """                if args.family == "bsh" and side == "mpmd":
                    if len(set(case["query_lengths"])) != 1 or case["query_lengths"][0] != 6:
                        raise ValueError("BSH入口只测原S6，不能用TND绑定冒充混长")
                    from bsh_entry import BSHEntry
                    entries[side] = BSHEntry(adapter, weights, fixture, groups, output, native)
                else:
                    entries[side] = importlib.import_module(package + ".binding").PreparedDynamicCSA(
                        native, max_query_tokens=max_query_tokens,
                        max_seq_len=max(case["seq_lens"]), allocator=allocate,
                    )"""
    text = change(text, old, new)
    text = change(text, "for sample in range(5):", "for sample in range(10):")
    text = change(
        text, '"min": min(samples),', '"median": statistics.median(samples),\n                    "min": min(samples),'
    )
    marker = '            report["windows"] = records'
    text = change(
        text,
        marker,
        """            if args.family == "tnd":
                from outer_profile import profile_mpmd
                save("mpmd_outer_profile")
                report["outer_profile"] = profile_mpmd(fixture, graphs["mpmd"], output, args.output)
                state, check = snapshot("mpmd")
                numerical = compare(state, expected)
                report["checks"]["mpmd_post_outer"] = {"guards": check, "numerical": numerical}
                if not all(check.values()) or not all(v["status"] == "PASS" for v in numerical.values()):
                    raise AssertionError("外层profile后输出/state/保护检查失败")
"""
        + marker,
    )
    ast.parse(text)
    (source / "run_pto.py").write_text(text)
    shutil.copyfile(previous / "scenarios.py", source / "scenarios.py")
    for name in ("bsh_entry.py", "outer_profile.py", "cases.json", "manifest.json"):
        shutil.copyfile(here / name, source / name)
    shutil.copyfile(repo / "tests/pypto_test/csa_five_way_latest_20261020/csa_measure.py", source / "csa_measure.py")
    # Native沿原静态编译/SuperKernel/多流入口；仅用固定query列表替换场景构造。
    text = (repo / "tests/pypto_test/csa_five_way_latest_20261020/native_case.py").read_text()
    text = change(
        text, "TESTS = HERE.parent", f'TESTS = __import__("pathlib").Path({str(repo / "tests/pypto_test")!r})'
    )
    text = change(
        text,
        "parser.add_argument('--history', type=int, required=True)",
        "parser.add_argument('--history', type=int, required=True)\n    parser.add_argument(\"--case\", required=True)",
    )
    text = change(
        text,
        "args = parser.parse_args()",
        "args = parser.parse_args()\n"
        '    case = next(c for c in json.loads((HERE / "cases.json").read_text())["cases"] if c["name"] == args.case)\n'
        '    assert (args.history, args.batch) == (case["history_label"], case["active_requests"])\n'
        '    tokens = case["token_capacity"]',
    )
    text = text.replace("args.batch * 6", "tokens")
    text = change(
        text,
        "fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device, query_tokens=6)",
        "from scenarios import make_fixture as make_tnd_fixture\n"
        "            fixture = make_tnd_fixture(config, layer.self_attn, args.batch, args.history, 1024, device,\n"
        '                                       query_lengths=case["query_lengths"], '
        'history_lengths=case["history_lengths"],\n'
        "                                       token_capacity=tokens, request_capacity=args.batch)\n"
        '            report["case"] = case',
    )
    ast.parse(text)
    (source / "run_native.py").write_text(text)
    native_source = here / "native_sources_v2"
    native_source.mkdir()
    for name in ("run_native.py", "scenarios.py", "cases.json"):
        shutil.copyfile(source / name, native_source / name)
    shutil.copyfile(repo / "tests/pypto_test/csa_five_way_latest_20261020/measure.py", native_source / "measure.py")
    for dirname in ("bsh_sources_v2", "bsh_short_sources"):
        destination = here / dirname
        destination.mkdir()
        for name in ("run_pto.py", "scenarios.py", "outer_profile.py", "cases.json", "manifest.json", "csa_measure.py"):
            shutil.copyfile(source / name, destination / name)
        shutil.copyfile(here / "bsh_entry.py", destination / "bsh_entry.py")
        if dirname == "bsh_short_sources":
            shutil.copytree(source / "pypto", destination / "pypto")
