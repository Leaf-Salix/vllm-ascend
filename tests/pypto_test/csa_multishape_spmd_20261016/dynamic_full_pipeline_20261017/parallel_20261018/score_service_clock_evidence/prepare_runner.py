"""冻结正式668并对既有Attention诊断入口作最小替换；不导入设备库。"""

import difflib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
MAIN = REPO / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
ORIGIN = MAIN / "parallel_20261018/attention_tick_clock_evidence"
BASE = "66882634"
PACKAGE = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"

HELPERS = '''def check_frozen_source(source):
    """仅接受随入口冻结的正式668源，不能借参数混入其它优化候选。"""
    snapshot = HERE / "source_snapshot"
    names = sorted(path.name for path in snapshot.glob("*.py"))
    assert len(names) == 33
    assert names == sorted(path.name for path in source.glob("*.py"))
    assert all((snapshot / name).read_bytes() == (source / name).read_bytes() for name in names)


def check_active_probe(kernels):
    """wrapper含两个函数；按活动guard检查，不能用整文件判断AIV执行内容。"""
    assert len(kernels) == 2 and {path.parent.name for path in kernels} == {"aic", "aiv"}
    audits = []
    for path in kernels:
        text = path.read_text()
        active = {}
        for role, macro in (("aic", "__DAV_CUBE__"), ("aiv", "__DAV_VEC__")):
            begin, end = f"#if defined({macro})", f"#endif // {macro}"
            assert text.count(begin) == text.count(end) == 1
            first = text.index(begin) + len(begin)
            last = text.index(end, first)
            body = text[first:last]
            marker_count = body.count("CSA_SCORE_PROBE_BEGIN")
            assert marker_count == body.count("CSA_SCORE_PROBE_END")
            assert (marker_count > 0) == (role == "aic")
            assert body.count("CSA_PROFILE_NOTIFY(") > 0
            assert "CSA_ATTN_PROBE_BEGIN" not in body
            active[role] = {"score_markers": marker_count,
                            "phase_notify_calls": body.count("CSA_PROFILE_NOTIFY(")}
        audits.append({"path": str(path), "wrapper_role": path.parent.name,
                       "active_role": active[path.parent.name], "guard_sections": active})
    return audits


'''

REPLACEMENTS = [
    ("冻结d253的Attention逐tick诊断", "冻结668的Score服务诊断"),
    ('parser.add_argument("--baseline-source", type=Path, required=True)',
     'parser.add_argument("--baseline-source", type=Path, default=HERE / "source_snapshot")'),
    ('raise ValueError("本诊断仅开放冻结d253的128K/B16 baseline-only；不扩档")',
     'raise ValueError("本诊断仅开放冻结668的128K/B16 S6 baseline-only；不扩档")\n'
     '    check_frozen_source(args.baseline_source)'),
    ('        baseline_only=args.baseline_only,\n',
     '        baseline_only=args.baseline_only,\n        source_revision="66882634",\n'),
    ('        ffts_meaning="wait_flag_dev原位Scalar起止；A3语义须结合既有核对，仍不把混合阶段区间当纯数学或DMA完成",\n',
     '        clock_meaning="Score原位Scalar端点包含发射/依赖/插桩，不能当纯计算、DMA完成或性能收益",\n'),
    ('        from attention_tick_clock import compilation_probe, export_clock\n',
     '        from score_service_clock import check_probe_padding, compilation_probe, export_clock\n'),
    ('            HERE / "attention_tick_clock.py",\n', '            HERE / "score_service_clock.py",\n'),
    ('                padding = bool((signal[:, :, 1:16] == 0).all() and (signal[:, :, 20:32] == 0).all() and (signal[:, :, 64:] == 0).all())\n',
     '                # 仅放行原phase-clock列16..19及phase34/AIC0..23的列32..63；\n'
     '                # 其它保留位由hook逐项检查，不能沿用Attention全phase放行。\n'
     '                padding = check_probe_padding(signal, generations[name])\n'),
    ('                    protocol=protocol,\n', '                    protocol=protocol,\n                    probe_padding=padding,\n'),
    ('            # 两入口各编译一次，hook 仅在该编译上下文生效；不改共享环境。\n',
     '            # 仅冻结的baseline入口；hook只在此次编译上下文生效，不改共享环境。\n'),
    ('                assert len(kernels) == 2\n'
     '                for kernel in kernels:\n'
     '                    text = kernel.read_text()\n'
     '                    assert text.count("CSA_PROFILE_NOTIFY(") > 2 and "CSA_ATTN_PROBE_BEGIN" in text\n'
     '                    assert "csa_dst[csa_i] = csa_at[csa_i]" in text\n',
     '                report.setdefault("active_probe_audits", {})[name] = check_active_probe(kernels)\n'),
]


def main():
    snapshot = HERE / "source_snapshot"
    runner = HERE / "run_score_diagnostic.py"
    wrapper = HERE / "run_device.sh"
    if any(path.exists() for path in (snapshot, runner, wrapper)):
        raise FileExistsError("冻结入口已存在，拒绝覆盖")
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE, PACKAGE], cwd=REPO, text=True
    ).splitlines()
    paths = [path for path in paths if path.endswith(".py")]
    assert len(paths) == 33 and all(Path(path).parent.as_posix() == PACKAGE for path in paths)
    contents = {Path(path).name: subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO)
                for path in paths}
    assert all(contents[Path(path).name] == (REPO / path).read_bytes() for path in paths)
    snapshot.mkdir()
    for name, data in contents.items():
        (snapshot / name).write_bytes(data)
    original = (ORIGIN / "run_tick_diagnostic.py").read_text()
    text = original
    for old, new in REPLACEMENTS:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    assert text.count("def main():\n") == 1
    text = text.replace("def main():\n", HELPERS + "def main():\n")
    runner.write_text(text)
    shell_original = (ORIGIN / "run_device.sh").read_text()
    assert shell_original.count("run_tick_diagnostic.py") == 1
    wrapper.write_text(shell_original.replace("run_tick_diagnostic.py", "run_score_diagnostic.py"))
    diff = list(difflib.unified_diff(original.splitlines(keepends=True), text.splitlines(keepends=True),
                                   fromfile="attention_tick_clock_evidence/run_tick_diagnostic.py",
                                   tofile="run_score_diagnostic.py"))
    diff.extend(difflib.unified_diff(shell_original.splitlines(keepends=True), wrapper.read_text().splitlines(keepends=True),
                                   fromfile="attention_tick_clock_evidence/run_device.sh", tofile="run_device.sh"))
    (HERE / "runner.diff").write_text("".join(diff))
    record = {
        "base_revision": BASE, "origin_runner": str(ORIGIN / "run_tick_diagnostic.py"),
        "origin_wrapper": str(ORIGIN / "run_device.sh"), "source_snapshot": str(snapshot),
        "source_file_count": 33, "formal_direct_bytes_equal_at_preparation": True,
        "scope": "仅正式668 long128K/B16 S6 baseline-only Score服务诊断，无DFX性能窗",
        "files": [{"path": str(snapshot / name), "bytes": len(data)} for name, data in sorted(contents.items())],
        "pypto_compiled": False, "device_submitted": False,
    }
    (HERE / "SOURCE_PREPARATION.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(runner)


if __name__ == "__main__":
    main()
