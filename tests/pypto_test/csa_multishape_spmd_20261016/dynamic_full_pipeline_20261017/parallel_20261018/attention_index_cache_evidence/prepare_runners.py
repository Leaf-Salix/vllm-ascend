# SPDX-License-Identifier: Apache-2.0
"""冻结原验证runner的私有副本，只扩展信号回读与隔离overlay入口；不执行CSA。"""

import difflib
import json
from pathlib import Path


def replace_once(text, before, after, count=1):
    actual = text.count(before)
    if actual != count:
        raise AssertionError((before, actual, count))
    return text.replace(before, after)


def main():
    here = Path(__file__).resolve().parent
    base = here.parents[1]
    names = (
        "run_device.py", "check_cpu.py", "output_policy_matrix.py", "source_loader.py",
        "check_output_phase_signals.py", "l1_compare.py", "compile_diagnostics.py",
        "plan_device_probe.py", "address_device_probe.py", "group_device_probe.py",
        "scale_page_device_probe.py", "metadata_replay.py", "padding_replay.py", "source_interface.json",
    )
    modifications = {}
    for name in names:
        original = text = (base / name).read_text()
        if name == "run_device.py":
            text = replace_once(text, "REPO = HERE.parents[3]", "BASE_D = HERE.parents[1]\nREPO = BASE_D.parents[3]\n\nfrom signal_checks import signal_padding_ok, signal_padding_status, signal_view")
            text = replace_once(text, 'parser.add_argument("--spmd-source", type=Path)', 'parser.add_argument("--spmd-source", type=Path, default=HERE / "source_candidate")')
            text = replace_once(text, '    args = parser.parse_args()\n', '    args = parser.parse_args()\n    if args.phase_clock:\n        raise ValueError("交错候选runner保持扩展padding全零；本入口不支持phase-clock插桩")\n')
            text = replace_once(text,
                '        for module in (pypto_core, _torch_npu):\n            assert pypto_root in Path(module.__file__).resolve().parents',
                '        assert pypto_root in Path(pypto.__file__).resolve().parents\n        for module in (pypto_core, _torch_npu):\n            # overlay只复用原二进制符号链接，不能要求其resolve目标位于overlay。\n            expected_binary = pypto_root / "python/pypto" / Path(module.__file__).name\n            assert Path(module.__file__).resolve() == expected_binary.resolve()')
            text = replace_once(text, '            "pypto": pypto_core.__file__,', '            "pypto": pypto_core.__file__,\n            "pypto_python": pypto.__file__,\n            "pypto_binary_resolved": str(Path(pypto_core.__file__).resolve()),')
            text = replace_once(text, '            HERE / "source_loader.py",', '            HERE / "source_loader.py",\n            HERE / "signal_checks.py",\n            HERE / "run_device.sh",')
            text = replace_once(text, 'candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)', 'signal_view(candidate)', count=3)
            text = replace_once(text,
                '                padding_ok = (\n                    (signal[:, :, 1:16] == 0).all() and (signal[:, :, 20:] == 0).all()\n                    if args.phase_clock\n                    else (signal[:, :, 1:] == 0).all()\n                )',
                '                padding_status = signal_padding_status(signal)\n                padding_ok = all(padding_status.values())')
            text = replace_once(text, '                    observed_phase_slots=signal[:, :, 0].tolist(),',
                '                    # 原输出owner检查继续只接收完全相同的60-phase前缀。\n                    observed_phase_slots=signal[:PHASE_COUNT, :, 0].tolist(),\n                    observed_wave_phase_slots=signal[PHASE_COUNT:, :, 0].tolist(),\n                    signal_phase_count=signal.shape[0],\n                    signal_padding=padding_status,')
            text = replace_once(text, '(signal[:, :, 1:] == 0).all()', 'signal_padding_ok(signal)')
        elif name == "check_cpu.py":
            text = replace_once(text, '    parser.add_argument("--spmd-source", type=Path)', '    parser.add_argument("--spmd-source", type=Path, default=Path(__file__).resolve().parent / "source_candidate")')
            text = replace_once(text, 'from pathlib import Path', 'from pathlib import Path\n\nfrom cpu_failure_report import record_compilation_failure')
            text = replace_once(text, '        report.update(error=str(exc), traceback=traceback.format_exc())', '        report.update(record_compilation_failure(output, exc))')
        elif name == "output_policy_matrix.py":
            text = replace_once(text, 'from l1_compare import export_window', 'from l1_compare import export_window\nfrom signal_checks import signal_padding_ok, signal_padding_status, signal_view')
            text = replace_once(text, '"源码矩阵尚不允许改变信号ABI"', '"源码矩阵必须保留原60-phase语义前缀"')
            text = replace_once(text, 'candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)', 'signal_view(candidate)')
            text = replace_once(text, 'item.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)', 'signal_view(item)', count=2)
            text = replace_once(text, '(signal[:, :, 1:] == 0).all()', 'signal_padding_ok(signal)', count=2)
            text = replace_once(text, '            report["protocol"][name] = dict(passed=passed, before=before[name], calls=calls[name])',
                '            report["protocol"][name] = dict(\n                passed=passed, before=before[name], calls=calls[name],\n                phase_count=signal.shape[0], padding=signal_padding_status(signal))')
        elif name in ("metadata_replay.py", "padding_replay.py"):
            text = replace_once(text, 'from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.runtime_abi import PHASE_COUNT',
                'from signal_checks import signal_padding_ok, signal_view')
            text = replace_once(text, 'candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)', 'signal_view(candidate)', count=2)
            var = 'signals' if name == 'metadata_replay.py' else 'signal'
            text = replace_once(text, f'({var}[:, :, 1:] == 0).all()', f'signal_padding_ok({var})')
        (here / name).write_text(text)
        modifications[name] = {'source': str(base / name), 'bytes': len(text.encode()), 'changed': text != original}
    diff = []
    for name in names:
        diff.extend(difflib.unified_diff((base / name).read_text().splitlines(True),
            (here / name).read_text().splitlines(True), fromfile='original_runner/' + name, tofile='candidate_runner/' + name))
    for name in ('runtime_abi.py', 'binding.py'):
        diff.extend(difflib.unified_diff((here / 'source_reference' / name).read_text().splitlines(True),
            (here / 'source_candidate' / name).read_text().splitlines(True), fromfile='source_reference/' + name, tofile='source_candidate/' + name))
    (here / 'RUNNER_CHANGES.diff').write_text(''.join(diff))
    (here / 'RUNNER_PREPARATION.json').write_text(json.dumps(modifications, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
