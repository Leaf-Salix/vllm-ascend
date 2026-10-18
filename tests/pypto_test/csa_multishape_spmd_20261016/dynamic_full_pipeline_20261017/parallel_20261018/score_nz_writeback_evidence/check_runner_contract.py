# SPDX-License-Identifier: Apache-2.0
"""标准库检查容量、保留位反例和runner语法；不导入Torch/PyPTO或执行CSA。"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from signal_checks import signal_padding_status, signal_view


class Selection:
    def __init__(self, values):
        self.values = values

    def __eq__(self, other):
        return Selection([value == other for value in self.values])

    def all(self):
        return all(self.values)


class SparseSignal:
    """只模拟被检查的CPU张量索引/比较，用显式非零位置验证真实检查函数。"""
    def __init__(self, phases, entries=()):
        self.shape = (phases, 72, 128)
        self.entries = dict(entries)

    def numel(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def cpu(self):
        return self

    def view(self, *shape):
        assert shape == self.shape
        return self

    def __getitem__(self, indexes):
        selected = []
        for coordinates, value in self.entries.items():
            matches = all(coordinate in range(*index.indices(size)) if isinstance(index, slice)
                          else coordinate == index for coordinate, index, size in zip(coordinates, indexes, self.shape))
            if matches:
                selected.append(value)
        return Selection(selected)


def main():
    here = Path(__file__).resolve().parent
    nodes = ast.parse((here / 'source_candidate/runtime_abi.py').read_text()).body
    capacity_nodes = [node for node in nodes if
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
            (target.id == 'PHASE_COUNT' or target.id.startswith('WAVE_')) for target in node.targets)
        or isinstance(node, ast.FunctionDef) and node.name == 'signal_phase_count']
    namespace = {}
    exec(compile(ast.Module(body=capacity_nodes, type_ignores=[]), '<capacity-source>', 'exec'), namespace)
    capacity = namespace['signal_phase_count']
    for tokens in range(4097):
        phases = capacity(tokens)
        assert phases == 60 + 3 * ((tokens + 7) // 8)
        signal = SparseSignal(phases)
        prepared = SimpleNamespace(tokens=tokens, signal_phase_count=phases, buffers={'signal': signal})
        assert signal_view(prepared) is signal
        assert all(signal_padding_status(signal).values())
    legacy = SimpleNamespace(tokens=96, buffers={'signal': SparseSignal(60)})
    assert signal_view(legacy).shape == (60, 72, 128)
    counterexamples = []
    for index in ((0, 0, 1), (59, 71, 127), (60, 0, 1), (60, 24, 16), (95, 71, 127)):
        status = signal_padding_status(SparseSignal(96, [(index, 7)]))
        assert not all(status.values()), index
        counterexamples.append({'index': index, 'status': status})
    assert all(signal_padding_status(SparseSignal(96, [((0, 0, 0), 3), ((60, 24, 0), 3)])).values())
    try:
        signal_view(SimpleNamespace(tokens=96, signal_phase_count=60, buffers={'signal': SparseSignal(96)}))
    except AssertionError:
        mismatched_capacity_rejected = True
    else:
        raise AssertionError('容量反例未被拒绝')
    parsed = []
    for name in json.loads((here / 'RUNNER_PREPARATION.json').read_text()):
        if name.endswith('.py'):
            ast.parse((here / name).read_text())
            parsed.append(name)
    for name in ('signal_checks.py', 'prepare_runners.py', 'check_runner_contract.py'):
        ast.parse((here / name).read_text())
        parsed.append(name)
    for name in ('runtime_abi.py', 'binding.py'):
        ast.parse((here / 'source_candidate' / name).read_text())
    for name in ('run_device.py', 'output_policy_matrix.py', 'metadata_replay.py', 'padding_replay.py'):
        assert '.view(PHASE_COUNT, 72, 128)' not in (here / name).read_text(), name
    for name in ('check_output_phase_signals.py', 'source_interface.json', 'l1_compare.py'):
        assert (here / name).read_bytes() == (here.parents[1] / name).read_bytes(), name
    result = dict(passed=True, tokens_checked=[0, 4096], old_reference_capacity=60,
                  padding_counterexamples=counterexamples, mismatched_capacity_rejected=mismatched_capacity_rejected,
                  syntax_files=parsed, old_output_phase_checker_unchanged=True, device_execution=False,
                  full_csa_compilation=False, dependency_imports=False)
    (here / 'RUNNER_CONTRACT_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
