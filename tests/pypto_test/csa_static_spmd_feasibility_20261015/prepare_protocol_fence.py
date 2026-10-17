# SPDX-License-Identifier: Apache-2.0
"""从原诊断源添加消费者 cacheinvalid 后的 fence，不改数学或组件。"""

import argparse
import ast
import json
import shutil
from pathlib import Path


class FencePoll(ast.NodeTransformer):
    def __init__(self):
        self.insertions = 0

    def visit_Expr(self, node):
        if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != 'pl.system.cacheinvalid':
            return node
        args = node.value.args
        if len(args) != 3 or not isinstance(args[2], ast.List) or len(args[2].elts) != 1:
            return node
        flag = args[2].elts[0]
        if not isinstance(flag, ast.Name) or not flag.id.startswith('flag_'):
            return node
        self.insertions += 1
        return [node, ast.parse('pl.system.fence()').body[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    before = args.source / 'protocol_program.py'
    transformer = FencePoll()
    tree = transformer.visit(ast.parse(before.read_text()))
    ast.fix_missing_locations(tree)
    after = args.output / 'protocol_program.py'
    after.write_text('# 诊断候选：仅消费者 cacheinvalid 后补 fence。\n' + ast.unparse(tree) + '\n')
    shutil.copy2(args.source / 'extraction.json', args.output / 'extraction.json')
    assert transformer.insertions > 0
    (args.output / 'diagnostic_variant.json').write_text(json.dumps({
        'source': str(before), 'poll_fence_insertions': transformer.insertions,
        'component_source_changed': False, 'full_static_source_changed': False,
        'same_records_waits_and_poll_budget': True, 'full_csa_validated': False},
        ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
