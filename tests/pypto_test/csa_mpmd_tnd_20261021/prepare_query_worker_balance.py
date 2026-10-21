# SPDX-License-Identifier: Apache-2.0
"""让Main反量化的head组选择匹配实际AIV数；每个head的数学保持不变。

参考SPMD会话已提交的2ecdf08b/prepare_front_partition.py；该候选已设备
功能通过，但其整链并无收益证明，因此这里仍须单独作MPMD完整L1对照。
"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path


class ActualWorkers(ast.NodeTransformer):
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id == "Q_DEQUANT_WORKERS":
            return ast.copy_location(ast.Name(id="WORKERS", ctx=ast.Load()), node)
        return node


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "query_projection_stages.py"
    source = path.read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    clone = ActualWorkers().visit(copy.deepcopy(fn))
    clone.name = "front_q_dequant_rope_stage"
    # 选择器之后的完整行、尾行循环不做变换；包含FP32乘法、RMS、RoPE和BF16边界。
    old_loop = next(n for n in fn.body if isinstance(n, ast.For))
    new_loop = next(n for n in clone.body if isinstance(n, ast.For))
    assert ast.dump(old_loop, include_attributes=False) == ast.dump(new_loop, include_attributes=False)
    path.write_text(source + "\n\n" + ast.unparse(ast.fix_missing_locations(clone)) + "\n")
    path = target / "front_pipelines.py"
    source = path.read_text()
    old = "q_flat = q_dequant_rope_stage("
    assert source.count(old) == 1
    source = source.replace(old, "q_flat = front_q_dequant_rope_stage(")
    source = source.replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\nfrom .query_projection_stages import front_q_dequant_rope_stage\n",
        1,
    )
    ast.parse(source)
    path.write_text(source)
    print(target)


if __name__ == "__main__":
    main()
