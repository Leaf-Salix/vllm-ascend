# SPDX-License-Identifier: Apache-2.0
"""仅为HC与输出链的DMA payload试验较轻的阶段缓存维护，保留完成等待。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    tree = ast.parse((target / "query_phase_sync.py").read_text())
    publish = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "publish_phase"))
    publish.name = "dma_publish_phase"
    publish.body = ast.parse("""
# 先等FIX/MTE3完成，再作DDR fence；不能用AIC分流水fence替代PIPE_ALL。
pl.system.bar_all()
pl.system.fence()
offset = (phase * participant_count + core_id) * SIGNAL_SLOT_WORDS
pl.system.signal_notify(signals, [offset], generation)
return signals
""").body
    wait = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "wait_row_owners"))
    wait.name = "dma_wait_row_owners"
    branch = next(n for n in wait.body if isinstance(n, ast.If))
    assert [ast.unparse(n) for n in branch.body[-3:]] == [
        "pl.system.fence()",
        "pl.system.cacheinvalid()",
        "pl.system.fence()",
    ]
    branch.body[-3:] = ast.parse("pl.system.fence()").body
    imports = ast.parse("""
import pypto.language as pl
from .query_phase_sync import SIGNAL_WORDS, SIGNAL_SLOT_WORDS
from .query_signals import csa_signal_wait
""").body
    prefix = '''# SPDX-License-Identifier: Apache-2.0
"""只用于DMA写回、DMA读取的业务交接；不适用于Scalar metadata payload。

发布保留PIPE_ALL和DDR fence，底层TNOTIFY仍清理信号所在cache line并DSB。
等待保留原150tick协议、owner集合及获取后的fence，仅省去业务DCache全清。
HC的x/inv/partial/raw/pre均由DMA生产消费，Scalar只读固定模型权重。
输出链OA/quant/OB同样由DMA写回；最终HCpost会Scalar读act_scale/post/comb，
它之前的PHASE_OB等待必须保留原完整DCache维护。begin_invocation不变。
不是提前发布完成，也不修改工作区寿命、ACK、数学或全局SDK实现。
"""
'''
    (target / "dma_phase.py").write_text(
        prefix
        + ast.unparse(ast.fix_missing_locations(ast.Module(body=imports + [publish, wait], type_ignores=[])))
        + "\n"
    )
    for filename in ("hc_pipeline.py", "output_group.py"):
        path = target / filename
        module = ast.parse(path.read_text())
        for node in ast.walk(module):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "publish_phase":
                node.func.id = "dma_publish_phase"
            elif node.func.id == "wait_row_owners":
                # HCpost的业务Scalar读需要fresh DCache，保留最终获取屏障。
                if filename == "hc_pipeline.py" or ast.unparse(node.args[1]) != "PHASE_OB":
                    node.func.id = "dma_wait_row_owners"
        module.body.insert(
            1,
            ast.ImportFrom(
                module="dma_phase",
                level=1,
                names=[ast.alias(name="dma_publish_phase"), ast.alias(name="dma_wait_row_owners")],
            ),
        )
        path.write_text(
            "# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(module)) + "\n"
        )
    print(target)


if __name__ == "__main__":
    main()
