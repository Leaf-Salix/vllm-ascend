# SPDX-License-Identifier: Apache-2.0
"""为静态融合图重新划分主 Q 投影列块，不改点积或输出布局。

原20个逻辑worker为动态编排预留其他任务的位置；融合图有自己的资源表，
可以逐档选择worker数。每列块仍执行原128×256、K256四段INT8点积。
逻辑worker数不同于同时占用的物理核数，后者仍由24个AIC的调度表约束。
"""

import ast
import collections
import copy
import json
from pathlib import Path


def repartition(schedule, catalog, output, workers):
    if workers not in (24, 32, 64, 128):
        raise ValueError("Q投影候选worker数不在当前已分析范围")
    module = ast.parse(Path(catalog["logical_source_ir"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "qproj_matmul")
    loop = next(node for node in fn.body if isinstance(node, ast.For))
    index_assignment = loop.body[0]
    if not isinstance(index_assignment, ast.AnnAssign):
        raise ValueError("主Q投影列块绑定结构改变")
    worker_names = {node.id for node in ast.walk(loop.iter.args[0])
                    if isinstance(node, ast.Name) and node.id.startswith("qproj_worker_")}
    round_names = {node.id for node in ast.walk(loop.target)
                   if isinstance(node, ast.Name) and node.id.startswith("qproj_round_")}
    if len(worker_names) != 1 or len(round_names) != 1:
        raise ValueError("无法唯一识别主Q投影分片变量")
    worker, round_name = next(iter(worker_names)), next(iter(round_names))
    expected_count = ast.parse(f"(19 - {worker} + 128) // 20", mode="eval").body
    expected_column = ast.parse(f"{worker} + {round_name} * 20", mode="eval").body
    if (ast.dump(loop.iter.args[0]) != ast.dump(expected_count)
            or ast.dump(index_assignment.value) != ast.dump(expected_column)):
        raise ValueError("主Q投影不是已核对的128列块/20worker分配")
    original_math = copy.deepcopy(loop.body[1:])
    loop.iter.args[0] = ast.parse(f"({workers - 1} - {worker} + 128) // {workers}", mode="eval").body
    index_assignment.value = ast.parse(f"{worker} + {round_name} * {workers}", mode="eval").body
    if ast.dump(ast.Module(body=loop.body[1:], type_ignores=[])) != ast.dump(
            ast.Module(body=original_math, type_ignores=[])):
        raise ValueError("列分配变换意外改变了点积体")
    columns = collections.Counter(w + r * workers for w in range(workers)
                                  for r in range((workers - 1 - w + 128) // workers))
    if columns != collections.Counter(range(128)):
        raise ValueError("列块遗漏或重复")
    selected = [task for task in schedule["tasks"] if task["kernel_names"][0] == "qproj_matmul"]
    if len(selected) != 1:
        raise ValueError("当前候选要求唯一的主Q投影任务")
    task = selected[0]
    call = schedule["call_bindings"][task["task_id"]]
    if task["block_num"] != 20 or call["block_num"] != 20:
        raise ValueError("原主Q投影worker数不符")
    if not any(isinstance(arg["value"], dict) and arg["value"].get("layout") == "NZ"
               for arg in call["arguments"]):
        raise ValueError("该变换要求生成器从逻辑NZ核体取源码")
    task.update(block_num=workers, member_count=workers)
    call["block_num"] = workers
    audit = {"kernel": "qproj_matmul", "task": task["task_id"], "original_workers": 20,
             "new_workers": workers, "column_blocks_exactly_once": 128,
             "math_body_unchanged": True, "k_tile": 256, "n_tile": 256,
             "physical_aic_limit": 24, "new_cost_is_unmeasured": True,
             "estimated_duration_scale": 20 / workers}
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "logical_qproj_partition.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# 仅改变主Q投影列块的worker分配；同目录audit.json记录范围。\n"
                    + ast.unparse(module) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    updated = copy.deepcopy(catalog)
    updated["logical_source_ir"] = str(path)
    return updated, audit
