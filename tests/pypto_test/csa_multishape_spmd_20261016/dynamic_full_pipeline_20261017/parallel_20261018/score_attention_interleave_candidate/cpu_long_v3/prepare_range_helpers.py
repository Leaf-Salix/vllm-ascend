"""只在候选源追加区间入口，原数学循环体逐字保留。"""
import ast
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "source_candidate"
REF = BASE / "source_reference"
SPECS = [
    ("indexer_publish_stages.py", "indexer_pair_publish_stage", "indexer_pair_publish_range", "    worker: pl.Scalar[pl.INDEX],", "    for query in pl.range(worker, tokens, WORKERS):", "    first_query = query_begin + (worker - query_begin % WORKERS + WORKERS) % WORKERS\n    for query in pl.range(first_query, query_end, WORKERS):"),
    ("attention_plan_stages.py", "attention_slot_plan_stage", "attention_slot_plan_range", "    worker_id: pl.Scalar[pl.INDEX],", "    for first in pl.range(worker_id * ROWS, tokens, WORKERS * ROWS):", "    begin_tile = query_begin // ROWS\n    first_tile = begin_tile + (worker_id - begin_tile % WORKERS + WORKERS) % WORKERS\n    for first in pl.range(first_tile * ROWS, query_end, WORKERS * ROWS):"),
]
records = []
for filename, old, new, marker, old_loop, new_loop in SPECS:
    source = (REF / filename).read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == old)
    lines = source.splitlines(keepends=True)
    definition = "".join(lines[min(d.lineno for d in node.decorator_list)-1:node.end_lineno])
    definition = definition.replace("def " + old + "(", "def " + new + "(", 1)
    definition = definition.replace(marker, "    query_begin: pl.Scalar[pl.INDEX],\n    query_end: pl.Scalar[pl.INDEX],\n" + marker, 1)
    assert definition.count(old_loop) == 1
    definition = definition.replace(old_loop, new_loop, 1)
    parsed = ast.parse(definition).body[0]
    original_loop = next(n for n in node.body if isinstance(n, ast.For))
    candidate_loop = next(n for n in parsed.body if isinstance(n, ast.For))
    assert [ast.dump(n) for n in original_loop.body] == [ast.dump(n) for n in candidate_loop.body]
    note = "\n\n# wave区间入口：只限制query范围，保留全局owner；原逐query/八行tile数学完全一致。\n# 内部边界须8行对齐，末边界可为T；wave完成才能发布聚合phase，不能逐tile发布。\n"
    (SRC / filename).write_text(source + note + definition + "\n")
    records.append(dict(file=filename, function=new, body_ast_unchanged=True))
(BASE / "range_source_audit.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False))
