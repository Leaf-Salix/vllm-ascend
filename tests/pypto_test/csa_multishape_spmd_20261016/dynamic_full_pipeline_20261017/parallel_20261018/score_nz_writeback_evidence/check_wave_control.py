#!/usr/bin/env python3
"""只用标准库解释同源整数AST，核对动态wave的叶所有权与依赖覆盖。

读取候选源中plan、请求几何、wave/control/dependency函数，去除注解与装饰器后
执行原函数体；显式pl.while_的init/cond/yield转为等价Python while。FakePL只提供整数Scalar、边界检查的Tensor读写及信号记录，不实现
Tile数学、PyPTO lowering、FFTS、DCache或真实并发。合法plan/offsets由原
indexer_group_plan_stage生成，独立枚举原global item%24有工作叶作对照。
Top-K/plan range只提取其真实外层Scalar循环，以记录器替代Tile数学体；该缩减
会在报告明确列出，绝不将整数覆盖PASS称为设备同步/数值/性能PASS。
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

POISON = object()


class Tensor:
    def __init__(self, values, name):
        self.values = list(values)
        self.name = name
        self.reads = 0
        self.writes = 0


class FakePL:
    INDEX = "INDEX"
    INT32 = "INT32"

    def __init__(self):
        self.tensor = SimpleNamespace(dim=self.dim, read=self.read)
        self.system = SimpleNamespace(fence=lambda: self.maintenance.append("fence"),
                                      cacheinvalid=lambda: self.maintenance.append("cacheinvalid"))
        self.maintenance = []

    @staticmethod
    def dim(t, axis):
        assert axis == 0
        return len(t.values)

    @staticmethod
    def read(t, index):
        assert len(index) == 1 and 0 <= index[0] < len(t.values), (t.name, index)
        value = t.values[index[0]]
        assert value is not POISON, ("read uninitialized capacity", t.name, index)
        t.reads += 1
        return value

    @staticmethod
    def write(t, index, value):
        assert len(index) == 1 and 0 <= index[0] < len(t.values), (t.name, index)
        t.values[index[0]] = value
        t.writes += 1

    yield_ = staticmethod(lambda *values: values[0] if len(values) == 1 else values)

    cast = staticmethod(lambda value, dtype: int(value))
    range = staticmethod(range)
    min = staticmethod(min)
    max = staticmethod(max)


class RemoveTypes(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        return self.generic_visit(node)


    def visit_For(self, node):
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                and isinstance(node.iter.func.value, ast.Name) and node.iter.func.value.id == "pl"
                and node.iter.func.attr == "while_"):
            return self.generic_visit(node)
        # 明确解释DSL的init/cond/yield；零次迭代时结果等于init，而不是Python未定义值。
        initial = next(k.value for k in node.iter.keywords if k.arg == "init_values")
        condition = node.body[0]
        assert isinstance(condition, ast.Expr) and isinstance(condition.value, ast.Call)
        assert ast.unparse(condition.value.func) == "pl.cond"
        last = node.body[-1]
        assert isinstance(last, ast.Assign) and ast.unparse(last.value.func) == "pl.yield_"
        target = copy.deepcopy(node.target)
        result = copy.deepcopy(last.targets[0])
        result_load = ast.parse(ast.unparse(result), mode="eval").body
        init_assign = ast.Assign(targets=[copy.deepcopy(target)], value=copy.deepcopy(initial))
        result_init = ast.Assign(targets=[copy.deepcopy(result)], value=copy.deepcopy(initial))
        body = []
        for stmt in node.body[1:]:
            transformed = self.visit(stmt)
            body.extend(transformed if isinstance(transformed, list) else [transformed])
        body.append(ast.Assign(targets=[target], value=result_load))
        loop = ast.While(test=copy.deepcopy(condition.value.args[0]), body=body, orelse=[])
        return [init_assign, result_init, loop]


def read_function(path, name):
    module = ast.parse(path.read_text())
    return copy.deepcopy(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name))


def install_function(env, path, name, inventory):
    function = RemoveTypes().visit(read_function(path, name))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), env)
    inventory.append({"path": str(path), "function": name, "transform": "删除注解/装饰器；显式pl.while_按init/cond/yield语义转Python while，保留零迭代结果"})


def install_range_projection(env, path, name, inventory):
    """只保留原query/first外层range和Scalar前缀，Tile体换成记录函数。"""
    function = RemoveTypes().visit(read_function(path, name))
    loop = next(n for n in function.body if isinstance(n, ast.For))
    prefix = function.body[:function.body.index(loop)]
    if name == "indexer_pair_publish_range":
        loop.body = ast.parse("record_query(query)").body
    else:
        rows = loop.body[0]
        assert isinstance(rows, ast.Assign) and ast.unparse(rows.targets[0]) == "rows"
        loop.body = [rows] + ast.parse("record_plan_rows(first, rows)").body
    function.body = prefix + [loop]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), env)
    inventory.append({"path": str(path), "function": name,
                      "transform": "保留原Scalar前缀/外层range，Tile数学体替换记录器；plan保留原rows=min(8,T-first)"})


def constants_from_config(config):
    tree = ast.parse(config.read_text())
    flash = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "FLASH" for t in n.targets))
    values = {k.arg: ast.literal_eval(k.value) for k in flash.keywords
              if k.arg in ("max_position_embeddings", "index_n_heads")}
    return values


def build_env(source, config):
    pl = FakePL()
    values = constants_from_config(config)
    env = {"pl": pl, "COMPRESS_RATIO": 4, "CANDIDATES_PER_LEAF": 8192,
           "TOPK_CANDIDATES_PER_LEAF": 8192, "SCORE_TILE": 1024,
           "SCORE_WORKERS": 24, "QUERY_GROUP": 6, "LEAF_LANE_CAPACITY": 4096,
           "MAX_CANDIDATES": values["max_position_embeddings"] // 4,
           "MAX_LEAVES": (values["max_position_embeddings"] // 4 + 8191) // 8192,
           "GROUP_DESCRIPTOR_ALIGNMENT": 16, "GROUP_DESCRIPTOR_WORDS": 4,
           "MIN_GROUPS_PER_WAVE": 8, "PLAN_QUERY_ROWS": 8, "SIGNAL_SLOT_WORDS": 128,
           "ROWS": 8}
    inventory = []
    selections = {
        "request_geometry.py": ["request_bounds", "query_request"],
        "indexer_group_stages.py": ["indexer_long_leaf_plan", "indexer_group_plan_stage",
            "indexer_query_visibility", "indexer_decode_group", "indexer_work_item", "indexer_leaf_range"],
        "indexer_publish_stages.py": ["indexer_visible_root_count"],
        "interleave_control.py": ["next_score_leaf", "score_wave_step_count"],
        "wave_boundaries.py": ["next_wave_boundary"],
        "wave_dependencies.py": ["wait_wave_roots", "wait_wave_topk_for_plan", "wait_wave_plan_for_attention"],
    }
    for filename, names in selections.items():
        for name in names:
            install_function(env, source / filename, name, inventory)
    install_range_projection(env, source / "indexer_publish_stages.py", "indexer_pair_publish_range", inventory)
    install_range_projection(env, source / "attention_plan_stages.py", "attention_slot_plan_range", inventory)
    return env, pl, inventory, values


def make_case(name, sizes, histories, *, inactive=(), stale=(), leading=0, trailing=0):
    bounds = [leading]
    for size in sizes:
        bounds.append(bounds[-1] + size)
    tokens = bounds[-1] + trailing
    positions = [-999] * tokens
    lengths = []
    for request, size in enumerate(sizes):
        history = histories[request % len(histories)]
        length = 0 if request in inactive else history + size
        lengths.append(length)
        for query in range(bounds[request], bounds[request + 1]):
            if request not in inactive:
                positions[query] = history + query - bounds[request]
    for query in stale:
        if 0 <= query < tokens:
            positions[query] += 17
    return {"name": name, "bounds": bounds, "lengths": lengths, "positions": positions,
            "tokens": tokens, "sizes": sizes, "inactive": list(inactive), "stale": list(stale)}


def cases():
    result = []
    for s, b, h in [(6, 4, 131072), (6, 8, 131072), (6, 16, 131072),
                    (6, 24, 131072), (6, 28, 131072), (6, 32, 8192),
                    (5, 25, 131072), (5, 9, 8192), (1, 65, 131072), (1, 23, 8192)]:
        result.append(make_case(f"S{s}_B{b}_H{h}", [s] * b, [h]))
    result += [
        make_case("mixed_requests", [1,5,6,7,13,2,11,4,9,3,17,1,6,8,5,14], [0,1,7,8192,131072]),
        make_case("mixed_dummy_gap_stale", [3,6,5,0,7,12,2,1,6,11,4,9,5,8,3,6,7], [131072,8192,7],
                  inactive=(2,3,7,10,14), stale=(8,9,34,35,45), leading=5, trailing=7),
        make_case("all_zero_visible", [1,2,1,2,1,2,1,2,1,2], [0], leading=2, trailing=3),
        make_case("all_positions_stale", [6] * 12, [131072], stale=tuple(range(72))),
        make_case("visible_crosses_1024_and_8192", [6] * 16, [4089,4095,32761,32767]),
        make_case("max_model_history", [6] * 16, [1048570]),
        make_case("empty_T", [], [0]),
        make_case("all_dummy", [3,5,2], [0], inactive=(0,1,2), leading=2, trailing=3),
    ]
    return result


def independent_requests(case, maximum):
    """不调用同源request/group函数的整数oracle，保留inactive与causal定义。"""
    requests = []
    for request, length in enumerate(case["lengths"]):
        begin, end = case["bounds"][request:request+2]
        active = length > 0 and 0 <= begin < end <= case["tokens"] and end - begin <= length
        visible = {}
        if active:
            history = length - (end - begin)
            for query in range(begin, end):
                position = case["positions"][query]
                visible[query] = max(min(length // 4, (position + 1) // 4, maximum), 0) if position == history + query - begin else 0
        requests.append((begin, end, active, visible))
    return requests


def run_case(case, env, pl):
    tensors = {k: Tensor(case[k], k) for k in ("bounds", "lengths", "positions")}
    bounds, lengths, positions = [tensors[k] for k in ("bounds", "lengths", "positions")]
    tokens = case["tokens"]
    descriptor_base = (len(case["lengths"]) + 16) // 16 * 16
    offsets = Tensor([POISON] * (descriptor_base + 4 * tokens), "offsets")
    plan = Tensor([POISON] * 8, "plan")
    env["indexer_group_plan_stage"](bounds, lengths, positions, offsets, plan, tokens, 0, 6, True, 1)
    groups, max_history, leaves, leaf_tiles, extra, group_size, balanced, roots_per_leaf = plan.values
    assert (group_size, balanced, roots_per_leaf) == (6,1,1)
    requests = independent_requests(case, env["MAX_CANDIDATES"])
    descriptors = []
    query_group = {}
    for request, (begin, end, active, visible) in enumerate(requests):
        if active:
            for first in range(begin, end, 6):
                live = min(6, end-first)
                group = len(descriptors)
                descriptors.append((request, first, live, max(visible[q] for q in range(first,first+live))))
                for q in range(first, first+live):
                    query_group[q] = (group, (q-first)//3, visible[q])
    assert groups == len(descriptors)
    for group, desc in enumerate(descriptors):
        assert offsets.values[descriptor_base+4*group:descriptor_base+4*group+4] == list(desc)
    assert all(v is POISON for v in offsets.values[descriptor_base+4*groups:])
    capacities = [(leaf_tiles + (leaf < extra))*1024 for leaf in range(leaves)]
    assert max(capacities) <= 8192
    leaf_starts = []
    start = 0
    for cap in capacities:
        leaf_starts.append(start); start += cap
    expected_items = {}
    all_written_roots = {}
    for group, (request, first, live, visible) in enumerate(descriptors):
        for leaf, (begin, cap) in enumerate(zip(leaf_starts, capacities)):
            valid = max(min(visible-begin,cap),0)
            if valid:
                item = leaf*groups+group if groups < 24 else group*leaves+leaf
                steps = (valid+1023)//1024
                expected_items[item] = (group,leaf,request,first,live,case["lengths"][request]//4,begin,valid,steps,min(steps*512,4096))
                for local in range(live):
                    all_written_roots[first+local,leaf] = 24+(item%24)*2+local//3
    waves = []
    gb = qb = 0
    while gb < groups:
        ge, qe = env["next_wave_boundary"](offsets,lengths,tokens,gb)
        assert gb < ge <= groups and qb <= qe <= tokens
        assert ge == groups or (ge-gb >= 8 and qe%8 == 0)
        assert qb%8 == 0
        for group in range(gb,ge):
            _, first, live, _ = descriptors[group]
            assert qb <= first < first+live <= qe
        waves.append((gb,ge,qb,qe));gb,qb=ge,qe
    if groups:
        assert waves[0][2] == 0 and waves[-1][3] == tokens
        assert len(waves) <= (groups+7)//8 <= (tokens+7)//8
    else:
        assert env["next_wave_boundary"](offsets,lengths,tokens,0) == (0,tokens)
        return {"name":case["name"],"T":tokens,"requests":len(case["lengths"]),"plan":plan.values,
                "waves":[],"status":"PASS-empty-group-fallback", "note":"groups=0必须由外层走原路径，本检查不伪造wave"}
    visited = []
    query_writes = Counter()
    plan_writes = Counter()
    attention_owners = Counter()
    reports = []
    for wave_id,(gb,ge,qb,qe) in enumerate(waves):
        expected_wave = {item:data for item,data in expected_items.items() if gb <= data[0] < ge}
        written_roots = {}
        root_writers = set()
        core_steps = []
        for worker in range(24):
            cursor = worker
            total_steps = 0
            actual_items = []
            while True:
                result = env["next_score_leaf"](bounds,lengths,positions,offsets,plan,cursor,gb,ge)
                next_item,found,group,leaf,request,first,live,cache_len,begin,valid,steps,lane_span = result
                if not found:
                    assert all(v==0 for v in result[2:]);break
                item = next_item-24
                assert item%24 == worker and item >= cursor
                assert expected_wave[item] == (group,leaf,request,first,live,cache_len,begin,valid,steps,lane_span)
                # AIC控制steps与AIV原G6整1024循环数量相同；容量<=8192，无截断隐患。
                assert steps == (valid+1023)//1024 and 1 <= steps <= 8
                actual_items.append(item); visited.append(item); total_steps += steps
                for local in range(live):
                    writer=24+worker*2+local//3
                    written_roots[first+local,leaf]=writer;root_writers.add(writer)
                cursor=next_item
            assert actual_items == sorted(i for i in expected_wave if i%24==worker)
            assert env["score_wave_step_count"](bounds,lengths,positions,offsets,plan,worker,gb,ge)==total_steps
            core_steps.append(total_steps)
        assert written_roots == {key:value for key,value in all_written_roots.items() if qb <= key[0] < qe}

        # 同源wait读取的信号必须由本wave真正写根/TopK/plan的owner发布。
        # 此记录器只检查集合和代际参数，不模拟A3缓存维护或异步完成时刻。
        wait_counts = {}
        duplicates = {}
        def capture(function, arguments, phase, published):
            observed=[]
            def wait(signal, offset, generation):
                assert generation == 7 and offset%128 == 0
                slot=offset//128; actual_phase, producer=divmod(slot,72)
                assert actual_phase==phase and producer in published,(function,phase,producer,published)
                observed.append(producer);return signal
            env["csa_signal_wait"]=wait
            before=len(pl.maintenance)
            env[function](None,*arguments)
            assert pl.maintenance[before:]==["fence","cacheinvalid","fence"]
            return observed

        root_reads=[]
        for worker in range(48):
            expected=[]
            for query in range(qb+(worker-qb)%48,qe,48):
                if query in query_group:
                    group,lane,visible=query_group[query]
                    for leaf,begin in enumerate(leaf_starts):
                        if visible>begin:
                            assert (query,leaf) in written_roots
                            expected.append(written_roots[query,leaf])
            observed=capture("wait_wave_roots",[bounds,lengths,positions,offsets,plan,60+3*wave_id,7,worker,qb,qe],60+3*wave_id,root_writers)
            assert observed==expected,("root waits",worker,observed,expected)
            root_reads.extend(observed)

        topk_writers=set()
        plan_writers=set()
        topk_per_worker={}
        plan_per_worker={}
        for worker in range(48):
            qrows=[]; prows=[]
            env["record_query"]=qrows.append
            env["record_plan_rows"]=lambda first,rows: prows.extend(range(first,first+rows))
            env["indexer_pair_publish_range"](bounds,lengths,positions,plan,None,None,None,qb,qe,worker,48,1,True)
            env["attention_slot_plan_range"](None,positions,bounds,lengths,None,None,None,None,None,0,0,qb,qe,worker,48,False)
            assert qrows==[q for q in range(qb,qe) if q%48==worker]
            assert prows==[q for q in range(qb,qe) if (q//8)%48==worker]
            query_writes.update(qrows);plan_writes.update(prows)
            topk_per_worker[worker]=qrows;plan_per_worker[worker]=prows
            if qrows:topk_writers.add(24+worker)
            if prows:plan_writers.add(24+worker)
        topk_reads=[]
        for worker in range(48):
            observed=capture("wait_wave_topk_for_plan",[60+3*wave_id+1,7,worker,qb,qe],60+3*wave_id+1,topk_writers)
            assert observed==[24+q%48 for q in plan_per_worker[worker]]
            topk_reads.extend(observed)
        plan_reads=[]
        for worker in range(24):
            qrows=[q for q in range(qb,qe) if q%24==worker]
            attention_owners.update(qrows)
            observed=capture("wait_wave_plan_for_attention",[60+3*wave_id+2,7,worker,qb,qe],60+3*wave_id+2,plan_writers)
            assert observed==[24+(q//8)%48 for q in qrows]
            plan_reads.extend(observed)
        for name,reads in (("root",root_reads),("topk",topk_reads),("plan",plan_reads)):
            wait_counts[name]=len(reads)
            duplicates[name]=len(reads)-len(set(reads))
        reports.append({"group_range":[gb,ge],"query_range":[qb,qe],"working_leaves":len(expected_wave),
                        "per_core_steps":core_steps,"root_writers":sorted(root_writers),
                        "signal_read_count":wait_counts,
                        "repeated_producer_references_across_consumers":duplicates})
    assert Counter(visited)==Counter({item:1 for item in expected_items})
    expected_coverage=Counter({query:1 for query in range(tokens)})
    assert query_writes==plan_writes==attention_owners==expected_coverage
    return {"name":case["name"],"status":"PASS","T":tokens,"requests":len(case["lengths"]),
            "plan":plan.values,"max_leaf_capacity":max(capacities),"active_group_count":groups,
            "working_leaf_count":len(expected_items),"dummy_queries":tokens-len(query_group),
            "zero_visible_queries":sum(v[2]==0 for v in query_group.values()),"waves":reports}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(__file__).parent/"source_candidate")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("wave_control_report.json"))
    args=parser.parse_args()
    source=args.source.resolve()
    root=next(p for p in source.parents if (p/"vllm_ascend/ops/pypto/deepseek_v4_flash_csa/config.py").is_file())
    config=root/"vllm_ascend/ops/pypto/deepseek_v4_flash_csa/config.py"
    env,pl,inventory,values=build_env(source,config)
    results=[]
    failure=None
    for case in cases():
        try: results.append(run_case(case,env,pl))
        except Exception as exc:
            failure={"case":case["name"],"type":type(exc).__name__,"detail":str(exc)}
            break
    report={"passed":failure is None,"scope":"标准库同源整数AST及独立所有权oracle；非PyPTO/设备同步证明",
            "source":str(source),"config":{"path":str(config),**values},"functions":inventory,
            "case_count":len(results),"planned_case_count":len(cases()),"cases":results,"failure":failure,
            "checks":["正式group_plan生成plan/offsets与独立合法描述符一致",
                      "每wave/core精确覆盖原global item%24有工作叶，无重复/遗漏",
                      "score_wave_step_count与逐叶AIV1024步数一致，叶容量<=8192",
                      "根wait逐项匹配本wave实际根writer，含stale/dummy/zero-visible",
                      "TopK q%48和plan M8全局owner准确且每T行只写一次",
                      "Attention q%24与plan等待owner对应，无跨waveM8访问",
                      "每wait helper保留fence/cacheinvalid/fence调用顺序"],
            "limitations":["不执行Tile数值与FFTS事件，不证明真实缓存可见性或异步完成",
                           "range仅执行原Scalar循环框架，Tile数学被记录器替代",
                           "groups==0只验证fallback条件，外层旧路径未执行",
                           "重复信号引用是集合计数，跨消费者合计，不是延迟或性能估计",
                           "代表18例不替代先前1235边界检查，不覆盖恶意非单调Q或非法plan"]}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"case_count":len(results),"failure":failure,"output":str(args.output)},ensure_ascii=False))
    if failure:raise SystemExit(1)


if __name__=="__main__":main()
