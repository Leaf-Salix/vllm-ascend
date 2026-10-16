"""只读分析已有 CSA 泳道：核内顺序、数据依赖和静态化所缺的信息。"""

import argparse
import collections
import json
from pathlib import Path


def audit(directory):
    records = json.loads((directory / "chip_swimlane_records.json").read_text())
    deps = json.loads((directory / "deps.json").read_text())
    rows = records["aicore_tasks"]
    # 当前六列是 core、task token、核内派发序号、开始、结束、接收到开始的差值。
    # 第三列不是逻辑 block_idx；不能据它重新绑定计算分片。
    assert rows and all(len(row) == 6 for row in rows)
    nodes = [dict(core=r[0], task=str(r[1]), dispatch_seq=r[2], start=r[3], end=r[4]) for r in rows]
    task_members = collections.defaultdict(list)
    core_members = collections.defaultdict(list)
    for i, node in enumerate(nodes):
        task_members[node["task"]].append(i)
        core_members[node["core"]].append(i)
    pred = collections.defaultdict(set)
    for edge in deps["edges"]:
        if "wait" in edge.get("flags", []):
            pred[edge["succ"]].add(edge["pred"])

    def nearest_compute(task, visiting):
        if task in task_members:
            return {task}
        assert task not in visiting, "原始依赖图存在环"
        return set().union(*(nearest_compute(p, visiting | {task}) for p in pred[task]))

    edges = set()
    task_edges = set()
    timing_violations = []
    for successor in task_members:
        for parent in pred[successor]:
            for producer in nearest_compute(parent, set()):
                task_edges.add((producer, successor))
                for a in task_members[producer]:
                    for b in task_members[successor]:
                        edges.add((a, b))
                        if nodes[a]["end"] > nodes[b]["start"]:
                            timing_violations.append({"producer": producer, "consumer": successor,
                                                      "overlap_cycles": nodes[a]["end"] - nodes[b]["start"]})
    per_core = {}
    for core, members in sorted(core_members.items()):
        members.sort(key=lambda i: nodes[i]["dispatch_seq"])
        for a, b in zip(members, members[1:]):
            edges.add((a, b))
        per_core[str(core)] = [nodes[i] for i in members]
    successor_nodes = collections.defaultdict(list)
    degree = [0] * len(nodes)
    for a, b in edges:
        successor_nodes[a].append(b)
        degree[b] += 1
    ready = collections.deque(i for i, value in enumerate(degree) if value == 0)
    visited = 0
    while ready:
        n = ready.popleft()
        visited += 1
        for other in successor_nodes[n]:
            degree[other] -= 1
            if degree[other] == 0:
                ready.append(other)
    return {
        "source": str(directory), "profiling_level": records["chip_swimlane_level"],
        "physical_members": len(nodes), "compute_submissions": len(task_members),
        "participating_cores": len(core_members), "compute_dependency_pairs": len(task_edges),
        "data_and_per_core_graph_acyclic": visited == len(nodes),
        "dependency_timing_violations": len(timing_violations),
        "max_dependency_overlap_cycles": max((v["overlap_cycles"] for v in timing_violations), default=0),
        "logical_block_index_available": False,
        "record_schema": ["core", "task_token", "per_core_dispatch_seq", "start", "end", "receive_to_start"],
        "per_core_sequences": per_core,
        "compute_dependency_pairs_list": sorted(task_edges),
        "timing_violation_examples": timing_violations[:8],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tests = Path(__file__).resolve().parents[1]
    source = tests / "results/hc_pre_layer_integration_20261012"
    reports = {}
    for level in (1, 4):
        parent = source / f"regression_l{level}_v1/csa_h131072_b4/baseline/dfx"
        for number, path in enumerate((parent, parent / "window_1", parent / "window_2")):
            key = f"L{level}_window_{number}"
            reports[key] = audit(path)
            print(key, json.dumps({k: v for k, v in reports[key].items() if k not in
                  ("per_core_sequences", "compute_dependency_pairs_list", "timing_violation_examples")}, ensure_ascii=False))
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
