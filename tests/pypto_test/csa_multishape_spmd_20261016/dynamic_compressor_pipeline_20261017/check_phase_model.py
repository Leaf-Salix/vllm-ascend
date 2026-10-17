# SPDX-License-Identifier: Apache-2.0
"""解释探针的角色控制流，随机推进阶段事件；不模拟GM/DCache/硬件流水。"""

import ast
import random
from types import SimpleNamespace

from build_probe import FIELDS


def check(here):
    source = ast.parse((here / "probe.py").read_text())
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in source.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    }
    roles = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ("cube", "vector")]
    for node in roles:
        node.decorator_list = []
        for argument in node.args.args:
            argument.annotation = None
    namespace = dict(constants)
    current = {"core": 0, "events": []}

    def event(*parts):
        current["events"].append(parts)

    def begin(signal, core):
        assert core == current["core"]
        event("begin", core)
        return signal, 1

    def publish(signal, phase, core, count, generation):
        assert count == 72 and core == current["core"]
        event("publish", phase, core)
        return signal

    def wait(signal, phase, producer, count, generation):
        assert count == 72
        event("wait", phase, producer)
        return signal

    def wait_rows(signal, phase, count, generation, first, rows, tile, workers, base, stride):
        if rows > 0:
            start = first // tile
            end = (first + rows + tile - 1) // tile
            for index in range(min(end - start, workers)):
                wait(signal, phase, base + ((start + index) % workers) * stride, count, generation)
        return signal

    def work(name, returns):
        def invoke(*args):
            event("work", name)
            values = tuple(args[index] for index in returns)
            return values if len(values) > 1 else values[0]

        return invoke

    def metadata(*args):
        if args[-1] == 0:
            event("work", "main_metadata" if args[0].name == "main_bounds" else "idx_metadata")
        return args[2], args[3]

    def read(tensor, indices):
        assert tensor.name == "idx_count" and indices == [0]
        event("read_count")
        return tensor.value

    namespace.update(
        begin_invocation=begin,
        publish_phase=publish,
        wait_producer=wait,
        wait_row_owners=wait_rows,
        compact_offsets_and_count_stage=metadata,
        compressor_project_stage=work("main_project", [3, 4]),
        indexer_compressor_project_stage=work("idx_project", [3, 4]),
        compressor_pool_stage=work("main_pool", [8]),
        indexer_compressor_pool_stage=work("idx_pool", [8]),
        compressor_state_commit_stage=work("main_commit", [7]),
        indexer_compressor_state_commit_stage=work("idx_commit", [7]),
        compressor_norm_rope_cache_stage=work("main_output", [9, 10, 11]),
        indexer_compressor_norm_rope_stage=work("idx_norm", [8]),
        indexer_compressor_hadamard_stage=work("idx_hadamard", [2]),
        indexer_compact_quant_cache_stage=work("idx_quant", [6, 7, 8]),
        indexer_compact_scale_commit_stage=work("idx_scale", [6]),
    )
    namespace["pl"] = SimpleNamespace(
        func_attr=lambda attributes: None,
        tile=SimpleNamespace(
            get_block_idx=lambda: current["core"] if current["core"] < 24 else (current["core"] - 24) // 2,
            get_subblock_idx=lambda: (current["core"] - 24) % 2,
        ),
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
        cast=lambda value, dtype: int(value),
        INDEX="INDEX",
        read=read,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=roles, type_ignores=[])), "probe_roles", "exec"), namespace)
    scenarios = [(0, 0, 0, 0, 0), (6, 1, 6, 1, 1), (17, 3, 13, 4, 0), (257, 51, 155, 31, 45)]
    scenarios += [(b * 6, b, b * 6, b, b) for b in (4, 8, 16, 24, 28, 32)]
    simulation_count = 0
    total_events = 0
    stage_phase = {
        "main_project": 1,
        "idx_project": 2,
        "main_metadata": 3,
        "idx_metadata": 4,
        "main_pool": 5,
        "main_commit": 6,
        "main_output": 7,
        "idx_pool": 8,
        "idx_commit": 9,
        "idx_norm": 10,
        "idx_hadamard": 11,
        "idx_quant": 12,
        "idx_scale": 13,
    }
    for main_t, main_b, idx_t, idx_b, compact_count in scenarios:
        dimensions = {name: 8192 for _, shape, _, _ in FIELDS for name in shape.split(",") if not name.isdecimal()}
        dimensions.update(MT=main_t, MB=main_b, MQ=main_b + 1, IT=idx_t, IB=idx_b, IQ=idx_b + 1)
        arguments = {
            name: SimpleNamespace(
                name=name,
                shape=[int(a) if a.isdecimal() else dimensions[a] for a in shape.split(",")],
                value=compact_count,
            )
            for name, shape, _, _ in FIELDS
        }
        traces = []
        for core in range(72):
            current.update(core=core, events=[])
            role = "cube" if core < 24 else "vector"
            function = next(n for n in roles if n.name == role)
            namespace[role](*(arguments[a.arg] for a in function.args.args))
            trace = list(current["events"])
            assert sum(item[0] == "begin" for item in trace) == 1
            assert len({item[1:] for item in trace if item[0] == "publish"}) == sum(
                item[0] == "publish" for item in trace
            )
            traces.append(trace)
        main_projects = range(min(24, ((main_t + 63) // 64) * 16))
        idx_projects = range(min(24, ((idx_t + 15) // 16) * 8))
        main_owners = range(24, 24 + min(48, main_b))
        idx_owners = range(24, 24 + min(48, idx_b))
        had_owners = range(min(24, ((compact_count + 15) // 16) * 2))
        for seed in range(8):
            rng = random.Random(seed)
            published = {}
            for generation in range(1, 4):
                indexes, completed = [0] * 72, set()
                while any(indexes[c] < len(traces[c]) for c in range(72)):
                    ready = []
                    for core in range(72):
                        if indexes[core] == len(traces[core]):
                            continue
                        item = traces[core][indexes[core]]
                        if item[0] != "wait" or published.get(item[1:], 0) >= generation:
                            ready.append(core)
                    assert ready, (main_t, main_b, idx_t, idx_b, compact_count, generation, "deadlock")
                    core = rng.choice(ready)
                    item = traces[core][indexes[core]]
                    if item[0] == "work":
                        stage = item[1]
                        dependencies = []
                        if stage == "main_pool":
                            dependencies = [("main_project", p) for p in main_projects]
                        elif stage == "main_commit":
                            dependencies = [("main_pool", p) for p in main_owners]
                        elif stage == "main_output":
                            dependencies = [("main_pool", core), ("main_metadata", 24)]
                        elif stage == "idx_pool":
                            dependencies = [("idx_project", p) for p in idx_projects]
                        elif stage == "idx_commit":
                            dependencies = [("idx_pool", p) for p in idx_owners]
                        elif stage == "idx_norm":
                            dependencies = [("idx_pool", core), ("idx_metadata", 24)]
                        elif stage == "idx_hadamard":
                            dependencies = [("idx_norm", p) for p in idx_owners] + [("idx_metadata", 24)]
                        elif stage == "idx_quant":
                            dependencies = [("idx_hadamard", p) for p in had_owners]
                        elif stage == "idx_scale":
                            assert core == 24
                            dependencies = [("idx_quant", p) for p in idx_owners]
                        elif stage in ("main_metadata", "idx_metadata"):
                            assert core == 24
                        assert all(dependency in completed for dependency in dependencies), (
                            item,
                            core,
                            dependencies - completed if isinstance(dependencies, set) else dependencies,
                        )
                        completed.add((stage, core))
                    elif item[0] == "publish":
                        phase, owner = item[1:]
                        stage = next(s for s, p in stage_phase.items() if p == phase)
                        assert (stage, owner) in completed
                        published[(phase, owner)] = generation
                    elif item[0] == "read_count":
                        assert published.get((4, 24), 0) >= generation
                    indexes[core] += 1
                    total_events += 1
                simulation_count += 1
    return {
        "source_roles_interpreted": True,
        "scenarios": scenarios,
        "random_schedule_series": simulation_count // 3,
        "generation_executions": simulation_count,
        "events": total_events,
        "generations_per_schedule": 3,
        "all_72_begin_exactly_once": True,
        "metadata_only_vector0": True,
        "state_commit_after_all_pool_history_reads": True,
        "scale_after_all_quant_owners": True,
        "count_read_after_metadata_wait": True,
        "no_deadlock_in_model": True,
        "hardware_cache_visibility_proven": False,
    }
