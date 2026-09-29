"""Measure backend-created Native graphs without timing the Python wrapper."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-native-superkernel-4ffccb7b"
DEST = WORKSPACE / ".cache/csa-native-graph-replay-4ffccb7b"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one source anchor: {before[:100]}")
    return text.replace(before, after)


def main():
    shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = DEST / "tests/pypto_test/coefficients_seven_experiment/compiled_case.py"
    text = replace_once(
        path.read_text(), "            static_results = []",
        """            from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph

            graph_instances = {}
            original_graph_run = AclGraph.run

            def record_graph_run(owner, key, *inputs, **kwargs):
                graph_instances[(id(owner), key)] = owner
                return original_graph_run(owner, key, *inputs, **kwargs)

            AclGraph.run = record_graph_run
            static_results = []""")
    text = replace_once(
        text, '            report["timing"] = measure_graph_interval(',
        """            if len(graph_instances) != 1:
                raise RuntimeError("Expected one complete npugraph_ex graph for isolated device timing")
            (_, graph_key), owner = next(iter(graph_instances.items()))
            if owner._updated_node_infos:
                raise RuntimeError("Host-updated graph nodes require the backend replay wrapper")
            graph = owner.graph[graph_key]
            # Addresses/shapes are fixed, clone_input/output=False. Retain the graph owner,
            # tensor fixture and module. Check raw replay against this compiled-call result.
            # The first compiled invocation includes backend warmups that mutate cache/state.
            restore(fixture)
            compiled_call()
            torch.npu.synchronize()
            if len(graph_instances) != 1:
                raise RuntimeError("Compiled-call reference unexpectedly created another graph")
            reference = collect_state(fixture, output, topk_value())
            report["measurement"] = {
                "capture_owner": "npugraph_ex",
                "entry": "torch.compile(backend=npugraph_ex)",
                "timed_call": "backend-created NPUGraph.replay",
                "graph_count": len(graph_instances), "host_updated_nodes": 0,
                "reference": "Same captured graph via the compiled callable, identical initial state",
                "scope": "Fixed-address single-CSA device replay; not changing-input model execution",
            }
            report["timing"] = measure_graph_interval(""")
    text = replace_once(text, "                fixture, compiled_call, output, topk_value, reference,",
                        "                fixture, graph.replay, output, topk_value, reference,")
    text = replace_once(text, '            if args.save_state:', """            if any(value["status"] != "PASS"
                   for value in report["timing"]["eager_comparison"].values()):
                raise RuntimeError("Direct replay differs from the same graph through the compiled callable")
            if args.save_state:""")
    compile(text, str(path), "exec")
    path.write_text(text)
    (ROOT / "source.json").write_text(json.dumps({
        "source": str(DEST), "base": str(BASE), "operator_commit": "4ffccb7b",
        "backend": "torch.compile(backend=npugraph_ex)", "super_kernel_optimize": True,
        "cases": [[131072, 16], [8192, 24]], "cann": "9.2.0-beta.2",
        "scope": "Same backend-created graph, isolate device replay from Python dispatch",
    }, indent=2) + "\n")
    original = ROOT.parent / "csa_native_superkernel_20260929/run_side.sh"
    (ROOT / "run_side.sh").write_text(original.read_text().replace(
        "csa_native_superkernel_20260929", "csa_native_graph_replay_20260929"))
    print(DEST)


if __name__ == "__main__":
    main()
