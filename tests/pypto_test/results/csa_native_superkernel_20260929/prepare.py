"""Freeze a Native-only direct npugraph_ex A/B without modifying queued sources."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-coefficients-seven-4ffccb7b"
DEST = WORKSPACE / ".cache/csa-native-superkernel-4ffccb7b"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one source anchor: {before[:100]}")
    return text.replace(before, after)


def main():
    shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helper = DEST / "tests/pypto_test/dsv4_csa_single_layer.py"
    text = helper.read_text()
    start = text.index("def measure_graph_interval(")
    end = text.index("\ndef run(args, report):", start)
    measure = text[start:end]
    measure = replace_once(measure, """    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
""", "    # npugraph_ex owns capture/replay; do not nest a manual outer graph.\n")
    if measure.count("graph.replay()") != 2:
        raise ValueError("Unexpected measurement replay sites")
    measure = measure.replace("graph.replay()", "run()")
    helper.write_text(text[:start] + measure + text[end:])
    runner = DEST / "tests/pypto_test/coefficients_seven_experiment/compiled_case.py"
    text = runner.read_text()
    decorator = '@support_torch_compile(dynamic_arg_dims={"hidden_states": 0, "positions": 0, "output": 0})'
    text = replace_once(text, decorator + "\nclass CompiledNativeHalf", "class CompiledNativeHalf")
    text = replace_once(text, 'choices=("native", "pto")', 'choices=("native",)')
    text = replace_once(text, '    parser.add_argument("--save-state", action="store_true")',
                        '    parser.add_argument("--save-state", action="store_true")\n'
                        '    parser.add_argument("--super-kernel", type=int, choices=(0, 1), required=True)')
    text = replace_once(text, '        "status": "RUNNING", "side": args.side,',
                        '        "super_kernel": bool(args.super_kernel),\n'
                        '        "compile_entry": "torch.compile(backend=npugraph_ex)",\n'
                        '        "status": "RUNNING", "side": args.side,')
    text = replace_once(text, "            static_results = []", """            static_results = []
            static_super_flags = []
            report["super_kernel_graph_calls"] = []
            original_super = torch.npu.NPUGraph.super_kernel_optimize

            def observe_super(graph, *inputs, **kwargs):
                record = {"status": "STARTED"}
                report["super_kernel_graph_calls"].append(record)
                result = original_super(graph, *inputs, **kwargs)
                record["status"] = "RETURNED_SUCCESSFULLY"
                return result

            torch.npu.NPUGraph.super_kernel_optimize = observe_super""")
    text = replace_once(
        text, "                success = original_compile(*inputs, **kwargs)", """                import inspect

                bound = inspect.signature(original_compile).bind_partial(*inputs, **kwargs)
                flag = bool(kwargs.get("super_kernel_optimize",
                                       bound.arguments.get("super_kernel_optimize", False)))
                static_super_flags.append(flag)
                if flag != bool(args.super_kernel):
                    raise RuntimeError("Static compiler did not receive the requested superkernel flag")
                success = original_compile(*inputs, **kwargs)""")
    text = replace_once(
        text, "            compiled = cls(vllm_config=config, layer=layer)",
        """            module = cls(vllm_config=config, layer=layer)
            options = {
                "force_eager": False, "inplace_pass": False,
                "clone_input": False, "clone_output": False,
                "static_kernel_compile": True,
                "super_kernel_optimize": bool(args.super_kernel),
            }
            report["backend_options"] = options
            report["compilation_scope"] = {
                "vllm_ascend_fx_passes": False,
                "note": "Direct npugraph_ex passes; EngineArgs alone does not apply vLLM FX passes",
            }
            # The named backend is mandatory; no vLLM compilation wrapper or fallback.
            torch.npu.set_compile_mode(jit_compile=False)
            compiled = torch.compile(module, backend="npugraph_ex", fullgraph=True,
                                     dynamic=False, options=options)""")
    text = replace_once(text, '                "wrapper_compiled": bool(compiled.compiled or '
                        'compiled.was_aot_compile_fn_loaded_from_disk),\n'
                        '                "fresh_compile_flag": compiled.compiled,\n'
                        '                "aot_loaded_from_disk": compiled.was_aot_compile_fn_loaded_from_disk,',
                        '                "wrapper_compiled": bool(static_results),\n'
                        '                "entry": "torch.compile(backend=npugraph_ex)",\n'
                        '                "static_super_flags": static_super_flags,')
    text = replace_once(
        text, '            if args.save_state:', """            calls = report["super_kernel_graph_calls"]
            if bool(calls) != bool(args.super_kernel):
                raise RuntimeError("npugraph_ex did not apply the requested graph optimization")
            if any(call["status"] != "RETURNED_SUCCESSFULLY" for call in calls):
                raise RuntimeError("Superkernel graph optimization did not finish")
            if args.save_state:""")
    text = replace_once(text, '            report["weights"] = weights', """            report["weights"] = weights
            report["multistream"] = {
                "dsa_overlap": layer.self_attn.dsa_attn.dsa_attn.impl.multistream_dsv4_dsa_overlap,
                "expression": "torch.npu.stream + record_event/wait_event/wait_stream",
                "boundary": "Native dsa_forward custom op; stream operations execute during graph capture",
                "graph_owner": "npugraph_ex (force_eager=False)",
                "shared_expert": "configured but outside the attention-half measurement",
            }
            if not report["multistream"]["dsa_overlap"]:
                raise RuntimeError("Native DSA multistream overlap must remain enabled")""")
    compile(text, str(runner), "exec")
    runner.write_text(text)
    (ROOT / "source.json").write_text(json.dumps({
        "source": str(DEST), "base": str(BASE), "operator_commit": "4ffccb7b",
        "backend": "torch.compile(backend=npugraph_ex)",
        "cases": [[131072, 16], [8192, 24]], "cann": "9.2.0-beta.2",
        "scope": "Native static-kernel versus static+superkernel; npugraph_ex owns capture and replay",
    }, indent=2) + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
