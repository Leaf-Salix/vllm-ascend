"""离线核验三臂保留构建后，条件性关联独立 dependency 与 L1 窗口。无设备操作。"""

import argparse
import ast
import collections
import hashlib
import json
import statistics
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parent
ARMS = ("baseline", "barrier", "candidate")
INPUTS = {}


def read(path):
    value = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(value).hexdigest()
    return value


def load(path):
    return json.loads(read(path))


# 与被冻结的 dependency helper 一致：保护注释/字面量/预处理，仅重命名本地 inline 数字标识符。
TOKENS = re.compile(
    r'(?P<protected>(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"'
    r'|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"'
    r"|(?:u8|u|U|L)?'(?:\\.|[^'\\])*'"
    r"|/\*.*?\*/|//[^\n]*|^[ \t]*\#(?:[^\n\\]|\\[^\n]|\\\n)*)"
    r"|(?P<identifier>\b[A-Za-z_]\w*\b)",
    re.DOTALL | re.MULTILINE,
)
INLINE = re.compile(r"[A-Za-z_]\w*inline[0-9]+\w*\Z")
DECL = re.compile(
    r"\b(?:int64_t|int32_t|uint32_t|uint64_t|float|Tensor|TensorCreateInfo|TaskOutputTensors|TaskId)"
    r"(?:\s*&\s*|\s+)([A-Za-z_]\w*inline[0-9]+\w*)\b"
)


def normalize(text):
    assert "CSA_ALPHA_INLINE_" not in text
    protected = []

    def mask(match):
        if match.group("protected") is not None:
            protected.append(match.group())
            return "".join("\n" if c == "\n" else " " for c in match.group())
        return match.group()

    plain = TOKENS.sub(mask, text)
    declared = set(DECL.findall(plain))
    renamed = {m.group() for m in TOKENS.finditer(plain) if m.group("identifier") and INLINE.fullmatch(m.group())}
    assert renamed <= declared, ("non-local identifier", sorted(renamed - declared))
    mapping = {}

    def replace(match):
        word = match.group()
        if match.group("identifier") is None or not INLINE.fullmatch(word):
            return word
        if word not in mapping:
            stem = re.sub(r"inline[0-9]+", "inline", word)
            mapping[word] = f"CSA_ALPHA_INLINE_{len(mapping)}_{stem}"
        return mapping[word]

    return TOKENS.sub(replace, text), protected, mapping


def leaf_maps(manifest):
    return (
        {str(k["func_id"]): k["name"] for k in manifest["kernels"]},
        {str(k["func_id"]): k["binary"]["sha256"] for k in manifest["kernels"]},
    )


def configuration(text):
    """解析两种已知生成配置，不执行模块或导入 simpler。"""
    tree = ast.parse(text)
    values = {}

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [value(n) for n in node.elts]
        if isinstance(node, ast.Dict):
            return {value(k): value(v) for k, v in zip(node.keys, node.values, strict=True)}
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "_D":
            assert node.attr in ("IN", "OUT", "INOUT")
            return node.attr
        if isinstance(node, ast.Name) and node.id == "_ROOT_DIR":
            return ""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = value(node.left), value(node.right)
            assert isinstance(left, str) and isinstance(right, str)
            return left + ("/" if left else "") + right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
            and not node.keywords
        ):
            return value(node.args[0])
        raise AssertionError(("unexpected configuration expression", ast.dump(node)))

    expected_root = ast.parse("_ROOT_DIR = Path(__file__).parent").body[0]
    expected_imports = {"from pathlib import Path", "from simpler.task_interface import ArgDirection as _D"}
    expected_tail = ast.parse("""for _entry in [*KERNELS, ORCHESTRATION]:
    _entry['source'] = str(_ROOT_DIR / _entry['source'])
    _entry['signature'] = [getattr(_D, s) for s in _entry['signature']]
    if 'extra_include_dirs' in _entry:
        _entry['extra_include_dirs'] = [str(_ROOT_DIR / p) for p in _entry['extra_include_dirs']]
""").body[0]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert ast.unparse(node) in expected_imports
        elif isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            name = node.targets[0].id
            if name == "_ROOT_DIR":
                assert ast.dump(node) == ast.dump(expected_root)
            else:
                assert name in ("KERNELS", "ORCHESTRATION", "RUNTIME_CONFIG") and name not in values
                values[name] = value(node.value)
        elif isinstance(node, ast.For):
            assert ast.dump(node) == ast.dump(expected_tail)
        else:
            raise AssertionError(("unexpected configuration statement", ast.dump(node)))
    assert values.keys() == {"KERNELS", "ORCHESTRATION", "RUNTIME_CONFIG"}
    assert all("extra_include_dirs" not in e for e in [*values["KERNELS"], values["ORCHESTRATION"]])
    return values


def local_raw(base, remote):
    # 保留报告中的实际窗口路径，而不是假定不同报告有相同的窗口编号。
    return base / "dfx" / remote.split("/dfx/", 1)[1]


def median_dict(rows):
    assert rows and all(r.keys() == rows[0].keys() for r in rows)
    return {k: statistics.median(r[k] for r in rows) for k in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "QUERY_DEPENDENCY_ANALYSIS.json")
    args = parser.parse_args()
    timing_dir = ROOT / "collected/query-island-v14-run1/results/query-owner-threeway-8k-b32-v1"
    dep_dir = ROOT / "collected/query-dependency-v1/results/query-dependency-8k-b32-v1"
    jit_root = ROOT / "collected/query-v14-timing-jit"
    cpu_root = ROOT / "collected/query-island-v14-cpu"
    timing = load(timing_dir / "result.json")
    dep = load(dep_dir / "result.json")
    assert timing["complete"] and timing["pass"] and timing["timing_executed"]
    assert dep["complete"] and dep["pass"] and dep["dependency_generation"]
    assert not dep["timing_executed"] and not dep["accuracy_executed"]
    assert not timing["performance"]["dep_gen"]
    assert dep["derived_parent_sha256"] == timing["helper_sha"]["probe_perf.py"]
    for key in (
        "manifest_sha256",
        "source_manifest_sha256",
        "runtime77",
        "qa_sha256",
        "tested_sources",
        "true_tokens",
        "lengths",
        "history",
        "layer",
    ):
        assert dep[key] == timing[key], key
    for key in ("source_heads", "binary_sha256", "ptoas_binary_sha256"):
        assert dep["sdk"][key] == timing["sdk"][key], key
    assert timing["explicit_runtime_init_kwargs"] == dep["explicit_runtime_init_kwargs"] == {}
    proof = {}
    for arm in ARMS:
        km = dep["kernel_maps"][arm]
        expected = (km["names"], km["binary_sha256"])
        matches = [p for p in jit_root.glob("*/binary_manifest.json") if leaf_maps(load(p)) == expected]
        assert len(matches) == 1, (arm, matches)
        jit_dir = matches[0].parent
        cpu_dir = cpu_root / ("cpu-" + arm) / "kernel"
        cpu = load(cpu_dir.parent / "receipt.json")
        assert cpu["arm"] == arm and cpu["pass"] and cpu["root_tensors"] == 64
        assert cpu["device_execution"] is False and cpu["npu_initialized"] is False
        assert cpu["helper_manifest_sha256"] == timing["manifest_sha256"]
        assert cpu["candidate_source_sha256"] == hashlib.sha256(read(ROOT / "query-island-v14/SOURCE.json")).hexdigest()
        for key in ("source_heads", "binary_sha256", "ptoas_binary_sha256"):
            assert cpu["sdk"][key] == dep["sdk"][key]
        assert leaf_maps(load(cpu_dir / "binary_manifest.json")) == expected
        (jit_cpp,) = (jit_dir / "orchestration").glob("*.cpp")
        (cpu_cpp,) = (cpu_dir / "orchestration").glob("*.cpp")
        jt, ct = read(jit_cpp).decode(), read(cpu_cpp).decode()
        jn, jp, ji = normalize(jt)
        cn, cp, ci = normalize(ct)
        assert jp == cp and jn == cn and len(ji) == len(ci)
        norm_sha = hashlib.sha256(jn.encode()).hexdigest()
        op = km["orchestration_proof"]
        assert op["alpha_equivalent"] and norm_sha == op["normalized_sha256"]
        assert hashlib.sha256(ct.encode()).hexdigest() == op["cpu_source_sha256"]
        assert read(jit_dir / "compiled_meta.json") == read(cpu_dir / "compiled_meta.json")
        assert configuration(read(jit_dir / "kernel_config.py").decode()) == configuration(
            read(cpu_dir / "kernel_config.py").decode()
        )
        proof[arm] = {
            "retained_timing_build_directory": str(jit_dir.relative_to(ROOT)),
            "per_run_registration_observed": False,
            "leaf_name_funcid_sha_equal_to_dep_and_cpu": True,
            "leaf_count": len(km["names"]),
            "orchestration_alpha_equal": True,
            "normalized_orchestration_sha256": norm_sha,
            "renamed_local_identifiers": len(ji),
            "compiled_meta_byte_equal": True,
            "kernel_config_AST_values_equal": True,
            "kernel_config_note": (
                "Artifact.load序列化改变配置写法；AST白名单逐项核对source/signature/runtime，不声称字节相等。"
            ),
            "timing_orchestration_binary_sha256": load(matches[0])["orchestration"]["binary"]["sha256"],
            "dependency_orchestration_binary_sha256": op["binary_sha256"],
            "binary_orchestration_exact_not_claimed": True,
        }

    mappings, dep_graphs = {}, {}
    for arm in ARMS:
        windows = [w for w in dep["DFX"]["windows"] if w["arm"] == arm]
        assert len(windows) == 2
        graphs = [load(local_raw(dep_dir, w["raw"])) for w in windows]
        maps = [{int(t["task_id"]): t["kernel_ids"] for t in g["tasks"]} for g in graphs]
        assert maps[0] == maps[1]
        mappings[arm], dep_graphs[arm] = maps[0], graphs[0]
    results = {a: [] for a in ARMS}
    for window in timing["performance"]["l1_windows"]:
        arm, repeat = window["arm"], window["repeat"]
        raw = load(local_raw(timing_dir, window["raw"]))
        meta, records = raw["metadata"], raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1
        assert meta["dropped_run_boundaries"] == 0 and len(meta["run_boundaries"]) == 1
        assert meta["core_types"] == ["aic"] * 24 + ["aiv"] * 48
        assert {r[0] for r in records} == set(range(72))
        freq = meta["clock_freq_hz"]
        assert freq > 0
        low, high = min(r[3] for r in records), max(r[4] for r in records)

        def us(value, frequency=freq):
            return value * 1e6 / frequency

        assert abs(us(high - low) - window["us"]) < 1e-8
        funcs, tasks = collections.defaultdict(list), collections.defaultdict(list)
        unique = collections.Counter()
        for core, token, registration, start, end, _ in records:
            assert 0 < start <= end and token in mappings[arm]
            ids = mappings[arm][token]
            vector_ids = {i for i in ids[1:] if i >= 0}
            assert len(vector_ids) <= 1
            fid = ids[0] if core < 24 else next(iter(vector_ids))
            assert fid >= 0
            name = dep["kernel_maps"][arm]["names"][str(fid)]
            funcs[name].append((core, start, end, token))
            tasks[token].append((start, end))
            unique[core, token, registration] += 1
        assert all(n == 1 for n in unique.values())
        functions = {}
        for name, rs in funcs.items():
            per_core = collections.defaultdict(list)
            for core, start, end, token in rs:
                per_core[core].append((start, end))
            durations = [us(end - start) for _, start, end, _ in rs]
            functions[name] = {
                "start_us": us(min(r[1] for r in rs) - low),
                "end_us": us(max(r[2] for r in rs) - low),
                "inclusive_span_us": us(max(r[2] for r in rs) - min(r[1] for r in rs)),
                "max_core_us": max(durations),
                "median_core_us": statistics.median(durations),
                "min_core_us": min(durations),
                "core_start_spread_us": us(max(r[1] for r in rs) - min(r[1] for r in rs)),
                "core_end_spread_us": us(max(r[2] for r in rs) - min(r[2] for r in rs)),
                "records": len(rs),
                "task_tokens": sorted({r[3] for r in rs}),
                "per_core_accumulated_us": {str(k): sum(us(e - s) for s, e in v) for k, v in per_core.items()},
            }
        query_names = (
            [
                "qproj_matmul",
                "qproj_dequant_rms_nope_rope",
                "idx_qr_proj_matmul",
                "idx_qr_dequant_rope",
                "qr_hadamard_matmul",
                "qr_hadamard_quant",
            ]
            if arm == "baseline"
            else ["island_cube", "island_vector"]
        )
        assert all(n in functions for n in query_names)
        score_names = [n for n in functions if n.startswith("indexer_score_topk")]
        assert score_names

        def start_of(ns, values=functions):
            return min(values[n]["start_us"] for n in ns)

        def end_of(ns, values=functions):
            return max(values[n]["end_us"] for n in ns)

        topk = functions["indexer_topk_single_leaf_publish"]["end_us"]
        plan = functions["csa_slots_build_valid_qk_plan"]["end_us"]
        heads = functions["qk_pv_aiv"]["end_us"]
        assert 0 < topk < plan < heads < us(high - low)
        milestones = {
            "whole_us": us(high - low),
            "entry_to_topk_us": topk,
            "entry_to_topk_plan_us": plan,
            "plan_to_heads_us": heads - plan,
            "heads_to_end_us": us(high - low) - heads,
            "query_start_us": start_of(query_names),
            "query_end_us": end_of(query_names),
            "query_inclusive_span_us": end_of(query_names) - start_of(query_names),
            "score_start_us": start_of(score_names),
            "score_end_us": end_of(score_names),
            "score_inclusive_span_us": end_of(score_names) - start_of(score_names),
            "query_end_to_score_start_us": start_of(score_names) - end_of(query_names),
        }
        # 依赖图只能说明等待关系；这里不把所有祖先跨度加成关键路径。
        parents = collections.defaultdict(set)
        for edge in dep_graphs[arm]["edges"]:
            if "wait" in edge["flags"]:
                parents[int(edge["succ"])].add(int(edge["pred"]))
        score_tokens = {r[3] for n in score_names for r in funcs[n]}
        ancestors, todo = set(), list(score_tokens)
        while todo:
            node = todo.pop()
            for pred in parents[node]:
                if pred not in ancestors:
                    ancestors.add(pred)
                    todo.append(pred)

        def labels(token, names=dep["kernel_maps"][arm]["names"], mapping=mappings[arm]):
            return [names[str(i)] for i in mapping.get(token, []) if i >= 0]

        rows = {
            "repeat": repeat,
            "all_recorded_tasks_covered": True,
            "recorded_task_count": len(tasks),
            "milestones": milestones,
            "functions": functions,
            "score_wait_ancestors": {
                str(t): {
                    "names": labels(t),
                    "recorded_end_us": us(max(e for s, e in tasks[t]) - low) if t in tasks else None,
                }
                for t in sorted(ancestors)
            },
        }
        results[arm].append(rows)
    output = {
        "pass": True,
        "scope": (
            "8K/B32/S6/T192/L4真实权重+合成hidden/history，保留构建+dep的条件性离线关联；"
            "不是精确因果证明或整模型/token验收。"
        ),
        "conditional_mapping_only": True,
        "limits": [
            "包含搬运、排队、同步等待的跨度不是纯计算；重叠函数不能相加。",
            "原计时未逐轮保存registration到保留build的直接见证；模块端点依赖其确为实际编排的假设。",
            "三臂都属于我方：baseline=Direct，barrier=完整屏障Query岛，candidate=owner等待Query岛。",
            "未通过性能门槛，因此没有扩测Native精度；不引用旧nalinaly累计cut作本实验内部归因。",
            "每个指标独立取12窗中位数，中位数之差不保证可加。",
        ],
        "identity_proof": proof,
        "arms": {},
        "timing_source": timing["performance"]["summary"],
        "performance_winners": timing["performance"]["winners"],
        "input_sha256": INPUTS,
    }
    output["analysis_script_sha256"] = hashlib.sha256(read(Path(__file__))).hexdigest()
    for arm, rows in results.items():
        assert len(rows) == 12 and {r["repeat"] for r in rows} == set(range(12))
        assert all(r["functions"].keys() == rows[0]["functions"].keys() for r in rows)

        def task_label(token, names=dep["kernel_maps"][arm]["names"], mapping=mappings[arm]):
            return list(dict.fromkeys(names[str(i)] for i in mapping.get(int(token), []) if i >= 0))

        named_edges = [
            {
                "pred": e["pred"],
                "pred_names": task_label(e["pred"]),
                "succ": e["succ"],
                "succ_names": task_label(e["succ"]),
                "argument": e["arg"],
                "source": e["source"],
                "consumer_shape": e.get("consumer_shape"),
            }
            for e in dep_graphs[arm]["edges"]
            if "wait" in e["flags"]
        ]
        output["arms"][arm] = {
            "milestones_medians": median_dict([r["milestones"] for r in rows]),
            "dependency_wait_edges": named_edges,
            "functions_medians": {
                n: median_dict(
                    [{k: v for k, v in r["functions"][n].items() if isinstance(v, (int, float))} for r in rows]
                )
                for n in rows[0]["functions"]
            },
            "windows": rows,
        }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for arm in ARMS:
        print(arm, json.dumps(output["arms"][arm]["milestones_medians"], ensure_ascii=False))


if __name__ == "__main__":
    main()
