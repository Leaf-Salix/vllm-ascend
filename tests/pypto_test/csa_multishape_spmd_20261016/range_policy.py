# SPDX-License-Identifier: Apache-2.0
"""历史静态 shape 生成器的范围包装，仅用于复现既有实验。

本文件不会调用目标动态 CSA，也不能为目标动态实现提供验收结论。
虽然能接受区间内新形状，内部仍会把 shape 实值化并展开阶段图。
"""

import json
import sys
from pathlib import Path

from historical_sources import load_historical_module

_policy = load_historical_module("policy")
QUERY_TOKENS = _policy.QUERY_TOKENS
MAX_TOKENS = _policy.MAX_TOKENS
MAX_SEQ_LEN = _policy.MAX_SEQ_LEN
VECTOR_LEAF_LIMIT = _policy.VECTOR_LEAF_LIMIT
MEDIUM_LEAF_LIMIT = _policy.MEDIUM_LEAF_LIMIT
COMPRESS_RATIO = _policy.COMPRESS_RATIO
CORE_GROUPS = _policy.CORE_GROUPS
RangePolicy = _policy.RangePolicy
select_policy = _policy.select_policy
policy_from_descriptors = _policy.policy_from_descriptors


def exact_cost_profile(descriptors, policy, *, evidence_root=None):
    """同规格历史采样仅用于离线排程；缺少样本仍允许生成新 shape。

    不能把相邻 B/H 的实测成本称为当前形状实测。没有完全匹配时，现有生成器
    使用明确标注的参考成本估计，后续可采本规格数据重新调优。
    """
    root = Path(evidence_root or Path(__file__).resolve().parent)
    registry = root / "selected_shape_policies.json"
    if not registry.exists():
        return None
    shapes = {name: desc["shape"] for name, desc in descriptors.items()}
    for row in json.loads(registry.read_text())["rows"]:
        if (row["batch"], row["query_tokens"], row["history"] + row["query_tokens"]) != (
            policy.batch,
            policy.query_tokens,
            policy.max_seq_len,
        ):
            continue
        old_path = Path(row["cost_profile"])
        path = root / (
            Path("per_shape_costs") / old_path.name if old_path.parent.name == "per_shape_costs" else old_path.name
        )
        if not path.exists():
            continue
        profile = json.loads(path.read_text())
        if profile.get("kind") == "same_shape_dynamic_l1_cost" and profile["root_shapes"] == shapes:
            return path
    return None


def build_range_variant(descriptors, *, max_seq_len, output_root, case_name=None, query_tokens=QUERY_TOKENS):
    """历史复现包装：由归档静态展开器生成，测量转换为显式权重。"""
    compiler = load_historical_module("compiler")
    policy = policy_from_descriptors(descriptors, max_seq_len=max_seq_len, query_tokens=query_tokens)
    name = case_name or f"h{max_seq_len - query_tokens}_b{policy.batch}"
    cost = exact_cost_profile(descriptors, policy)
    member_costs = None
    if cost is not None:
        # 旧测量用旧task_id，必须先按本规格原call ordinal/role重映射。
        # 正式生产包不会读此JSON，也不会把旧task_id当作新任务编号。
        frozen = Path(__file__).resolve().parent.parent / "csa_static_spmd_feasibility_20261015"
        reference = json.loads(
            (frozen / "bindings" / f"h{max_seq_len - query_tokens}_b{policy.batch}_prepared.json").read_text()
        )
        profile = json.loads(cost.read_text())
        bound = load_historical_module("binding").bind(
            Path(load_historical_module("catalog").catalog()["logical_source_ir"]),
            {
                k: {field: value for field, value in v.items() if field != "metadata_values"}
                for k, v in descriptors.items()
            },
            max_seq_len=max_seq_len,
        )
        actual = {call["ordinal"]: call for call in bound["tasks"] if call["kind"] == "compute"}
        tasks = {task["task_id"]: task for task in reference["tasks"]}
        member_costs = {}
        for row in profile["task_slots"]:
            old_call = reference["call_bindings"][row["task"]]
            ordinal = old_call["ordinal"]
            slot = ("AIC", "AIV0", "AIV1").index(row["slot"])
            if (
                actual[ordinal]["logical_function"] != old_call["logical_function"]
                or tasks[row["task"]]["kernel_names"][slot] != row["kernel"]
                or row["samples"] <= 0
                or row["median_us"] <= 0
            ):
                raise ValueError("历史测量与当前源码调用合同不匹配")
            key = ordinal, row["slot"]
            if key in member_costs:
                raise ValueError("历史测量含重复调用成员")
            # 旧scheduler使用median+1；在测试侧显式保留，不藏在生产算法里。
            member_costs[key] = row["median_us"] + 1.0
    root = Path(output_root)
    generated = compiler.build_csa(
        descriptors,
        max_seq_len=max_seq_len,
        query_tokens=query_tokens,
        workspace_directory=root / "math_variants" / name,
        core_tensor_capacity=256,
        member_costs=member_costs,
    )
    preparation = generated.preparation
    from validate_static_protocol import validate

    preparation.update(
        anchor="source_derived_task_dag",
        wait_plan=generated.schedule["wait_plan"],
        balanced_schedule=generated.schedule["balanced_schedule"],
        completion_aggregation=None,
        gm_reuse=None,
        descriptors=descriptors,
        production_package=False,
        implementation_kind="historical_shape_specialized_spmd",
        target_dynamic_shape=False,
        cpu_protocol=validate(generated.schedule, generated.layout),
    )
    preparation["range_policy"].update(
        cost_profile=str(cost) if cost else None, cost_is_same_shape_measurement=cost is not None
    )
    case = generated.save(root / "variants" / name)
    (root / "bindings").mkdir(exist_ok=True)
    (root / "bindings" / f"{name}_prepared.json").write_text(
        json.dumps(generated.schedule, ensure_ascii=False, indent=2) + "\n"
    )
    return case, preparation


def configure_device_arguments(parser, args, *, argv=None):
    """范围入口拥有策略选项，阻止命令行叠加旧实验开关造成隐式混合。"""
    allowed = {
        "help",
        "range_policy",
        "batch",
        "history",
        "output",
        "device",
        "warmup",
        "iters",
        "windows",
        "metadata_replay_only",
        "metadata_history",
    }
    explicit = set()
    for token in sys.argv[1:] if argv is None else argv:
        parsed = parser._parse_optional(token)
        if parsed is not None and parsed[0] is not None:
            explicit.add(parsed[0].dest)
    conflicting = [
        action.dest
        for action in parser._actions
        if action.dest not in allowed
        and (action.dest in explicit or getattr(args, action.dest, action.default) != action.default)
    ]
    if conflicting:
        raise ValueError("范围入口不能叠加手工实验策略：" + ", ".join(conflicting))
    policy = select_policy(batch=args.batch, tokens=args.batch * QUERY_TOKENS, max_seq_len=args.history + QUERY_TOKENS)
    for name, value in policy.build_options().items():
        setattr(args, name, value)
    args.cached_wait = True
    args.wait_backoff_ticks = 100
    return policy
