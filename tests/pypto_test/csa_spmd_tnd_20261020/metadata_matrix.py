# SPDX-License-Identifier: Apache-2.0
"""Same-address TND graph and continuous-state validation with Native metadata.

Original single-SPMD is only the no-regression reference. Native diagnostics and
independent C4/write-boundary checks remain separate; fixed-S6 multi-task is unused.
State branches retain their own previous-step cache, never reset to initial state
inside the continuous decode sequence.
"""

import json

import torch
from dsv4_csa_single_layer import guard_checks, make_fixture
from dsv4_csa_validation import compare_tensor, compare_topk
from metadata_replay import METADATA_ARGUMENTS

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.dsv4_csa import _native_attention_half
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.runtime_abi import PHASE_COUNT


def validate(
    config,
    layer,
    fixture,
    candidate,
    baseline,
    allocations,
    options,
    output_path,
    history,
    table_history,
    *,
    continuous_only=False,
    b1_only=False,
    plan_diagnostic=None,
):
    root = candidate.native
    capacity = fixture["tokens"]
    requests = fixture["request_capacity"]
    if b1_only and (capacity != 6 or requests != 1 or fixture["query_lengths"] != [6]):
        raise ValueError("B1矩阵入口要求一个S6请求，T容量6，请求容量1")
    required_requests = 6 + (capacity - 6 + 5) // 6
    if not b1_only and (
        not 41 <= capacity <= 72 or requests != required_requests or fixture["query_lengths"] != [6] * 6
    ):
        raise ValueError("本矩阵需6个S6请求及足够槽位：6 + ceil((T容量 - 6) / 6)，覆盖S1的dummy尾部")
    if table_history is None or table_history < history + 128:
        raise ValueError("本矩阵需至少history+128的固定页表容量")
    addresses = {name: tensor.data_ptr() for name, tensor in candidate.buffers.items()}
    initial = {name: group["initial"] for name, group in fixture["groups"].items()}
    result = {
        "passed": False,
        "metadata_cases": [],
        "continuous_steps": [],
        "scope": "同址单SPMD图；Native语义/写界与旧单SPMD无新增回归分开判定",
        "performance_measured": False,
    }

    def save():
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def new_case(queries, histories, owners=None, reverse=False):
        case = make_fixture(
            config,
            layer.self_attn,
            len(queries),
            max(histories),
            1024,
            root["x_hc"].device,
            table_history=table_history,
            query_lengths=queries,
            history_lengths=histories,
            token_capacity=capacity,
            request_capacity=requests,
            page_owners=owners,
            reverse_pages=reverse,
            allow_unassigned_tail=b1_only,
        )
        case["hidden"].copy_(fixture["hidden"])
        groups = {name: (case["metadata"][group["prefix"]], group["views"]) for name, group in case["groups"].items()}
        source = bind_native_inputs(
            root,
            case["hidden"],
            case["positions"],
            groups,
            layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
            compact_metadata=case["compact"],
            output=root["x_out"],
            topk=root["idx_topk"],
            topk_scores=root["idx_topk_scores"],
        )
        metadata = {}
        for name in METADATA_ARGUMENTS:
            actual, target = source[name], root[name]
            if actual.dtype != target.dtype or actual.ndim != target.ndim or actual.shape[1:] != target.shape[1:]:
                raise ValueError(f"元数据描述符不兼容：{name} {actual.shape} / {target.shape}")
            if actual.shape[0] > target.shape[0]:
                raise ValueError(f"固定根元数据容量不足：{name}")
            # Native compact outputs can have a shorter capacity. Only the test
            # loader copies them into the captured root; production has no copy.
            value = torch.zeros(target.shape, dtype=target.dtype)
            if name.endswith("slot_mapping"):
                value.fill_(-1)
            if "freqs_cos" in name:
                value.fill_(1)
            value[: actual.shape[0]].copy_(actual.cpu())
            metadata[name] = value
        case["root_metadata"] = metadata
        current_policy = select_policy(
            tokens=capacity,
            requests=requests,
            max_query_tokens=6,
            max_seq_len=max(h + q for h, q in zip(histories, queries)),
        )
        if current_policy != candidate.policy:
            raise ValueError("用例跨越Host策略边界，须通过服务重新分派，不能硬重放旧图")
        return case

    def load(case, caches):
        for name, value in case["root_metadata"].items():
            root[name].copy_(value)
        for name, value in caches.items():
            fixture["groups"][name]["allocation"].copy_(value)
        root["x_out"].fill_(float("nan"))
        root["idx_topk"].fill_(-777)
        root["idx_topk_scores"].fill_(float("nan"))

    def collect(case):
        count = case["actual_tokens"]
        return {
            "output": root["x_out"][:count].cpu(),
            "topk": root["idx_topk"][:count].cpu(),
            "scores": root["idx_topk_scores"][:count].cpu(),
            **{name: group["allocation"].cpu() for name, group in fixture["groups"].items()},
        }

    def checked_guards(case, before):
        view = {
            "groups": {
                name: {
                    "allocation": group["allocation"],
                    "initial": before[name],
                    "allowed": case["groups"][name]["allowed"],
                }
                for name, group in fixture["groups"].items()
            }
        }
        checks = guard_checks(view)
        checks.update(
            {
                "metadata." + name: compare_tensor(root[name], value, 0, 0)
                for name, value in case["root_metadata"].items()
            }
        )
        return checks

    def geometry(case):
        prefix, offsets = 0, []
        for q, start in zip(case["query_lengths"], case["history_lengths"]):
            offsets.append(prefix - start // 4 - 1)
            prefix += sum((pos + 1) % 4 == 0 for pos in range(start, start + q))
        offsets += [0] * (requests - len(offsets))
        checks = {}
        for branch in ("main", "idx"):
            checks[branch + ".offsets"] = compare_tensor(
                candidate.buffers[branch + "_offsets"], torch.tensor(offsets, dtype=torch.int32), 0, 0
            )
            checks[branch + ".count"] = compare_tensor(
                candidate.buffers[branch + "_count"], torch.tensor([prefix], dtype=torch.int32), 0, 0
            )
        return checks

    def candidate_run(case, before, expected, graph):
        load(case, before)
        for name, value in candidate.allocations.items():
            if name not in ("signal", "ffts"):
                value.fill_(37)
        old_generation = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)[0, :, 0]
        graph.replay()
        torch.npu.synchronize()
        actual = collect(case)
        comparisons = {name: compare_tensor(value, expected[name], 0, 0) for name, value in actual.items()}
        guards = checked_guards(case, before)
        geo = geometry(case)
        signal = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
        protocol = bool((signal[0, :, 0] == old_generation + 1).all() and (signal[:, :, 1:] == 0).all())
        same = addresses == {name: tensor.data_ptr() for name, tensor in candidate.buffers.items()}
        scratch = {
            name: bool((storage[:guard].cpu() == 37).all() and (storage[-guard:].cpu() == 37).all())
            for name, (storage, guard) in allocations.items()
        }
        topk = compare_topk(
            actual["topk"],
            expected["topk"],
            (case["positions"][: case["actual_tokens"]].cpu() + 1) // 4,
        )
        passed = (
            same
            and protocol
            and all(scratch.values())
            and topk["status"] != "FAIL"
            and all(value["status"] == "PASS" for value in (*comparisons.values(), *guards.values(), *geo.values()))
        )
        checked = dict(
            passed=passed,
            comparisons=comparisons,
            guards=guards,
            geometry=geo,
            same_addresses=same,
            protocol=protocol,
            scratch=scratch,
            topk=topk,
        )
        if plan_diagnostic is not None:
            checked["scale_writer_plan"] = plan_diagnostic(candidate)
        return actual, checked

    # All cases below use the same captured root addresses and signal generations.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        candidate(config=options)
    if b1_only:
        cases = [(f"b1_s{q}_tail_capacity6", [q], [history + q % 4], False) for q in range(1, 7)]
    else:
        cases = [(f"uniform_s{q}", [q] * 6, [history + r for r in range(6)], False) for q in range(1, 7)]
        cases.extend(
            [
                ("mixed_261", [2, 6, 1], [history + 3, history + 7, history + 11], True),
                ("mixed_123456", [1, 2, 3, 4, 5, 6], [history + r * 3 for r in range(6)], False),
            ]
        )
    if continuous_only:
        cases = []
        result["metadata_cases_skipped"] = "已完成的独立metadata矩阵不重跑；本次聚焦连续状态故障"
    for name, queries, histories, reverse in cases:
        case = new_case(queries, histories, reverse=reverse)
        load(case, initial)
        baseline(config=options)
        torch.npu.synchronize()
        expected = collect(case)
        baseline_guards = checked_guards(case, initial)
        _, checked = candidate_run(case, initial, expected, graph)
        checked.update(
            name=name,
            query_lengths=queries,
            history_lengths=histories,
            active_tokens=case["actual_tokens"],
            baseline_guards=baseline_guards,
        )
        checked["passed"] &= all(value["status"] == "PASS" for value in baseline_guards.values())
        result["metadata_cases"].append(checked)
        save()
        if not checked["passed"]:
            return result

    if b1_only:
        result["passed"] = all(row["passed"] for row in result["metadata_cases"])
        result["scope"] = "B1同址图S1～S6，T容量6且Q末端为实际长度；无dummy请求吸收尾部"
        save()
        return result

    # Request IDs own stable physical pages, while row order and membership change.
    # Each branch loads its last produced state, not fixture.initial.
    old_cache, new_cache, native_cache = dict(initial), dict(initial), dict(initial)
    history_by_id = {request: history + request for request in range(7)}
    steps = [
        ([0, 1, 2, 3, 4, 5], [2, 6, 1, 4, 3, 5]),
        ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]),
        ([5, 0, 4, 1], [3, 6, 1, 2]),
        ([1, 6, 5, 0, 4], [4, 2, 1, 5, 3]),
    ]
    original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
    native_topk = {}

    def capture_topk(*inputs, **kwargs):
        value = original_qli(*inputs, **kwargs)
        native_topk["value"] = value[0].detach().cpu()
        return value

    try:
        torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = capture_topk
        for index, (ids, queries) in enumerate(steps):
            histories = [history_by_id[request] for request in ids]
            case = new_case(queries, histories, owners=ids)
            # Native's SparseAttn input Q contains only the effective tokens.
            # Give the independent reference the corresponding unpadded Q/L
            # descriptors while retaining identical physical pages and state.
            # The candidate retains the original token/request capacity.
            native_case = make_fixture(
                config,
                layer.self_attn,
                len(queries),
                max(histories),
                1024,
                root["x_hc"].device,
                table_history=table_history,
                query_lengths=queries,
                history_lengths=histories,
                page_owners=ids,
                physical_request_capacity=requests,
            )
            native_case["hidden"].copy_(fixture["hidden"][: case["actual_tokens"]])
            for name, group in native_case["groups"].items():
                group["allocation"].copy_(native_cache[name])
                group["initial"] = native_cache[name]
            native_output = torch.empty_like(native_case["hidden"])
            result["current_step"] = {"step": index, "phase": "native_reference", "request_ids": ids}
            save()
            with set_ascend_forward_context(
                native_case["metadata"],
                config,
                num_tokens=case["actual_tokens"],
                num_actual_tokens=case["actual_tokens"],
            ):
                _native_attention_half(
                    layer.self_attn.dsa_attn,
                    native_case["hidden"],
                    native_case["positions"],
                    native_output,
                )
            torch.npu.synchronize()
            native_guards = guard_checks(native_case)
            native_cache = {name: group["allocation"].cpu() for name, group in native_case["groups"].items()}
            result["current_step"]["phase"] = "original_spmd"
            save()
            load(case, old_cache)
            baseline(config=options)
            torch.npu.synchronize()
            expected = collect(case)
            baseline_guards = checked_guards(case, old_cache)
            result["current_step"]["phase"] = "tnd_graph_replay"
            save()
            actual, checked = candidate_run(case, new_cache, expected, graph)
            count = case["actual_tokens"]
            checked.update(
                step=index,
                request_ids=ids,
                query_lengths=queries,
                history_lengths=histories,
                baseline_guards=baseline_guards,
                native_guards=native_guards,
                native_output_diagnostic=compare_tensor(actual["output"], native_output[:count], 0.01, 0.01),
                native_topk_diagnostic=compare_topk(
                    actual["topk"],
                    native_topk["value"].reshape(-1, 512)[:count],
                    (case["positions"][:count].cpu() + 1) // 4,
                ),
                starts_from_previous_state=index > 0,
            )
            checked["passed"] &= all(
                value["status"] == "PASS" for value in (*baseline_guards.values(), *native_guards.values())
            )
            result["continuous_steps"].append(checked)
            save()
            if not checked["passed"]:
                return result
            old_cache = {name: expected[name] for name in fixture["groups"]}
            new_cache = {name: actual[name] for name in fixture["groups"]}
            for request, query in zip(ids, queries):
                history_by_id[request] += query
    finally:
        torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
    result["passed"] = True
    result["current_step"] = {"phase": "complete"}
    result["same_graph_replays"] = len(cases) + len(steps)
    save()
    return result
