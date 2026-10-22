# SPDX-License-Identifier: Apache-2.0
"""同址TND图与连续状态功能矩阵；不采性能、不增加生产Host metadata。"""

import importlib

import torch
from scenarios import make_fixture

METADATA_NAMES = (
    "position_ids",
    "kv_seq_lens",
    "cmp_query_start_loc",
    "cmp_seq_lens",
    "idx_query_start_loc",
    "ori_block_table",
    "cmp_block_table",
    "idx_block_table",
    "state_block_table",
    "inner_state_block_table",
    "ori_slot_mapping",
    "state_slot_mapping",
    "inner_state_slot_mapping",
    "cmp_slot_mapping",
    "idx_slot_mapping",
    "freqs_cos",
    "freqs_sin",
    "cmp_freqs_cos",
    "cmp_freqs_sin",
    "inner_freqs_cos",
    "inner_freqs_sin",
)


def validate(*, config, layer, fixture, root, native_bind, entries, graphs, guards, compare, case, report, save):
    capacity, requests = fixture["tokens"], fixture["request_capacity"]
    if set(entries) != {"spmd", "mpmd"} or capacity < 36 or requests < 7:
        raise ValueError("状态验证需要容纳六个S6及dummy的SPMD/MPMD两侧同址根")
    history = case["history_label"]
    initial = {name: group["initial"] for name, group in fixture["groups"].items()}
    addresses = {side: {k: v.data_ptr() for k, v in entry.buffers.items()} for side, entry in entries.items()}
    result = {"passed": False, "performance_measured": False, "metadata_cases": [], "continuous_steps": []}
    report["state_validation"] = result

    def new_case(queries, histories, owners=None, page_pattern="original"):
        other = make_fixture(
            config,
            layer.self_attn,
            len(queries),
            max(histories),
            case["seed"],
            root["x_hc"].device,
            query_lengths=queries,
            history_lengths=histories,
            token_capacity=capacity,
            request_capacity=requests,
            table_history=case["table_history"],
            page_owners=owners,
            indexer_page_pattern=page_pattern,
            cache_template=fixture["groups"],
        )
        groups = {name: (other["metadata"][g["prefix"]], g["views"]) for name, g in other["groups"].items()}
        source = native_bind(
            root,
            fixture["hidden"],
            other["positions"],
            groups,
            layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
            compact_metadata=other["compact"],
            output=root["x_out"],
            topk=root["idx_topk"],
            topk_scores=root["idx_topk_scores"],
        )
        metadata = {}
        for name in METADATA_NAMES:
            value, destination = source[name], root[name]
            if (
                value.dtype != destination.dtype
                or value.shape[1:] != destination.shape[1:]
                or value.shape[0] > destination.shape[0]
            ):
                raise ValueError(f"本次metadata不能装入既有根容量：{name} {value.shape}/{destination.shape}")
            copied = torch.zeros(destination.shape, dtype=destination.dtype)
            if "slot_mapping" in name:
                copied.fill_(-1)
            elif "freqs_cos" in name:
                copied.fill_(1)
            copied[: value.shape[0]].copy_(value.cpu())
            metadata[name] = copied
        for entry in entries.values():
            select = importlib.import_module(type(entry.policy).__module__).select_policy
            selected = select(
                tokens=capacity,
                requests=requests,
                max_query_tokens=6,
                max_seq_len=max(h + q for h, q in zip(histories, queries)),
            )
            if selected != entry.policy:
                raise ValueError("场景跨越资源策略，必须重新捕获而不能重放旧图")
        slots = {}
        for name, group in other["groups"].items():
            value = (
                other["compact"][name][2]
                if name in other["compact"]
                else other["metadata"][group["prefix"]].decode.slot_mapping
            )
            pairs = sorted({(p, r) for p, r in value.cpu().tolist() if p >= 0 and r >= 0})
            slots[name] = ([p for p, _ in pairs], [r for _, r in pairs])
        return other, metadata, slots

    def generation(entry):
        states = {}
        for name, value in entry.allocations.items():
            if not name.endswith("signal") or value.numel() == 1:
                continue
            count = 72
            if name == "hc_signal":
                count = 3 * entry.policy.hc_groups
            elif name == "idx_front_signal":
                count = 3 * entry.policy.idx_front_groups
            elif name == "main_front_signal":
                count = 3 * entry.policy.main_front_groups
            raw = value.cpu().view(-1, 128)
            states[name] = (raw[:count, 0].clone(), bool((raw[:, 1:] == 0).all()))
        return states

    def run(side, other, metadata, slots, before):
        for name, value in metadata.items():
            root[name].copy_(value)
        for name, value in before.items():
            fixture["groups"][name]["allocation"].copy_(value)
        root["x_out"].fill_(float("nan"))
        root["idx_topk"].fill_(-777)
        root["idx_topk_scores"].fill_(float("nan"))
        prior = generation(entries[side])
        graphs[side].replay()
        torch.npu.synchronize()
        after_generation = generation(entries[side])
        protocol = {
            name: bool(after_generation[name][1] and torch.equal(after_generation[name][0], value[0] + 1))
            for name, value in prior.items()
        }
        values = {
            "output": root["x_out"].cpu(),
            "topk": root["idx_topk"].cpu(),
            "scores": root["idx_topk_scores"].cpu(),
        }
        caches, protected = {}, {}
        for name, group in fixture["groups"].items():
            raw = group["allocation"].cpu()
            caches[name] = raw
            protected[name] = bool(((raw == before[name]) | other["groups"][name]["allowed"]).all())
            for index, view in enumerate(group["views"]):
                cpu_view = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                values[f"{name}.{index}"] = cpu_view[slots[name]].clone()
        protected.update({"metadata/" + name: torch.equal(root[name].cpu(), value) for name, value in metadata.items()})
        for (owner, name), (storage, padding) in guards.items():
            if owner == side:
                protected["scratch/" + name] = bool(
                    (storage[:padding].cpu() == 37).all() and (storage[-padding:].cpu() == 37).all()
                )
        same = addresses[side] == {k: v.data_ptr() for k, v in entries[side].buffers.items()}
        geometry = {}
        prefix, offsets = 0, []
        for q, h in zip(other["query_lengths"], other["history_lengths"]):
            offsets.append(prefix - h // 4 - 1)
            prefix += (h + q) // 4 - h // 4
        offsets += [0] * (requests - len(offsets))
        for branch in ("main", "idx"):
            geometry[branch + "_offsets"] = torch.equal(
                entries[side].buffers[branch + "_offsets"].cpu(), torch.tensor(offsets, dtype=torch.int32)
            )
            geometry[branch + "_count"] = torch.equal(
                entries[side].buffers[branch + "_count"].cpu(), torch.tensor([prefix], dtype=torch.int32)
            )
        return (
            values,
            caches,
            {"guards": protected, "same_addresses": same, "generation": protocol, "geometry": geometry},
        )

    def pair(label, queries, histories, previous, owners=None, page_pattern="original"):
        other, metadata, slots = new_case(queries, histories, owners, page_pattern)
        row = {
            "name": label,
            "query_lengths": queries,
            "history_lengths": histories,
            "physical_owners": owners,
            "indexer_page_pattern": page_pattern,
            "valid_tokens": sum(queries),
            "token_capacity": capacity,
            "request_capacity": requests,
            "checks": {},
            "passed": False,
        }
        expected, next_caches = None, {}
        for side in ("spmd", "mpmd"):
            save("state_" + label + "_" + side)
            values, next_caches[side], checked = run(side, other, metadata, slots, previous[side])
            if expected is None:
                expected = values
            checked["numerical"] = compare(values, expected)
            checked["passed"] = (
                checked["same_addresses"]
                and all(checked["guards"].values())
                and all(checked["generation"].values())
                and all(checked["geometry"].values())
                and all(v["status"] == "PASS" for v in checked["numerical"].values())
            )
            row["checks"][side] = checked
        row["passed"] = all(c["passed"] for c in row["checks"].values())
        return row, next_caches

    # 同一捕获根从小S切到大S，再回到S1；包含C4/页边界及单dummy承接长padding。
    histories = [history - n for n in (1, 4, 31, 32, 127, 128)]
    schedules = [(f"uniform_s{s}", [s] * 6, histories, None) for s in (1, 6, 2, 3, 4, 5)]
    schedules += [
        ("no_compact_output", [1] * 6, [history - 2 - 4 * r for r in range(6)], None),
        ("mixed_reverse_owners", [6, 1, 5, 2, 4, 3], histories, [5, 4, 3, 2, 1, 0]),
        ("uniform_s1_again", [1] * 6, histories, None),
    ]
    schedules = [(*schedule, "original") for schedule in schedules]
    schedules.append(("fragmented_pages", [6, 1, 5, 2, 4, 3], histories, None, "fragmented_history"))
    for label, queries, history_values, owners, pattern in schedules:
        row, _ = pair(label, queries, history_values, {side: initial for side in entries}, owners, pattern)
        result["metadata_cases"].append(row)
        save()
        if not row["passed"]:
            return result

    previous = {side: initial for side in entries}
    history_by_id = {i: history - 4 + i for i in range(7)}
    steps = [
        ([0, 1, 2, 3, 4, 5], [2, 6, 1, 4, 3, 5]),
        ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]),
        ([5, 0, 4, 1], [3, 6, 1, 2]),
        ([1, 6, 5, 0, 4], [4, 2, 1, 5, 3]),
    ]
    for step, (ids, queries) in enumerate(steps):
        row, next_caches = pair(f"continuous_{step}", queries, [history_by_id[i] for i in ids], previous, ids)
        row["starts_from_previous_state"] = step > 0
        result["continuous_steps"].append(row)
        save()
        if not row["passed"]:
            return result
        previous = next_caches
        for request, q in zip(ids, queries):
            history_by_id[request] += q
    result["passed"] = True
    return result
