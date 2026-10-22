# SPDX-License-Identifier: Apache-2.0
"""同一组设备地址上修改 TND metadata 的直接调用验证，不使用 ACLGraph。

每个场景先用新建的 Native metadata 描述符运行同一 PTO 入口，再把其内容
复制进第一次调用的固定地址。两侧完整输出/cache/state 必须逐 bit 相同，
同时检查 Native 输出容差、只读输入与允许 slot 之外的完整分配。
新 metadata 由原 Native builder 生成；测试不手写生产地址映射算法。
"""

import torch

from csa_mpmd_tnd_20261021.scenarios import make_fixture
from dsv4_csa_validation import compare_tensor
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.dsv4_hca import native_attention_half

METADATA = (
    "positions", "query_bounds", "seq_lens", "state_table", "state_slots",
    "cmp_cos", "cmp_sin", "cmp_slots", "cmp_table", "ori_table", "ori_slots",
    "freqs_cos", "freqs_sin",
)


def validate(*, adapter, operators, weights, pinned, config, layer, fixture, output, history, save):
    if fixture["tokens"] < 24 or fixture["request_capacity"] < 5:
        raise ValueError("该矩阵需要至少 T24/请求容量5，容纳四个真实请求与一个 dummy")
    capacity, requests = fixture["tokens"], fixture["request_capacity"]
    base = history // 128 * 128
    histories = [base - 1, base, base + 127, base + 128]
    schedules = [
        ("A_s1", [1] * 4, histories, None),
        ("B_mixed_reverse_owners", [6, 1, 5, 2], histories, [3, 2, 1, 0]),
        ("C_s5", [5] * 4, [base, base + 126, base + 129, base - 2], None),
        ("D_no_compact", [1, 5, 1, 5], [base, base, base + 128, base + 128], None),
        ("A_s1_again", [1] * 4, histories, None),
    ]
    addresses = {name: tensor.data_ptr() for name, tensor in pinned.args.items()}
    initial = {name: group["initial"] for name, group in fixture["groups"].items()}
    retained = []  # 新描述符在矩阵结束前保留，避免地址复用掩盖缓存错误。
    result = {"passed": False, "execution_mode": "eager_direct", "aclgraph_replay": False,
              "performance_measured": False, "cases": []}

    def restore():
        for name, group in fixture["groups"].items():
            group["allocation"].copy_(initial[name])
        output.fill_(float("nan"))

    def snapshot():
        return {"output": output.cpu(), **{
            name: group["allocation"].cpu() for name, group in fixture["groups"].items()
        }}

    first_a = None
    for label, queries, history_values, owners in schedules:
        other = make_fixture(
            config, layer.self_attn, 4, max(history_values), 1024, output.device,
            query_lengths=queries, history_lengths=history_values,
            token_capacity=capacity, request_capacity=requests,
            table_history=fixture["table_history"], page_owners=owners,
            cache_template=fixture["groups"],
        )
        other["hidden"] = fixture["hidden"]
        groups = {name: (other["metadata"][group["prefix"]], tuple(group["views"]))
                  for name, group in other["groups"].items()}
        fresh = adapter.NativeHCACall(
            operators, weights, other["hidden"], other["positions"], groups,
            layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
            compact_metadata=other["compact"]["compressed"], output=output,
        )
        retained.append((other, fresh))
        for name in METADATA:
            source, destination = fresh.args[name], pinned.args[name]
            if (source.shape[1:] != destination.shape[1:] or source.shape[0] > destination.shape[0]):
                raise ValueError(f"{name} 超出固定容量：{source.shape}/{destination.shape}")

        restore()
        with set_ascend_forward_context(other["metadata"], config, num_tokens=capacity,
                                       num_actual_tokens=sum(queries)):
            native_attention_half(layer.self_attn.dsa_attn, other["hidden"], other["positions"], output)
        torch.npu.synchronize()
        native_output = output[:sum(queries)].cpu()
        native_guards = {
            name: bool(((group["allocation"].cpu() == initial[name]) | other["groups"][name]["allowed"]).all())
            for name, group in fixture["groups"].items()
        }
        restore()
        fresh()
        torch.npu.synchronize()
        reference = snapshot()
        metadata = {}
        for name in METADATA:
            source, destination = fresh.args[name], pinned.args[name]
            # 只在测试中填入已有容量。compact producer 输出数随边界变化，尾行
            # 的 slots=-1 让同址调用忽略它们；正式绑定不做这个复制或填充。
            value = torch.zeros(destination.shape, dtype=destination.dtype)
            if name.endswith("slots"):
                value.fill_(-1)
            elif "cos" in name:
                value.fill_(1)
            value[:source.shape[0]].copy_(source.cpu())
            metadata[name] = value
        for name, value in metadata.items():
            pinned.args[name].copy_(value)
        restore()
        pinned()
        torch.npu.synchronize()
        actual = snapshot()
        exact = {name: compare_tensor(value, reference[name], 0, 0) for name, value in actual.items()}
        native_check = compare_tensor(actual["output"][:sum(queries)], native_output, 0.01, 0.01)
        guards = {name: bool(((actual[name] == value) | other["groups"][name]["allowed"]).all())
                  for name, value in initial.items()}
        readonly = {name: torch.equal(pinned.args[name].cpu(), value) for name, value in metadata.items()}
        stable = addresses == {name: tensor.data_ptr() for name, tensor in pinned.args.items()}
        row = {"name": label, "queries": queries, "histories": history_values, "owners": owners,
               "actual_tokens": sum(queries), "token_capacity": capacity,
               "same_addresses": stable, "fresh_vs_pinned": exact, "native_output": native_check,
               "native_guards": native_guards, "pto_guards": guards, "metadata_unchanged": readonly}
        if first_a is None:
            first_a = actual
        if label == "A_s1_again":
            row["aba_exact"] = {name: torch.equal(value, first_a[name]) for name, value in actual.items()}
        row["passed"] = (stable and all(check["status"] == "PASS" for check in exact.values())
                         and native_check["status"] == "PASS" and all(guards.values())
                         and all(native_guards.values()) and all(readonly.values())
                         and all(row.get("aba_exact", {}).values()))
        result["cases"].append(row)
        save(label, result)
        if not row["passed"]:
            return result
    result["passed"] = True
    return result
