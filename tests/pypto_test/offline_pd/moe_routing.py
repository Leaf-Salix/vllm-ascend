# SPDX-License-Identifier: Apache-2.0
"""小规模路由诊断：图内保存整数，图外读取；不作为性能测量。"""

import re


class RoutingCapture:
    def __init__(self, tokens, layers):
        self.tokens = tokens
        self.layers = layers
        self.buffers = {}
        self.active_layer = None

    def save(self, layer, topk_ids, group_list, group_list_type, active_mask=None):
        # clone 被捕获进 ACL Graph，后续 replay 会更新；保留 Python hook 本身不能观测 replay。
        self.buffers[layer] = {
            "topk_ids": topk_ids.detach().clone(),
            "group_list": group_list.detach().clone(),
            "group_list_type": int(group_list_type),
            "active_mask": None if active_mask is None else active_mask.detach().clone(),
        }

    def install(self):
        import torch

        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
        from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2

        original_forward = AscendMoERunner._forward_impl
        original_dispatch = TokenDispatcherWithMC2.token_dispatch

        def forward(module, *args, **kwargs):
            previous = self.active_layer
            match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$", module.layer_name)
            self.active_layer = int(match[1]) if match else None
            try:
                return original_forward(module, *args, **kwargs)
            finally:
                self.active_layer = previous

        def dispatch(dispatcher, token_dispatch_input):
            result = original_dispatch(dispatcher, token_dispatch_input)
            layer = self.active_layer
            if (layer is not None and 0 <= layer < self.layers
                    and token_dispatch_input.topk_ids.shape[0] == self.tokens
                    and torch.npu.is_current_stream_capturing()):
                mask = token_dispatch_input.routing.mc2_mask if dispatcher.global_bs == 0 else None
                self.save(layer, token_dispatch_input.topk_ids, result.group_list, result.group_list_type, mask)
            return result

        AscendMoERunner._forward_impl = forward
        TokenDispatcherWithMC2.token_dispatch = dispatch

    def arm(self):
        import torch

        if set(self.buffers) != set(range(self.layers)):
            raise RuntimeError(f"路由图内采集层不完整：{sorted(self.buffers)}，预期 {self.layers} 层")
        # 排除读取捕获期 dummy 数据：运行前毒化，真实 replay 必须重写这些独立缓冲。
        # Worker graph capture creates inference tensors. The RPC that arms
        # observation runs outside model inference, so re-enter that context
        # before poisoning these fixed-address capture buffers.
        with torch.inference_mode():
            for values in self.buffers.values():
                values["topk_ids"].fill_(-1)
                values["group_list"].fill_(-1)
        torch.npu.synchronize()

    def snapshot(self):
        result = {}
        for layer, values in sorted(self.buffers.items()):
            record = {name: value.cpu().tolist() if hasattr(value, "cpu") else value
                      for name, value in values.items()}
            if any(x < 0 for row in record["topk_ids"] for x in row) or any(x < 0 for x in record["group_list"]):
                raise RuntimeError(f"layer {layer} 路由缓冲没有被真实图回放更新")
            result[str(layer)] = record
        return result


def begin(worker, warmup_steps, tokens, requests, samples):
    import torch

    collector = worker._offline_moe_routing
    collector.arm()
    runner = worker.model_runner
    original_execute, original_forward = runner.execute_model, runner._model_forward
    state = {"dp_rank": worker.vllm_config.parallel_config.data_parallel_rank,
             "expected_tokens": tokens, "expected_requests": requests, "warmup_steps": warmup_steps,
             "requested_samples": samples, "seen_steady_steps": 0, "samples": [], "observed": {},
             "scope": "图内额外复制专家索引/分组计数，图外同步采样；不能用本轮耗时评价性能"}
    active = [None]

    def forward(num_tokens_padded, input_ids=None, positions=None, *args, **kwargs):
        entry = active[0]
        if entry is None:
            return original_forward(num_tokens_padded, input_ids, positions, *args, **kwargs)
        entry["forward_calls"] += 1
        if num_tokens_padded != tokens or input_ids is None or positions is None:
            raise RuntimeError("路由采样未命中约定的满档模型输入")
        entry.update(input_ids=input_ids.cpu().tolist(), positions=positions.cpu().tolist())
        result = original_forward(num_tokens_padded, input_ids, positions, *args, **kwargs)
        torch.npu.synchronize()
        entry["layers"] = collector.snapshot()
        return result

    def execute(scheduler_output, *args, **kwargs):
        actual_tokens = scheduler_output.total_num_scheduled_tokens
        actual_requests = len(scheduler_output.num_scheduled_tokens)
        key = f"{actual_tokens}/{actual_requests}"
        state["observed"][key] = state["observed"].get(key, 0) + 1
        steady = (actual_tokens, actual_requests) == (tokens, requests)
        index = state["seen_steady_steps"]
        if steady:
            state["seen_steady_steps"] += 1
        if not steady or index < warmup_steps or len(state["samples"]) >= samples:
            return original_execute(scheduler_output, *args, **kwargs)
        entry = {"steady_step_index": index, "forward_calls": 0}
        active[0] = entry
        try:
            return original_execute(scheduler_output, *args, **kwargs)
        finally:
            active[0] = None
            state["samples"].append(entry)

    runner.execute_model, runner._model_forward = execute, forward
    worker._offline_moe_routing_window = (state, original_execute, original_forward)
    return {"captured_layers": sorted(collector.buffers), "tokens": collector.tokens}


def end(worker):
    state, execute, forward = worker._offline_moe_routing_window
    worker.model_runner.execute_model, worker.model_runner._model_forward = execute, forward
    worker._offline_moe_routing_window = None
    samples = state["samples"]
    state["sufficient"] = (len(samples) == state["requested_samples"] and all(
        x["forward_calls"] == 1 and len(x.get("layers", {})) == worker._offline_moe_routing.layers
        for x in samples))
    return state


def self_check(device):
    """单卡先验证采集缓冲在 replay 更新，且保持先前 CPU 快照独立。"""
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    topk = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32, device="npu")
    counts = torch.tensor([0, 1, 2, 1], dtype=torch.int64, device="npu")
    collector = RoutingCapture(2, 1)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        collector.save(0, topk, counts, 1)
    collector.arm()
    topk.fill_(3)
    counts.copy_(torch.tensor([0, 0, 0, 4], dtype=torch.int64, device="npu"))
    graph.replay()
    first = collector.snapshot()
    assert first["0"]["topk_ids"] == [[3, 3], [3, 3]]
    assert first["0"]["group_list"] == [0, 0, 0, 4]
    topk.fill_(1)
    graph.replay()
    assert collector.snapshot()["0"]["topk_ids"] == [[1, 1], [1, 1]]
    assert first["0"]["topk_ids"] == [[3, 3], [3, 3]]
    print("ROUTING_GRAPH_REPLAY_PASS", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    self_check(parser.parse_args().device)
