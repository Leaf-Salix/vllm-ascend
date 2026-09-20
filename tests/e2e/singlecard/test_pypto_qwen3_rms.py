"""Qwen3 q/k RMSNorm through PyPTO kernel-mode eager and ACLGraph replay."""

import os

import pytest
import torch


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    values = x.float().cpu()
    gamma = weight.float().cpu()
    return (values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6) * gamma).to(torch.bfloat16)


@pytest.mark.skipif("PYPTO_DEVICE" not in os.environ, reason="requires a scheduled A2/A3 NPU")
def test_qwen3_qk_rms_eager_and_replay():
    import torch_npu  # noqa: F401
    from pypto.runtime.kernel.context import get_process_kernel_state

    from vllm_ascend.ops import pypto_qwen3_rms

    device_id = int(os.environ["PYPTO_DEVICE"])
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    pypto_qwen3_rms.warmup(device)

    q = torch.randn((1, 40, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 8, 128), device=device, dtype=torch.bfloat16)
    q_weight = torch.rand((128,), device=device, dtype=torch.bfloat16)
    k_weight = torch.rand((128,), device=device, dtype=torch.bfloat16)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    pypto_qwen3_rms.run(q, q_weight, q_out)
    pypto_qwen3_rms.run(k, k_weight, k_out)
    torch.npu.synchronize()
    torch.testing.assert_close(q_out.cpu(), _reference(q, q_weight), atol=0.02, rtol=0.02)
    torch.testing.assert_close(k_out.cpu(), _reference(k, k_weight), atol=0.02, rtol=0.02)

    op = pypto_qwen3_rms.registered_op()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op(q.view(40, 128), q_weight.view(1, 128), q_out.view(40, 128))
        op(k.view(8, 128), k_weight.view(1, 128), k_out.view(8, 128))
    for _ in range(2):
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn_like(k))
        q_weight.copy_(torch.rand_like(q_weight))
        k_weight.copy_(torch.rand_like(k_weight))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(q_out.cpu(), _reference(q, q_weight), atol=0.02, rtol=0.02)
        torch.testing.assert_close(k_out.cpu(), _reference(k, k_weight), atol=0.02, rtol=0.02)

    graph.reset()
    get_process_kernel_state().close()
