# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#


import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm, RMSNormGated

from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu
from vllm_ascend.utils import enable_custom_op, get_weight_prefetch_method

_QWEN3_14B_HIDDEN_SIZE = 5120
_QWEN3_14B_NUM_LAYERS = 40
_QWEN3_14B_NUM_HEADS = 40
_QWEN3_14B_NUM_KV_HEADS = 8
_QWEN3_HEAD_DIM = 128
_QWEN3_MAX_PADDED_TOKENS = 1024
_PYPTO_QWEN3_MODES = {"off", "partial", "attention_block", "full"}


def _pypto_qwen3_mode() -> str:
    from vllm_ascend import envs

    mode = envs.VLLM_ASCEND_PYPTO_QWEN3_MODE
    if mode not in _PYPTO_QWEN3_MODES:
        raise ValueError(f"VLLM_ASCEND_PYPTO_QWEN3_MODE must be one of {sorted(_PYPTO_QWEN3_MODES)}, got {mode!r}")
    return mode


def _validate_pypto_qwen3_config() -> None:
    vllm_config = get_current_vllm_config()
    config = vllm_config.model_config.hf_config
    supported = (
        config.model_type == "qwen3"
        and config.hidden_size == _QWEN3_14B_HIDDEN_SIZE
        and config.num_hidden_layers == _QWEN3_14B_NUM_LAYERS
        and config.num_attention_heads == _QWEN3_14B_NUM_HEADS
        and config.num_key_value_heads == _QWEN3_14B_NUM_KV_HEADS
        and getattr(config, "head_dim", None) == _QWEN3_HEAD_DIM
        and config.rms_norm_eps == 1e-6
        and vllm_config.model_config.dtype == torch.bfloat16
        and vllm_config.parallel_config.tensor_parallel_size == 1
        and vllm_config.quant_config is None
        and vllm_config.speculative_config is None
        and not vllm_config.scheduler_config.enable_chunked_prefill
    )
    if not supported:
        raise ValueError(
            "PyPTO Qwen3 mode currently requires Qwen3-14B BF16, TP1, "
            "head_dim=128, no quantization, no speculative decoding, and chunked prefill disabled"
        )


class AscendRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        self.bias_loaded = False

        self._pypto_qwen3_mode = _pypto_qwen3_mode()
        self._pypto_qk_op = None
        self._pypto_full_ops = None
        if self._pypto_qwen3_mode in ("partial", "full"):
            _validate_pypto_qwen3_config()
            if hidden_size == _QWEN3_HEAD_DIM:
                from vllm_ascend.ops import pypto_qwen3_rms

                pypto_qwen3_rms.warmup(torch.device("npu", torch.npu.current_device()))
                self._pypto_qk_op = pypto_qwen3_rms.registered_op()
            elif self._pypto_qwen3_mode == "full":
                from vllm_ascend.ops import pypto_qwen3_full

                pypto_qwen3_full.init()
                self._pypto_full_ops = pypto_qwen3_full.registered_ops()

        # quantization with anti_method m4 will generate none-zero norm bias
        if vllm_config.quant_config is not None and any(
            "norm.bias" in name for name in vllm_config.quant_config.quant_description
        ):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
            self.bias.weight_loader = self._bias_weight_loader

    def _bias_weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
        self.bias_loaded = True

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if self._pypto_qwen3_mode in ("partial", "full") and self.hidden_size == _QWEN3_HEAD_DIM:
            if residual is not None:
                raise ValueError("Qwen3 q/k RMSNorm does not accept a residual tensor")
            if x.ndim != 3 or x.shape[1] not in (_QWEN3_14B_NUM_HEADS, _QWEN3_14B_NUM_KV_HEADS):
                raise ValueError(f"PyPTO Qwen3 q/k RMSNorm requires [tokens, 40|8, 128] input, got {tuple(x.shape)}")
            if not 1 <= x.shape[0] <= _QWEN3_MAX_PADDED_TOKENS:
                raise ValueError(
                    f"PyPTO Qwen3 q/k RMSNorm supports 1..{_QWEN3_MAX_PADDED_TOKENS} padded tokens, got {x.shape[0]}"
                )
            contiguous_x = x.contiguous()
            output = torch.empty_like(contiguous_x)
            assert self._pypto_qk_op is not None
            return self._pypto_qk_op(
                contiguous_x.view(-1, _QWEN3_HEAD_DIM),
                self.weight.contiguous().view(1, _QWEN3_HEAD_DIM),
                output.view(-1, _QWEN3_HEAD_DIM),
            ).view_as(x)

        if self._pypto_qwen3_mode == "full":
            if x.dtype != torch.bfloat16 or x.shape[-1] != _QWEN3_14B_HIDDEN_SIZE:
                raise ValueError("PyPTO Qwen3 full RMSNorm requires BF16 [..., 5120] input")
            if not x.is_contiguous() or not self.weight.is_contiguous():
                raise ValueError("PyPTO Qwen3 full RMSNorm requires contiguous input and weight")
            rows = x.numel() // x.shape[-1]
            x_2d = x.view(rows, x.shape[-1])
            assert self._pypto_full_ops is not None
            if residual is None:
                output = torch.empty_like(x_2d)
                result = self._pypto_full_ops["rms_norm"](
                    x_2d,
                    self.weight.view(1, -1),
                    output,
                )
                return result.view_as(x)
            if residual.shape != x.shape or residual.dtype != torch.bfloat16 or not residual.is_contiguous():
                raise ValueError("PyPTO Qwen3 full add-RMSNorm requires matching contiguous BF16 residual")
            norm_output = torch.empty_like(x_2d)
            residual_output = torch.empty_like(x_2d)
            normalized, added = self._pypto_full_ops["add_rms_norm"](
                x_2d,
                residual.view_as(x_2d),
                self.weight.view(1, -1),
                norm_output,
                residual_output,
            )
            return normalized.view_as(x), added.view_as(residual)

        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            if enable_custom_op() and self._pypto_qwen3_mode == "off":
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, self.weight, self.bias, self.variance_epsilon
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
                if self.bias is not None:
                    x.add_(self.bias)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias_loaded:
            x.add_(self.bias)

        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(x)
        return x


class AscendGemmaRMSNorm(GemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            if enable_custom_op():
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, 1.0 + self.weight, None, self.variance_epsilon
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
        activation: str = "swish",
    ):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)


class AscendRMSNormGated(RMSNormGated):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        # `activation` was added in vLLM #40245 (Qwen3-Next/GDN). Accept and
        # forward it; older vllm versions did not pass this kwarg so the
        # default keeps existing behavior.
        activation: str = "swish",
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            hidden_size,
            eps,
            group_size,
            norm_before_gate,
            device,
            dtype,
            activation=activation,
        )
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_oot(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)
