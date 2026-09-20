#
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
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul, SwigluOAIAndMul

from vllm_ascend.utils import get_weight_prefetch_method


class AscendQuickGELU(QuickGELU):
    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        import torch_npu

        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pypto_full_op = None
        from vllm_ascend import envs

        if envs.VLLM_ASCEND_PYPTO_QWEN3_MODE == "full":
            from vllm_ascend.ops import pypto_qwen3_full

            pypto_qwen3_full.init()
            self._pypto_full_op = pypto_qwen3_full.registered_ops()["silu_and_mul"]

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if self._pypto_full_op is not None:
            if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 2 or not x.is_contiguous():
                raise ValueError(
                    "PyPTO Qwen3 full SwiGLU requires contiguous 2D BF16 input with an even last dimension"
                )
            output = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
            return self._pypto_full_op(x, output)

        import torch_npu

        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(weight_prefetch_method.MLP_DOWN, x)
        out = torch_npu.npu_swiglu(x)
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(out)
        return out


class AscendSwigluOAIAndMul:
    def swiglu_oai_forward(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
        class MinimalSwigluOAIAndMul:
            def __init__(self):
                self.alpha = alpha
                self.limit = limit

        layer = MinimalSwigluOAIAndMul()
        return SwigluOAIAndMul.forward_native(layer, x)
