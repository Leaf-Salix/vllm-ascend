# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable HC weight bytes once, retaining the Native-order service.

Weight reload requires a new runtime and new graphs. The metadata guard catches
object/storage/layout changes and ordinary versioned mutations at Python entry;
inference tensors, .data writes and raw graph replay cannot provide a complete
mutation check. Updating weights during a live runtime is unsupported.
"""

import weakref

import torch

from ..deepseek_v4_flash_csa_tnd_precision_spmd.service import CSAServiceRuntime as PrecisionSPMDRuntime

PACKED_HC_SHAPE = (32768, 32)


def tensor_identity(value):
    """Host metadata only; inference tensors have no readable version counter."""
    version = None if torch.is_inference(value) else value._version
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device, version)


class CSAServiceRuntime(PrecisionSPMDRuntime):
    """One full-CSA root per forward; original DSL pack runs only at loading."""

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        super().__init__(attention, operators, max_num_seqs, layer)
        self._hc_layer_ref = weakref.ref(layer)
        self._hc_source = layer.hc_attn_fn
        self._hc_source_identity = tensor_identity(self._hc_source)
        # Make our own constant versioned even when the model loads under
        # inference_mode. This does not add tracking to the Native source.
        with torch.inference_mode(False):
            packed = torch.empty(PACKED_HC_SHAPE, dtype=torch.bfloat16, device=self._hc_source.device)
        # Model loading can have grad enabled and Parameter.requires_grad=True;
        # the preparer is inference-only and never constructs an autograd graph.
        with torch.no_grad():
            operators.workspace.prepare_hc_weight(self._hc_source, packed)
        torch.npu.current_stream(self._hc_source.device).synchronize()
        self.weights["hc_packed_weight"] = packed
        self._hc_packed_weight = packed
        self._hc_packed_identity = tensor_identity(packed)

    def __call__(self, context, hidden, positions, output, kv_cache):
        layer = self._hc_layer_ref()
        if (
            layer is None
            or layer.hc_attn_fn is not self._hc_source
            or tensor_identity(self._hc_source) != self._hc_source_identity
            or self.weights["hc_packed_weight"] is not self._hc_packed_weight
            or tensor_identity(self._hc_packed_weight) != self._hc_packed_identity
        ):
            raise RuntimeError("HC weights changed: rebuild the precision runtime and captured graphs")
        return super().__call__(context, hidden, positions, output, kv_cache)
