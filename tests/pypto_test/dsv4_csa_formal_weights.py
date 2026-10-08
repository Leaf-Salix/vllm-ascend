"""Load the selected CSA layer through Native Ascend ModelSlim parameters.

This path consumes the formal quant_model_weights index and stored scales and
offsets directly. It does not apply the reference checkpoint's scale aliases or
derive missing quantization tensors.
"""

import json
from pathlib import Path


def load_formal_layer_weights(attention, checkpoint: Path, layer_index=2):
    import torch
    from safetensors import safe_open
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

    config = get_current_vllm_config()
    if config.model_config.quantization != "ascend" or not isinstance(config.quant_config, AscendModelSlimConfig):
        raise ValueError("Formal W8A8 weights require the Native Ascend ModelSlim configuration")
    description = json.loads((checkpoint / "quant_model_description.json").read_text())
    if description["model_quant_type"] != "W8A8_DYNAMIC":
        raise ValueError("This CSA validation expects a W8A8_DYNAMIC ModelSlim checkpoint")
    quantized = ("wq_b", "wo_b", "indexer.wq_b") if attention.compress_ratio == 4 else ("wq_b", "wo_b")
    for suffix in quantized:
        if description[f"layers.{layer_index}.attn.{suffix}.weight"] != "W8A8_DYNAMIC":
            raise ValueError(f"Unexpected formal CSA quantization for {suffix}")
    prefix = f"layers.{layer_index}.attn."
    index = checkpoint / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    parameters = dict(attention.named_parameters())
    expected = {prefix + name for name in parameters}
    if set(selected) != expected:
        raise ValueError(
            f"Formal CSA tensors do not match Native parameters: missing={expected - set(selected)}, "
            f"unexpected={set(selected) - expected}"
        )
    records = []
    with torch.no_grad():
        for shard in sorted(set(selected.values())):
            with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
                for name, mapped_shard in selected.items():
                    if mapped_shard != shard:
                        continue
                    suffix = name.removeprefix(prefix)
                    value = reader.get_tensor(name)
                    parameter = parameters[suffix]
                    if value.shape != parameter.shape:
                        raise ValueError(
                            f"Unexpected formal parameter shape: {name}: {value.shape} vs {parameter.shape}"
                        )
                    # The TP1 attention branches of Native model.load_weights
                    # use these same parameter loaders (sink uses a copy).
                    getattr(parameter, "weight_loader", default_weight_loader)(parameter, value)
                    records.append(
                        {
                            "name": name,
                            "parameter_name": suffix,
                            "shape": list(value.shape),
                            "checkpoint_dtype": str(value.dtype),
                            "loaded_dtype": str(parameter.dtype),
                            "shard": shard,
                            "source": "formal_modelslim_stored_tensor",
                        }
                    )
        methods = {
            name: type(module.quant_method).__module__ + "." + type(module.quant_method).__name__
            for name, module in attention.named_modules()
            if getattr(module, "quant_method", None) is not None
        }
        process_weights_after_loading(attention, config.model_config, next(attention.parameters()).device)
    return records, methods
