"""Run the Qwen3-14B attention-only PyPTO integration end to end."""

from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    prompts = ("Hello", "The capital of France is")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.75,
        enforce_eager=args.enforce_eager,
        # The fixed CANN 9.0 test host does not provide
        # aclnnAddRmsNormBias, which is used only by optional graph fusion
        # passes around the native MLP. Keep the native MLP itself and
        # ACLGraph enabled while disabling those unavailable rewrites.
        additional_config={
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
                "fuse_qknorm_rope": False,
                "fuse_allreduce_rms": False,
                "fuse_muls_add": False,
            }
        },
    )
    results = llm.generate(
        list(prompts),
        SamplingParams(temperature=0, max_tokens=4),
    )
    payload = [
        {
            "prompt": result.prompt,
            "token_ids": list(result.outputs[0].token_ids),
            "text": result.outputs[0].text,
        }
        for result in results
    ]
    print("QWEN3_ATTENTION_ONLY_E2E", json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
