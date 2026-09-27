# SPDX-License-Identifier: Apache-2.0
"""Queue a complete offline comparison batch before starting its generation."""


def generate_aligned_batch(llm, prompts, params):
    # vLLM's public level-0 pause only stops scheduling; it neither offloads
    # weights nor frees KV memory. All operations here precede measured steps.
    llm.sleep(level=0, mode="keep")
    try:
        if prompts:
            llm.enqueue(prompts, params, use_tqdm=False)
        llm.collective_rpc("offline_batch_barrier")
    finally:
        llm.wake_up(tags=["scheduling"])
    return llm.wait_for_completion(use_tqdm=False)
