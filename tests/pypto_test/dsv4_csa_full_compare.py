"""Compare the complete reference-derived CSA chain with Native attention."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json
from dsv4_csa_validation import compare_tensor


def check_untouched(fixture, snapshots):
    import torch

    result = {}
    for name, group in fixture["groups"].items():
        req = fixture["metadata"][group["prefix"]].req_metadata
        compact = name in ("compressed", "indexer")
        slots = req.compressor_metadata[2] if compact else req.slot_mapping
        all_slots = slots.cpu().tolist()
        if compact:
            count = int(((fixture["positions"].cpu() + 1) % 4 == 0).sum())
            slots = slots[:count]
        writable = torch.zeros_like(snapshots[name], dtype=torch.bool)
        for page, offset in slots.cpu().tolist():
            if page < 0 or offset < 0:
                continue
            for view in group["views"]:
                start = (view.storage_offset() + page * view.stride(0) + offset * view.stride(1)) * view.element_size()
                width = view.shape[-1] * view.element_size()
                writable[start : start + width] = True
        changed = group["allocation"].cpu() != snapshots[name]
        wrong = changed & ~writable
        result[name] = {
            "status": "PASS" if not bool(wrong.any()) else "FAIL",
            "changed_outside_slots": int(wrong.sum()),
            "changed_bytes": int(changed.sum()),
            "first_changed_outside_slots": wrong.nonzero()[:20].flatten().tolist(),
            "native_slots": slots.cpu().tolist(),
            "all_native_slots": all_slots,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    repo = activate()
    report = {
        "status": "FAIL",
        "batch": args.batch,
        "history": args.history,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "scope": "reference_single_layer_not_target_checkpoint_acceptance",
        "thresholds_frozen_before_execution": {
            "output_bf16": [1e-2, 1e-2],
            "cache_bf16": [1e-2, 1e-2],
            "state_fp32": [1e-3, 1e-3],
            "scale_fp16": [1e-5, 1e-3],
            "indexer_int8": [0, 0],
            "topk_int32": [0, 0],
            "untouched_bytes": [0, 0],
        },
    }
    write_json(args.output_dir / "full_compare.json", report)
    try:
        import pypto.torch
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_native_fixture import make_attention, make_config, native_session
        from dsv4_csa_native_forward import execute_native, make_numerical_fixture
        from dsv4_csa_native_layout import load_layer_weights

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import (
            CSAOperators,
            NativeCSACall,
            prepare_weights,
        )

        # load_layer_weights dispatches ModelSlim to its separate Native
        # loader; the reference scale/offset adaptations never run there.
        if (args.checkpoint / "quant_model_description.json").exists():
            report["scope"] = "formal_ModelSlim_CSA_single_layer_not_full_model_acceptance"
        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device), torch.inference_mode():
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
            attention = make_attention(config, device)
            records, methods = load_layer_weights(attention, args.checkpoint)
            report.update(weights=records, quant_methods=methods)
            native = make_numerical_fixture(config, device, attention, args.batch, args.history, args.seed)
            pto = make_numerical_fixture(config, device, attention, args.batch, args.history, args.seed)
            torch.testing.assert_close(native["hidden"], pto["hidden"], atol=0, rtol=0)
            snapshots = {}
            for name in native["groups"]:
                left, right = native["groups"][name], pto["groups"][name]
                torch.testing.assert_close(left["allocation"], right["allocation"], atol=0, rtol=0)
                snapshots[name] = left["allocation"].cpu()
                left["owner"].kv_cache = left["views"]
            captured = {}
            diagnostic_hooks = []
            select_topk = attention.indexer.ops.select_topk
            from vllm_ascend.models.deepseek_v4 import indexer as native_indexer

            hadamard_linear = native_indexer.hadamard_linear
            native_impl = attention.dsa_attn.dsa_attn.impl
            qa_matmul = native_impl.cv_wq_a.matmul
            if args.diagnostics:

                def capture_hadamard(x, matrix):
                    if x.ndim == 3 and x.shape[1] == 64:
                        captured["before_hadamard"] = x.detach().clone()
                    return hadamard_linear(x, matrix)

                def capture_select(query, weights, query_scale, key_cache, scale_cache, metadata):
                    captured.update(
                        query=query.detach().clone(),
                        weights=weights.detach().clone(),
                        query_scale=query_scale.detach().clone(),
                    )
                    return select_topk(query, weights, query_scale, key_cache, scale_cache, metadata)

                def capture_indexer_inputs(module, inputs, kwargs):
                    captured["qr"] = kwargs["qr"].detach().clone()
                    captured["qr_scale"] = kwargs["qr_pertoken_scale"].detach().clone()

                def capture_inner(module, inputs, output):
                    captured["inner_compressed"] = output[0].detach().clone()
                    captured["inner_slots"] = output[1].detach().clone()

                def capture_qa(module, inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    captured["qa_bf16"] = value.detach().clone()

                def capture_qa_matmul(*inputs, **kwargs):
                    value = qa_matmul(*inputs, **kwargs)
                    captured["qa_bf16"] = value.detach().clone()
                    return value

                attention.indexer.ops.select_topk = capture_select
                native_indexer.hadamard_linear = capture_hadamard
                native_impl.cv_wq_a.matmul = capture_qa_matmul
                diagnostic_hooks.extend(
                    [
                        attention.indexer.register_forward_pre_hook(capture_indexer_inputs, with_kwargs=True),
                        attention.indexer.compressor.register_forward_hook(capture_inner),
                        attention.wq_a.register_forward_hook(capture_qa),
                    ]
                )

            def capture_topk(module, inputs, output):
                captured["topk"] = output.detach().clone()

            hook = attention.indexer.register_forward_hook(capture_topk)
            print("Running Native full attention", flush=True)
            try:
                expected = execute_native(config, attention, native)
            finally:
                hook.remove()
                attention.indexer.ops.select_topk = select_topk
                native_indexer.hadamard_linear = hadamard_linear
                native_impl.cv_wq_a.matmul = qa_matmul
                for diagnostic_hook in diagnostic_hooks:
                    diagnostic_hook.remove()
            assert torch.isfinite(expected).all(), "Native output is nonfinite"
            report["native_untouched"] = check_untouched(native, snapshots)
            if args.diagnostics:
                captured["hidden"] = native["hidden"].detach().clone()
                torch.save({name: value.cpu() for name, value in captured.items()}, args.output_dir / "native_debug.pt")
            if args.native_only:
                report["scope"] = "native_only_diagnostic_not_full_comparison"
                report["status"] = (
                    "PASS" if all(item["status"] == "PASS" for item in report["native_untouched"].values()) else "FAIL"
                )
                if report["status"] != "PASS":
                    raise AssertionError("Native untouched-region diagnostic failed")
                return
            bindings = {
                name: (pto["metadata"][group["prefix"]], group["views"]) for name, group in pto["groups"].items()
            }
            hadamard = bindings["indexer"][0].hadamard
            weights = prepare_weights(attention, hadamard)
            operators = CSAOperators.register()
            call = NativeCSACall(
                operators,
                weights,
                pto["hidden"],
                pto["positions"],
                bindings,
                layer_name=pto["groups"]["compressed"]["prefix"],
            )
            assert call.args["freqs_cos"].data_ptr() == call.native_cos.data_ptr()
            assert call.args["freqs_sin"].data_ptr() == call.native_sin.data_ptr()
            report["native_rope_zero_copy"] = True
            direct_indices = {
                "position_ids": pto["positions"],
                "ori_slot_mapping": call.req["swa"].slot_mapping,
                "state_slot_mapping": call.req["state"].slot_mapping,
                "inner_state_slot_mapping": call.req["indexer_state"].slot_mapping,
                "ori_block_table": call.req["swa"].block_table,
            }
            report["native_indices_zero_copy"] = {
                name: call.args[name].data_ptr() == value.data_ptr() for name, value in direct_indices.items()
            }
            assert all(report["native_indices_zero_copy"].values())
            compact_inputs = {}
            for group, slot_name, rope_name in (
                ("compressed", "cmp_slot_mapping", "cmp_freqs"),
                ("indexer", "idx_slot_mapping", "inner_freqs"),
            ):
                cos, sin, slots = call.req[group].compressor_metadata
                compact_inputs.update({slot_name: slots, f"{rope_name}_cos": cos, f"{rope_name}_sin": sin})
            compact_inputs.update(
                cmp_query_start_loc=call.req["compressed"].query_start_loc,
                cmp_seq_lens=call.req["compressed"].seq_lens,
                idx_query_start_loc=call.req["indexer"].query_start_loc,
            )
            report["native_compact_zero_copy"] = {
                name: call.args[name].data_ptr() == value.data_ptr() for name, value in compact_inputs.items()
            }
            assert all(report["native_compact_zero_copy"].values())
            report["native_state_zero_copy"] = {}
            for group, state_name, table_name in (
                ("state", "compress_state", "state_block_table"),
                ("indexer_state", "inner_compress_state", "inner_state_block_table"),
            ):
                view = pto["groups"][group]["views"][0]
                state = call.args[state_name]
                assert state.shape == (view.shape[0], view.stride(0))
                report["native_state_zero_copy"][state_name] = state.data_ptr() == view.data_ptr()
                report["native_state_zero_copy"][table_name] = (
                    call.args[table_name].data_ptr() == call.req[group].block_table.data_ptr()
                )
            assert all(report["native_state_zero_copy"].values())
            executor = pto["executor"]
            executor.submit(pto["tasks"])
            for task in pto["tasks"]:
                executor.wait(task.stage, task.group_id)
            print("Running complete PTO attention with Native cache storage", flush=True)
            actual = call()
            executor.release()
            torch.npu.synchronize()
            if args.diagnostics:
                from dsv4_csa_precision_kernels import (
                    QPROJ_T_PAD,
                    T_PAD,
                    diagnose_indexer_query,
                    diagnose_qa,
                    diagnose_qr,
                    diagnose_qr_normalize,
                    diagnose_qr_rms,
                )

                qa = torch.empty((QPROJ_T_PAD, 1024), dtype=torch.float32, device=device)
                diagnose_qa(pto["hidden"], weights["wq_a"], qa)
                qr = torch.empty_like(captured["qr"])
                qr_scale = torch.empty_like(captured["qr_scale"]).reshape(-1, 1)
                diagnose_qr(pto["hidden"], weights["wq_a"], weights["gamma_cq"], qr, qr_scale)
                normalized_qr = torch.empty_like(qr)
                normalized_scale = torch.empty_like(qr_scale)
                diagnose_qr_normalize(captured["qa_bf16"].float(), weights["gamma_cq"], normalized_qr, normalized_scale)
                qr_square_sum = torch.empty_like(qr_scale)
                qr_inverse_rms = torch.empty_like(qr_scale)
                qr_rms = torch.empty_like(qr_scale)
                diagnose_qr_rms(captured["qa_bf16"].float(), qr_square_sum, qr_inverse_rms, qr_rms)
                rows = args.batch * 6 * 64
                before_hadamard = torch.empty((T_PAD * 64, 128), dtype=torch.bfloat16, device=device)
                query = torch.empty((T_PAD * 64, 128), dtype=torch.int8, device=device)
                query_scale = torch.empty((T_PAD * 64, 1), dtype=torch.float32, device=device)
                sign = (torch.arange(64, device=device) % 2 * 2 - 1).float()
                diagnose_indexer_query(
                    pto["hidden"],
                    captured["qr"],
                    captured["qr_scale"].view(-1, 1),
                    weights["idx_wq_b"],
                    weights["idx_wq_b_scale"],
                    call.native_cos.view(-1, 64),
                    call.native_sin.view(-1, 64) * sign,
                    weights["hadamard_idx"],
                    before_hadamard,
                    query,
                    query_scale,
                )
                torch.npu.synchronize()
                report["precision_diagnostics"] = {
                    "qa_bf16": compare_tensor(qa[: args.batch * 6].bfloat16(), captured["qa_bf16"], 0, 0),
                    "qr": compare_tensor(qr, captured["qr"], 0, 0),
                    "qr_scale": compare_tensor(qr_scale, captured["qr_scale"].view(-1, 1), 0, 0),
                    "qr_given_native_qa": compare_tensor(normalized_qr, captured["qr"], 0, 0),
                    "qr_scale_given_native_qa": compare_tensor(
                        normalized_scale, captured["qr_scale"].view(-1, 1), 0, 0
                    ),
                    "indexer_before_hadamard_given_native_qr_and_rope": compare_tensor(
                        before_hadamard[:rows], captured["before_hadamard"].view(rows, 128), 0, 0
                    ),
                    "indexer_query_given_native_qr_and_rope": compare_tensor(
                        query[:rows], captured["query"].view(rows, 128), 0, 0
                    ),
                    "indexer_query_scale_given_native_qr_and_rope": compare_tensor(
                        query_scale[:rows], captured["query_scale"].view(rows, 1), 0, 0
                    ),
                }
                torch.save(
                    {
                        "qa_fp32": qa[: args.batch * 6].cpu(),
                        "qr": qr.cpu(),
                        "qr_scale": qr_scale.cpu(),
                        "qr_square_sum": qr_square_sum.cpu(),
                        "qr_inverse_rms": qr_inverse_rms.cpu(),
                        "qr_rms": qr_rms.cpu(),
                        "qr_given_native_qa": normalized_qr.cpu(),
                        "qr_scale_given_native_qa": normalized_scale.cpu(),
                        "before_hadamard": before_hadamard[:rows].cpu(),
                        "query": query[:rows].cpu(),
                        "query_scale": query_scale[:rows].cpu(),
                    },
                    args.output_dir / "pto_debug.pt",
                )
            report["output"] = compare_tensor(actual, expected, 1e-2, 1e-2)
            report["pto_untouched"] = check_untouched(pto, snapshots)
            report["cache_comparisons"] = {}
            for name, group in native["groups"].items():
                for index, reference in enumerate(group["views"]):
                    dtype = reference.dtype
                    tolerance = (
                        (0, 0)
                        if dtype == torch.int8
                        else (1e-3, 1e-3)
                        if dtype == torch.float32
                        else (1e-5, 1e-3)
                        if dtype == torch.float16
                        else (1e-2, 1e-2)
                    )
                    report["cache_comparisons"][f"{name}.{index}"] = compare_tensor(
                        pto["groups"][name]["views"][index], reference, *tolerance
                    )
            report["topk"] = compare_tensor(call.args["idx_topk"], captured["topk"].view(args.batch * 6, 512), 0, 0)
            torch.save(
                {
                    "native": expected.cpu(),
                    "pto": actual.cpu(),
                    "native_topk": captured["topk"].cpu(),
                    "pto_topk": call.args["idx_topk"].cpu(),
                    "pto_scores": call.args["idx_topk_scores"].cpu(),
                },
                args.output_dir / "outputs.pt",
            )
            checks = [
                report["output"],
                report["topk"],
                *report["cache_comparisons"].values(),
                *report["native_untouched"].values(),
                *report["pto_untouched"].values(),
            ]
            report["status"] = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
            if report["status"] != "PASS":
                raise AssertionError("Native/PTO full CSA comparison failed; inspect full_compare.json")
            print("Complete Native/PTO CSA comparison: PASS", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "full_compare.json", report)


if __name__ == "__main__":
    main()
