# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Observe local handoffs in the fixed chain; this is not a performance sample."""

import argparse
import json
import os
from pathlib import Path

import task_boundary_acceptance as acceptance


_TRACE_PROGRAM = """import pypto.language as pl

@pl.program
class TraceProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def task(self, data: pl.InOut[pl.Tensor[[27664,512],pl.FP32]],
             trace: pl.Out[pl.Tensor[[48,8],pl.INT64]], stage: pl.Scalar[pl.INT64],
             work: pl.Scalar[pl.INT64], scalar_steps: pl.Scalar[pl.INT64], seed: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[27664,512],pl.FP32],pl.Tensor[[48,8],pl.INT64]]:
        pl.func_attr({"async_return_trace": trace})
        block = pl.tile.get_block_idx()
        state = seed + stage + block
        for _ in pl.range(scalar_steps):
            state = (state * 17 + 3) % 65521
        pl.system.wait_pre_task_end()
        value = pl.load(data, [(stage * 48 + block) * 32, 0], [32,512])
        value = pl.add(value, pl.cast(state % 17, pl.FP32))
        for _ in pl.range(work):
            value = pl.add(value, 1.0)
        data = pl.store(value, [((stage + 1) * 48 + block) * 32,0],data)
        pl.system.set_next_task_start()
        return data, trace

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pl.InOut[pl.Tensor[[27664,512],pl.FP32]],
             trace: pl.Out[pl.Tensor[[817,8],pl.INT64]], length: pl.Scalar[pl.INT64],
             work: pl.Scalar[pl.INT64], scalar_steps: pl.Scalar[pl.INT64], seed: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[27664,512],pl.FP32],pl.Tensor[[817,8],pl.INT64]]:
        for stage in pl.range(length):
            stage_trace = pl.slice(trace,[48,8],[stage * 48,0])
            with pl.spmd(48, allow_early_resolve=True):
                data, stage_trace = self.task(data,stage_trace,stage,work,scalar_steps,seed)
        return data, trace
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compile', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    def traced_program(directory, mode):
        import importlib.util
        source = _TRACE_PROGRAM
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "program.py"
        if path.exists():
            assert path.read_text() == source
        else:
            path.write_text(source)
        spec = importlib.util.spec_from_file_location("fixed_chain_trace", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TraceProgram

    acceptance.make_program = traced_program
    if args.action == 'compile':
        acceptance.artifact(root / 'compiled', 'early', compile_now=True)
        return
    assert os.environ.get('TASK_DEVICE') == str(args.device)
    import torch
    import torch_npu
    from simpler_setup.runtime_builder import RuntimeBuilder
    from pypto.runtime.kernel.context import get_process_kernel_state
    from pypto.torch import init
    from pypto.torch.launch import enqueue

    RuntimeBuilder._CACHE_DIR = args.runtime_root / 'runtime/cache'
    RuntimeBuilder._LIB_DIR = args.runtime_root / 'runtime/lib'
    torch.set_num_threads(2)
    torch_npu.npu.set_device(args.device)
    init(device=args.device, runtime='tensormap_and_ringbuffer', ring_task_window=128)
    runtime = get_process_kernel_state()
    acceptance.SCALAR_STEPS = 256
    rows = []
    try:
        kernel = runtime.ensure_callable(acceptance.artifact(root/'compiled', 'early'), runtime.require_config())
        data = torch.empty((acceptance.ROWS, acceptance.COLS), device=f'npu:{args.device}')
        trace = torch.empty((817,8), dtype=torch.int64, device=data.device)
        initial = (torch.arange(48 * 32 * 512, dtype=torch.float32) % 31).reshape(-1,512)
        for work in (5,167):
            data.fill_(float('nan')); data[:1536].copy_(initial); trace.fill_(-1)
            enqueue(kernel, (data, trace, 17, work, 256, 7))
            torch_npu.npu.synchronize()
            acceptance.verify(data, initial, work, 17)
            values = trace.cpu().tolist()
            assert values[-1] == [-1]*8
            assert all(row[4] & 255 == 7 for row in values[:-1])
            entries=[dict(stage=i//48,block=i%48,core=v[0],start=v[1],returned=v[2],end=v[3],
                          flags=v[4],wait_start=v[5],wait_ticks=v[6],compute_end=v[7]) for i,v in enumerate(values[:-1])]
            cores={e['core'] for e in entries}; pairs=[]
            for core in sorted(cores):
                ordered=sorted((e for e in entries if e['core']==core),key=lambda e:e['start'])
                for before,after in zip(ordered,ordered[1:]):
                    pairs.append(dict(core=core,previous_stage=before['stage'],previous_block=before['block'],
                        next_stage=after['stage'],next_block=after['block'],
                        overlap_before_fin=after['start']<before['end'],
                        prefix_before_wait_us=(before['wait_start']-after['start'])/50,
                        remaining_wait_us=before['wait_ticks']/50))
            rows.append(dict(work=work,device=args.device,entries=entries,pairs=pairs,
                local_pairs=len(pairs),before_fin_count=sum(p['overlap_before_fin'] for p in pairs),
                notifications=sum(bool(e['flags'] & 256) for e in entries)))
    finally:
        runtime.drain();runtime.close()
    (root/'results.json').write_text(json.dumps(dict(diagnostic_only=True,output_and_guards_exact=True,runs=rows),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('entries','pairs')} for r in rows],indent=2))


if __name__ == '__main__':
    main()
