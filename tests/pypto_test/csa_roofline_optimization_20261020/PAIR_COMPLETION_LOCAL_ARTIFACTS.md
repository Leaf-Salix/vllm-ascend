# 成对压缩完整验证的本地大产物

完整源码及脚本、结果、五窗L1、组件补丁入Git。大编译快照、ELF/bin、权重和pt不提交，不做hash。

根目录：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`

- `tests/pypto_test/csa_roofline_optimization_20261020/pair_cpu_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_cpu_b28_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_micro_ccec_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_ccec_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_ptoas_regressions_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_legacy_ptoas_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_direct_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_direct_v2`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_direct_cpu_v2`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_merge_device_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b16_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b28_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/validshape_nested_read_v1`：保留完整原始构建、编译dump及实际CCEC输入。
- `tests/pypto_test/csa_roofline_optimization_20261020/validshape_nested_read_v2`：保留完整原始构建、编译dump及实际CCEC输入。

未纳入Git的大文本：
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b16_v1/codegen_probe/modified.cpp`：2720819 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b16_v1/codegen_probe/original.cpp`：2719775 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b16_v1/source_matrix/spmd_reference_codegen/modified.cpp`：2581460 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b16_v1/source_matrix/spmd_reference_codegen/original.cpp`：2580416 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b28_v1/codegen_probe/modified.cpp`：2756499 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b28_v1/codegen_probe/original.cpp`：2755455 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b28_v1/source_matrix/spmd_reference_codegen/modified.cpp`：2617134 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_b28_v1/source_matrix/spmd_reference_codegen/original.cpp`：2616090 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_cpu_b28_v1/kernel/kernels/aic/cube_schedule.cpp`：2799521 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_cpu_b28_v1/kernel/kernels/aiv/vector_schedule.cpp`：2799687 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_cpu_b28_v1/kernel/ptoas/dynamic_csa_full.cpp`：2749135 字节。
- `tests/pypto_test/csa_roofline_optimization_20261020/pair_full_cpu_b28_v1/kernel/ptoas/dynamic_csa_full.pto`：4609703 字节。
