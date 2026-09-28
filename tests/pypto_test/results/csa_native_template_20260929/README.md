# 对齐Native decode模板：配置及依赖

完整差异和验收边界见[Native基线](../../DSV4_FLASH_CSA_NATIVE_BASELINE.md)。
2026-09-29已检查实际LLM构造参数、16rank环境传递及Worker配置返回：10项CPU回归通过。
没有启动16卡基线测试；先补齐模板默认融合需要的AddRmsNormBias。

指定CANN9.2 libopapi.so、现有CSA libcust_opapi.so均不导出aclnnAddRmsNormBias；
当前release包含csrc/moe/add_rms_norm_bias，按原算子构建，不改其计算实现。
独立build/stage目录为.cache/csa/native-template-cann92-build及native-template-cann92-stage，
vendor名csa_template。只构建缺失算子，复用本地third_party；旧Native包与共享环境未覆盖。
构建/配置日志为本目录configure.log、build.log、install.log；独立包已编译安装成功，
导出aclnnAddRmsNormBias与GetWorkspaceSize。复现入口[build_dependency.sh](build_dependency.sh)。

[单卡探针](probe.py)验证真实算子调用、Native完整norm/quant融合注册及实际static compiler调用，
随后检查图重放；不加载全模型，不导入PyPTO，不产出性能结论。
[task.txt](task.txt)为有效任务；修正探针类名时曾取消尚未启动的004215任务，其未占卡执行。
最终任务task_20260929_010148_105901111670已完成exit=0，auto分配card1；[结果](probe.json)。
实际AddRmsNormBias、融合注册、static_compile返回True、非空静态包安装与图重放全部通过。
只在探针进程追加补充vendor，公共环境尚未切换；不能将“构建成功”称为模板基线已完成。


## 静态编译中的实际缺口

第一次探针用全静态Tensor输入，npugraph_ex的vLLM sym_range只接受符号token维度，故跳过编译。
按真实编译入口标注token维度动态后，编译被调用，但op_compiler在打包时返回1；
CANN9.2的OPP是其他用户的只读目录，工具要求OPP可写，不能仅凭编译函数返回就判通过。
permission_probe.json的旧PASS只覆盖函数返回/重放，已由[失败判定](permission_assessment.json)撤回；
当前探针同时要求static_compile=True和非空已安装包。

没有修改共享CANN权限。用[prepare_opp.py](prepare_opp.py)创建私有可写OPP根，
其余payload指向同一CANN9.2；静态kernel安装到私有static_kernel目录。
浅层built-in软链接会导致Cast tiling未注册，因此将其op_tiling目录链实体化，
复制原始两份tiling库共约40MiB；不改变库内容和算子实现。
最终设备任务实际通过，失败阶段日志和JSON保留用于解释环境要求，不作为性能证据。

后续两侧运行前：source公共env-dsv4-0251rc1.sh，再source本目录[env.sh](env.sh)。
尚未改共享环境或启动16卡；真正CSA编译路径与新配置性能基线仍待完成。
