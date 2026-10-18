# AIC Score G6完整step：独立动态源码准备

2026-10-18，formal668。已在`source_candidate/indexer_score_stages.py`新增`indexer_score_cube_step_g6`；**完整step数学、控制和FFTS子树经六个策略名逆归一化后与原文相同，原Cube/Vector两个整leaf函数逐字不变。**未接入schedule，未导入/编译PyPTO，未运行NPU，也未修改正式源、共享依赖或runner。

参考文件来自主仓`parallel_20261018/qb_global_n128_evidence/source_reference/indexer_score_stages.py`，与本次正式源及`oa_pair_manual_evidence/source_reference`直接字节对照相同；路径与大小记录在`reference_origin.json`。本目录只准备一个可放回完整候选包的模块，不是已安装或独立可运行的33文件包。

## 提取范围与接口

原`indexer_score_cube_stage`第174行的`for buf_score_step`外壳没有搬入。提取的是其175–375行完整正文：step≥2的双AIV ACK等待、首Key页装入、16个N64 panel的Key页保护/预取、QK与上一panel的WS次序、FP16舍入/ReLU、末WS和FIX READY。原14个顶层语句均保留；没有展开16个panel、重排MMA、改负页/超界保护、删分支数学或改valid_shape。

固定模块常量仅表示已有数学策略：G6、N64、score1024、balanced=True、prefetch=1、toL0=True。helper保留原if/unroll形态，不新增可变策略参数。T/B、请求Q/L、历史、页表、Native页字节数、leaf和step地址全部仍来自动态Tensor/设备Scalar。`buf_lane_span`虽然只参与随后被balanced分支覆盖的原地址式，仍显式保留，以便整个step直接AST对照，而非另写一份简化数学。

两个Tile入参为普通只读注解：

- `buf_query_left: pl.Tile[[384,128], pl.INT8, pl.MemorySpace.Left]`
- `buf_coefficient_pair: pl.Tile[[16,384], pl.FP16, pl.MemorySpace.Left]`

二者在正文中仅作为`pl.tile.matmul`左操作数，没有重绑、store或原位写。其余入参是四个原Native/cache Tensor、原InOut transfer Tensor、七个INDEX Scalar和两个FFTS事件constexpr；完整签名及自由名字检查见`source_audit.json`。返回值是原transfer Tensor别名，不返回片上临时，也不表示跨核数据已经消费。

## 调用者必须保留的合同

1. 外部沿原device group/leaf逻辑计算work_valid、cache_len、logical_begin、lane_span、step数和Native实际page_bytes。无工作叶不调用step；有效叶按原`step=0..iters-1`完整递增，不能切换Attention后重置到0。
2. 原leaf初始化仍在外部：只从GM加载live条query；先`set_validshape(384,128)`，再按原规则用首条有效query补齐片上尾行。每次step前，caller把驻Mat Query重新TMOV为**完整valid_shape的Left**传入；Coeff保持原完整16×384和无效query的零系数，不能以live行数缩小物理归约。这里不支持任意动态valid Tile。
3. Query、Coeff、Native key/scale、plan获取和FFTS域设置不在本helper里；原同组AIC/两AIV事件配对必须保留。READY仍用FIX发布，两AIV仍完整消费并ACK；空query lane不能跳过ACK。helper内部不发布phase或TopK根。
4. 叶最后的`min(iters,2)`个ACK drain仍留在原外部位置（参考第377–378行）。最后一步返回后先完整drain，才能重新从下个leaf的slot0开始；不能把READY或helper返回替代drain。AIV根完成另由root store后的release表达。
5. Tile注解不表示异步流水已完成。未来交错caller在重装Query或复用Mat/Left/Right/Acc前，仍须由真实lowering保留原MTE1/M/FIX依赖；本次没有增加fence来掩盖生命周期问题，也没有声称片上allocation不变。

独立JIT最小修复已证明普通Tile原样返回和模块常量shape入口可进入CPU，但同helper constexpr出现在Tile shape、任意动态valid Tile循环均不在该修复保证内。本文两个Tile使用固定数学shape和full-valid合同；**该Score helper本身仍未CPU编译**，不把最小identity结果扩大成完整Score/交错调用可用。

## 审计与复现

`prepare_source.py`只使用标准库：复制冻结参考，按token仅替换六个策略名字，生成helper，并直接比较完整step AST。删掉新增模块总览、常量与helper后可逐字恢复原文件；原两个函数也单独逐字对照。新增模块docstring段和helper docstring是明确变化，因此不称完整文件AST未变。检查还覆盖所有动态依赖显式传参、两Left只读使用、返回transfer和原ACK/READY站点。

复现时将本目录复制到新的私有目录，再在该目录运行（只重建同级源码/文本证据，不编译）：

```bash
python prepare_source.py > prepare_stdout.txt 2> prepare_stderr.txt
```

第一次参考复制的精确输入路径保存在`reference_origin.json`；之后仅使用同级冻结参考。结果为`source_audit.json`、`candidate.diff`及stdout/stderr。没有生成物或设备精度/性能结果。

主会话AIV三个helper的额外有限只读检查见`vector_interface_review.json`：检查原init/step/root-store三段AST、两个状态显式返回、ACK不受空lane数学guard限制、根发布必须在外层release之前。该检查不修改主文件、不替代后续联合lowering。
