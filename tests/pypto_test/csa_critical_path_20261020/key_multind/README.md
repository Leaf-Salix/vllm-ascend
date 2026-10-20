# 当前双QK流水的Key跨页多ND搬运

用户要求本轮优化结束后暂停。本目录只推进这一个候选，不开下一方向或
参数扫描；未满足组件、设备和完整性能验证前不迁入正式算子。

## 与历史工作的区别

正式算子13259f92的Indexer每个N64面板仍逐页搬两份32×128 INT8 Key。
Native每页Key后有scale，直接把两页当成连续64×128会读错页尾。
现有前置连续页证明可以选择快速路径，但只复用地址的旧候选并未减少
两次DMA，B28回退3.46μs，不能将它再包装为新方案。

本次拟在device页表确认正序相邻且不触发尾页重复钳制时，以三维GM视图
`[P,32,128]`和stride`[page_bytes,128,1]`一次多ND→NZ载入两页Key。
目标Mat仍64×128，其余预取槽、QK/WS数学和双槽交接保持。非相邻、逆序
或尾页必须走原逐页回退；不修改Native缓存布局，不复制Key，不新增Host
页表取值。最终是否省20μs必须由同轮完整CSA五窗证明。

旧indexer_page_coalesce_evidence在Torch入口正确拒绝非连续根Tensor，
并已提交PyPTO #3038；本轮保留连续Native根，在incore内部建立视图。
旧pair_dma_full_pipeline处理Attention任意Top-K地址，带地址排序和UB
重排，本轮Key两页原序直达Mat，不重复该负例。

## 隔离实现和能力边界

新树`.worktrees/pypto-csa-key-multind-20261020`从正式配套2dfa7f12创建，
没有改共享安装、正式依赖树、Simpler或PTOAS。所有构建在本树，未复制
扩展或构建产物；BUILD_COMMAND.json记录loader给定的build/test并发2。

原型共14文件：DSL和IR wrapper增加显式strides操作数；类型推导保持ND
非重叠约束，检查正stride、rank、静态可访问跨度及int64溢出；仅支持
packed完全有效本地根和完全有效目标，不支持FP4/分布式/编排侧视图。
符号维度延续既有动态slice/view的调用方边界合同，不能声称编译器已证明
所有运行时地址：正式Native binding必须继续检查页宽和容量，设备分支
继续检查页表和物理页。普通根的contiguous校验和普通view默认语义保留。

BackendHandler显式声明A2/A3多ND Mat能力；FlattenTileNdTo2D仅对带页间
空隙的三维ND Mat源保留原窗口/stride，目标Tile仍二维。packed源仍走原
折叠，未给所有形状或所有后端开放任意地址布局。

同一根view不需要旧GM动态slice特性#3042。若当前shape/stride引用触发
已知#3046定义失配，需要独立确认后只迁移相应修复，不整套复制旧原型。

## 验证顺序与停止条件

1. 独立构建和有针对性的view/Mat load回归：合法padding，非法重叠、
   非正stride、超范围、溢出、普通view与入口保护不变。
2. 完整CPU编译probe.py，核对真实三维源stride、两页窗口和二维Mat，
   不能仅以API存在或lower成功宣称搬运可用。
3. 队列上验证4160/4224/8192页宽，固定图页号0→1→7→0；INT8点积全部
   精确，输入不变，输出两端保护保持。本probe不采性能、不代表完整CSA。
4. 只有上述真实机制成立，再在当前正式源上接入单个候选并做重点档完整
   三方五窗。至少一重点档净省20μs、另一档无明确退步才有采用依据；
   冻结精确/状态/保护/metadata/padding/图/代际不变。

若能力无法合理实现或完整收益不成立，记录证据结束本轮；不把局部probe
PASS代替完整性能，不放宽30%总目标，不为了本轮结束而宣布总目标完成。
本轮结束后按用户要求调用goal paused。所有仓库仍禁止push。

## 初始及中间状态（历史，最终结论见文末）

接口与有针对性的测试、英文/中文文档已写，独立构建session44151进行中；
当前无设备任务。观察超时不能重启同一构建。build_v1.log保存日志。
尚未有完整CPU、UT、设备或性能PASS，正式37文件保持原版。

准备probe时进一步对齐完整CSA目标：多ND N64加载后放入128×128 Mat双槽，
按device页号奇偶选择槽0/64，再从相同父Tile提取Right做INT8点积。
实际生成必须将多ND直接写入相应Mat槽，不能产生不支持的Mat→Mat复制；
这一步尚未编译或上卡，不能把独立64×128加载当作双槽接入证明。
底层srcNdMatrixStride为16位，实际快速路径还须限定page_bytes<=65535。

首轮独立构建成功。UT 53通过/1失败，失败是新测试把三维tile和二维reshape
复用同一变量名，改为flat_tile，不改接口或测试预期。CPU v1中内联tensor.dim
保留在TensorView类型里，PTO字段变空；v2先绑定device Scalar再建view，
同时继续核对Mat双槽问题。v1已生成额外Mat→Mat搬运，不能声称直接落点。

UT v2已54项通过。CPU v2绑定Scalar后已越过描述符空字段，随后明确被
PTOAS拒绝Mat→Mat TMOV；该双槽组装草案停止，不绕过后端限制。v3基础
probe改为N64 Mat直接送Right，取消父Mat组装；完整候选若继续必须同步
使用该直接路径，并核对原双Right槽与MTE依赖，不冒称原Mat双槽已验证。

基础NPU task_20261020_101730_37553223693 completed/exit0，三页宽共12项
同址图、INT8点积、输入不变和保护检查通过，无性能采样。完整候选仅改
正式Score的Key供数；两份8KiB Right预取保持，原128行父Mat变为直接N64
Mat；循环prefetch Mat实际地址复用可能改变MTE1/MTE2依赖，不能假定无成本。
完整CPU候选和新编译器下正式参考均58轴/130根/单MIX通过，所有资源上限
保持。正式参考C++与原工具链仅去注释/空白后相同，不把新接口混为基线优化。
B28 task_20261020_102448_407774415351已提交三方五窗，待终态。

内联dim问题已用原接口在正式父工具链独立复现，完整CPU失败，说明不是
新strides接口才出现；相近既有#2951已关闭，将以相关变体补充，不重复提
issue，也不宣称已在干净最新main复现。原始CPU/PTO与建议正文均保留。

## 2026-10-20 10:38 最终结论：B28功能通过，完整性能回退，本轮结束

详见 [FINAL_REPORT.md](FINAL_REPORT.md)。正式37文件保持13259f92内容。
独立编译器构建、54项UT、三页宽12项设备probe、完整符号编译均通过；
完整B28/S6/128K任务终态exit0，冻结数值、状态、保护、metadata/padding
图与代际通过。同轮五窗多task/正式/候选为1289.56/1163.64/1310.50μs，
净省-146.86μs，0/5更快。不采用、不扩档、不扫参数，按用户要求归档后
暂停goal。Mat父双槽改为直接N64路径，不能把退步全部归因于multi-ND。
新旧代码生成及实际设备代码审计见FULL_CPU_AUDIT_V1、DEVICE_SOURCE_AUDIT。
所有源和补丁、失败证据保留；权重/bin/大编译产物不入Git，不做hash。
上游评论已提交，未提交PR、未加标签、所有仓库均未push。
