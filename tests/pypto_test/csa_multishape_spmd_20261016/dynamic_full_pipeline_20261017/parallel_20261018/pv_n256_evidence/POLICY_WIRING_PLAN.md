# PV区间策略的最小接线方案（仅设计，尚未实施）

不选择启用阈值。只有后续区间实测形成依据才考虑实现，不能用B4/B16白名单。

正式最小涉及4个文件：

- policy.py：新增不可变字段pv_cols，默认128；constexprs映射为PV_N_TILE。
  精确限制int且只能128/256，未来仅由select_policy按真实工作量区间选择。
- entry.py：decode_csa_spmd和_csa_core_group均增加constexpr，root→group→cube
  两处显式转传；不转给vector_schedule。
- cube_schedule.py：签名与Attention调用转传PV_N_TILE。
- attention_stream_stages.py：cube helper用constexpr参数取代文件级固定常量，
  尾部可默认128兼容旧caller。extract、512//tile列循环、store偏移共用参数。

两值均整除512；K128、三槽、GM pv_ring[...,512]、AIV两段256列读取和FFTS
协议不变。binding.py已整体**self.policy.constexprs()，无需改workspace或
逐字段适配。只增加编译特化键，不增加runtime标量或Tensor根，仍130根、
58动态轴、60phase；这项须在真正实现时用完整符号编译核实，非本设计已验收。

测试入口：check_cpu.py默认controls补PV_N_TILE=128，其--controls-json直接
update、不做白名单过滤；run_device.py的dataclasses.replace会保留新增policy
字段，--policy-json使用小写pv_cols。source_loader不额外过滤。source matrix
候选使用自身policy，reference重新调用参考包select_policy，仍默认128，
不可把候选覆盖误传旧参照。check_schedule.py解释模型controls和
source_interface.json.controls同步补字段，runtime_tensors不变。

旧assemble_source.py属于历史生成器，不为此重新生成静态入口。此方案由独立
agent只读提供，没有修改正式源、没有编译或设备任务，不能当作实现完成。


最终状态：2026-10-18七档和区间点已经完成，长G6仍有回退，策略不采用、不扩测。此计划保留为历史；两CPU接线证据见 ../pv_policy_wiring_evidence/README.md。
