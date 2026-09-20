"""hc_pre→hc_post 的整条 attention 半层（动态形状，TP1/TP2/TP4），外加设备侧 metadata kernel。

上游：hw-native-sys/pypto-lib  models/deepseek_v4_flash_dspark/
入口模块：
  decode_csa       —— decode_csa_tp1_test（单卡）/ decode_csa_test（TP≥2）
  decode_metadata  —— 设备侧一次算出全部 slot_mapping + swa_indices / swa_lens
"""

UPSTREAM_SUBDIR = "models/deepseek_v4_flash_dspark"
ENTRY_MODULES = ("decode_csa", "decode_metadata")
