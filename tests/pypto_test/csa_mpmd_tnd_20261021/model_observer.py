# SPDX-License-Identifier: Apache-2.0
"""16卡功能验证的TND路由和代际观测；不用于性能计时。"""

from offline_pd.observer import OfflineCSAObserver

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import TNDGraphContract


def install_graph_observer():
    original = TNDGraphContract.allows_replay
    if getattr(original, "_tnd_validation_observer", False):
        return

    def counted(contract, context):
        allowed = original(contract, context)
        runtime = contract.runtime
        if not hasattr(runtime, "_validation_graph_counts"):
            runtime._validation_graph_counts = {"allowed": 0, "rejected": 0}
        runtime._validation_graph_counts["allowed" if allowed else "rejected"] += 1
        return allowed

    counted._tnd_validation_observer = True
    TNDGraphContract.allows_replay = counted


install_graph_observer()


class TNDModelObserver(OfflineCSAObserver):
    def _tnd_runtime_rows(self, read_generation=False):
        rows = {}
        for index, layer in enumerate(self.model_runner.get_model().model.layers):
            if layer.self_attn.compress_ratio != 4:
                continue
            runtime = getattr(layer.self_attn.dsa_attn, "_pto_csa_runtime", None)
            if runtime is None:
                raise RuntimeError(f"C4 layer{index}没有安装TND runtime")
            row = {
                "runtime_class": type(runtime).__module__,
                "submissions": runtime.submissions,
                "empty_calls": runtime.empty_calls,
                "bindings": len(runtime._bindings),
                "graph_contract_checks": dict(getattr(runtime, "_validation_graph_counts", {})),
            }
            if read_generation:
                # 全部DP结束生成后在RPC读取，不放到forward，也不参与调度或资源选择。
                # 仅证明PTO协议实际推进，不能当作全部cache内容一致的证明。
                prepared = runtime.last_binding
                row["generation"] = {}
                if prepared is not None:
                    policy = prepared.policy
                    row["policy"] = policy.constexprs()
                    counts = {
                        "signal": 72,
                        "middle_signal": 72,
                        "output_signal": 72,
                        "hc_signal": 3 * getattr(policy, "hc_groups", 24),
                        "idx_front_signal": 3 * getattr(policy, "idx_front_groups", 8),
                        "main_front_signal": 3 * getattr(policy, "main_front_groups", 16),
                    }
                    for name, count in counts.items():
                        tensor = prepared.buffers.get(name)
                        if tensor is not None:
                            values = tensor[: count * 128 : 128].cpu().tolist()
                            row["generation"][name] = {"min": min(values), "max": max(values), "participants": count}
            rows[str(index)] = row
        return rows

    def offline_begin_observation(self):
        result = super().offline_begin_observation()
        self._tnd_before = self._tnd_runtime_rows()
        return result

    def offline_end_observation(self):
        result = super().offline_end_observation()
        result["tnd_before"] = self._tnd_before
        result["tnd_after"] = self._tnd_runtime_rows(read_generation=True)
        return result
