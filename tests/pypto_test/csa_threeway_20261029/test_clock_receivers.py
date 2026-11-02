# SPDX-License-Identifier: Apache-2.0
"""诊断信号证明的拒绝路径；在227不占卡执行。"""

import unittest

from csa_threeway_20261029.clock_receivers import verify, verify_host


def fixture(root="v56", stride=128):
    return "\n".join(
        f"""AICORE void {role}(__gm__ int32_t* v56, __gm__ int32_t* v55) {{
  v100 = {stride};
  v101 = core * v100;
  v102 = GlobalTensor<int32_t>({root} + v101, shape, strides);
  pto::comm::TNOTIFY(v102, expected, operation);
}}
"""
        for role in ("cube_schedule", "vector_schedule")
    )


class ClockReceiverTests(unittest.TestCase):
    def test_owner_slot(self):
        self.assertEqual(verify(fixture())["vector_schedule"]["receivers"], 1)

    def test_foreign_allocation_rejected(self):
        with self.assertRaises((AssertionError, ValueError)):
            verify(fixture(root="v55"))

    def test_unaligned_slot_rejected(self):
        with self.assertRaises((AssertionError, ValueError)):
            verify(fixture(stride=127))

    def test_host_parameter_mapping(self):
        source = "const Tensor& ext_clock_signal = orch_args.tensor(55).ref();\nCoreTaskArgs params_t0;\n"
        source += (
            "params_t0.add_input(other);\n" * 55 + "params_t0.add_inout(ext_clock_signal);\n"
        )  # codespell:ignore inout
        self.assertEqual(verify_host(source)["mixed_signal_parameter"], 56)
        with self.assertRaises(AssertionError):
            verify_host(source.replace("tensor(55)", "tensor(54)"))
        with self.assertRaises(AssertionError):
            verify_host(source.replace("params_t0.add_input(other);\n", "", 1))


if __name__ == "__main__":
    unittest.main()
