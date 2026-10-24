# SPDX-License-Identifier: Apache-2.0
"""CPU回归：EOS完成不能与达到上限混为一谈。"""

import unittest

from check import check_completion


class CompletionTest(unittest.TestCase):
    def test_eos_and_limit_are_distinct(self):
        request = dict(output_token_ids=[1, 2], output_text="answer", finished=True, finish_reason="stop")
        self.assertEqual(check_completion(request, 2048, True), "stop")
        request.update(output_token_ids=[1] * 2048, finish_reason="length")
        self.assertEqual(check_completion(request, 2048, True), "length")
        request["output_token_ids"] = [1]
        with self.assertRaises(AssertionError):
            check_completion(request, 2048, True)

    def test_unfinished_and_legacy_short_output_fail(self):
        request = dict(output_token_ids=[1, 2], output_text="answer", finished=False, finish_reason="stop")
        with self.assertRaises(AssertionError):
            check_completion(request, 2048, True)
        with self.assertRaises(AssertionError):
            check_completion(request, 192, False)


if __name__ == "__main__":
    unittest.main()
