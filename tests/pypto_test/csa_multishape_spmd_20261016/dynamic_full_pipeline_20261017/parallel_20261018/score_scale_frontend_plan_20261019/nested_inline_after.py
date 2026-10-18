# pypto.program: Nested
import pypto.language as pl


@pl.program
class Nested:
    @pl.function
    def main(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        group_base_inline2_inline7: pl.Scalar[pl.INDEX] = (requests + 16) // 16 * 16
        group_end_inline1_inline6: pl.Scalar[pl.INDEX] = group_base_inline2_inline7 + 4 * tokens
        t__inline_arg_v0_inline5: pl.Scalar[pl.INDEX] = (group_end_inline1_inline6 + 15) // 16 * 16
        result_inline0: pl.Scalar[pl.INDEX] = t__inline_arg_v0_inline5 + 48 + 4 * tokens
        result: pl.Scalar[pl.INDEX] = result_inline0 + 2 * requests
        return result