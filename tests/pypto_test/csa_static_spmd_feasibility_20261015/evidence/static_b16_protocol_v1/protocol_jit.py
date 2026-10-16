# 诊断抽取：仅进度协议，无 CSA 数学，不作为完整算子通过证据。
import pypto.language as pl

@pl.jit.incore
def csa_static_aic(progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> pl.Tensor[[4608], pl.INT64]:
    pl.func_attr({'mx_tensor_views_blocked': True})
    failed = 0
    group_id = pl.tile.get_block_idx()
    core_id = group_id
    own_flag = core_id * 64
    pl.system.cacheinvalid(progress, [1], [own_flag])
    prior_progress = pl.tensor.read(progress, [own_flag])
    generation_base = (prior_progress // 1024 + 1) * 1024
    pl.tensor.write(progress, [own_flag], generation_base)
    pl.system.cacheinvalid(progress, [1], [own_flag])
    pl.system.fence()
    record_0 = 0 + core_id * 6
    mapped_block_0 = pl.cast(pl.tensor.read(records, [record_0]), pl.INDEX)
    block_0 = pl.max(mapped_block_0, 0)
    if mapped_block_0 >= 0 and failed == 0:
        subslot_0 = pl.cast(pl.tensor.read(records, [record_0 + 1]), pl.INDEX)
        done_0 = pl.tensor.read(records, [record_0 + 2])
        wait_base_0 = pl.cast(pl.tensor.read(records, [record_0 + 3]), pl.INDEX)
        wait_count_0 = pl.cast(pl.tensor.read(records, [record_0 + 4]), pl.INDEX)
        publish_0 = pl.tensor.read(records, [record_0 + 5])
        for dependency_0 in pl.range(wait_count_0):
            if failed == 0:
                wait_index_0 = (wait_base_0 + dependency_0) * 2
                other_0 = pl.cast(pl.tensor.read(waits, [wait_index_0]), pl.INDEX)
                required_0 = pl.tensor.read(waits, [wait_index_0 + 1])
                target_0 = generation_base + required_0
                flag_0 = other_0 * 64
                pl.system.cacheinvalid(progress, [1], [flag_0])
                initial_0 = pl.tensor.read(progress, [flag_0])
                for (ready_0, polls_0) in pl.while_(init_values=(initial_0, 0)):
                    pl.cond(ready_0 < target_0 and polls_0 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_0])
                    (after_0, spin_count_0) = pl.yield_(pl.tensor.read(progress, [flag_0]), polls_0 + 1)
                if after_0 < target_0:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 0)
                    pl.tensor.write(progress, [own_flag + 10], other_0)
                    pl.tensor.write(progress, [own_flag + 11], target_0)
                    pl.tensor.write(progress, [own_flag + 12], after_0)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_0)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_0 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_0)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_1 = 432 + core_id * 6
    mapped_block_1 = pl.cast(pl.tensor.read(records, [record_1]), pl.INDEX)
    block_1 = pl.max(mapped_block_1, 0)
    if mapped_block_1 >= 0 and failed == 0:
        subslot_1 = pl.cast(pl.tensor.read(records, [record_1 + 1]), pl.INDEX)
        done_1 = pl.tensor.read(records, [record_1 + 2])
        wait_base_1 = pl.cast(pl.tensor.read(records, [record_1 + 3]), pl.INDEX)
        wait_count_1 = pl.cast(pl.tensor.read(records, [record_1 + 4]), pl.INDEX)
        publish_1 = pl.tensor.read(records, [record_1 + 5])
        for dependency_1 in pl.range(wait_count_1):
            if failed == 0:
                wait_index_1 = (wait_base_1 + dependency_1) * 2
                other_1 = pl.cast(pl.tensor.read(waits, [wait_index_1]), pl.INDEX)
                required_1 = pl.tensor.read(waits, [wait_index_1 + 1])
                target_1 = generation_base + required_1
                flag_1 = other_1 * 64
                pl.system.cacheinvalid(progress, [1], [flag_1])
                initial_1 = pl.tensor.read(progress, [flag_1])
                for (ready_1, polls_1) in pl.while_(init_values=(initial_1, 0)):
                    pl.cond(ready_1 < target_1 and polls_1 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_1])
                    (after_1, spin_count_1) = pl.yield_(pl.tensor.read(progress, [flag_1]), polls_1 + 1)
                if after_1 < target_1:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 1)
                    pl.tensor.write(progress, [own_flag + 10], other_1)
                    pl.tensor.write(progress, [own_flag + 11], target_1)
                    pl.tensor.write(progress, [own_flag + 12], after_1)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_1)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_1 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_1)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_2 = 864 + core_id * 6
    mapped_block_2 = pl.cast(pl.tensor.read(records, [record_2]), pl.INDEX)
    block_2 = pl.max(mapped_block_2, 0)
    if mapped_block_2 >= 0 and failed == 0:
        subslot_2 = pl.cast(pl.tensor.read(records, [record_2 + 1]), pl.INDEX)
        done_2 = pl.tensor.read(records, [record_2 + 2])
        wait_base_2 = pl.cast(pl.tensor.read(records, [record_2 + 3]), pl.INDEX)
        wait_count_2 = pl.cast(pl.tensor.read(records, [record_2 + 4]), pl.INDEX)
        publish_2 = pl.tensor.read(records, [record_2 + 5])
        for dependency_2 in pl.range(wait_count_2):
            if failed == 0:
                wait_index_2 = (wait_base_2 + dependency_2) * 2
                other_2 = pl.cast(pl.tensor.read(waits, [wait_index_2]), pl.INDEX)
                required_2 = pl.tensor.read(waits, [wait_index_2 + 1])
                target_2 = generation_base + required_2
                flag_2 = other_2 * 64
                pl.system.cacheinvalid(progress, [1], [flag_2])
                initial_2 = pl.tensor.read(progress, [flag_2])
                for (ready_2, polls_2) in pl.while_(init_values=(initial_2, 0)):
                    pl.cond(ready_2 < target_2 and polls_2 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_2])
                    (after_2, spin_count_2) = pl.yield_(pl.tensor.read(progress, [flag_2]), polls_2 + 1)
                if after_2 < target_2:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 2)
                    pl.tensor.write(progress, [own_flag + 10], other_2)
                    pl.tensor.write(progress, [own_flag + 11], target_2)
                    pl.tensor.write(progress, [own_flag + 12], after_2)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_2)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_2 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_2)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_3 = 1296 + core_id * 6
    mapped_block_3 = pl.cast(pl.tensor.read(records, [record_3]), pl.INDEX)
    block_3 = pl.max(mapped_block_3, 0)
    if mapped_block_3 >= 0 and failed == 0:
        subslot_3 = pl.cast(pl.tensor.read(records, [record_3 + 1]), pl.INDEX)
        done_3 = pl.tensor.read(records, [record_3 + 2])
        wait_base_3 = pl.cast(pl.tensor.read(records, [record_3 + 3]), pl.INDEX)
        wait_count_3 = pl.cast(pl.tensor.read(records, [record_3 + 4]), pl.INDEX)
        publish_3 = pl.tensor.read(records, [record_3 + 5])
        for dependency_3 in pl.range(wait_count_3):
            if failed == 0:
                wait_index_3 = (wait_base_3 + dependency_3) * 2
                other_3 = pl.cast(pl.tensor.read(waits, [wait_index_3]), pl.INDEX)
                required_3 = pl.tensor.read(waits, [wait_index_3 + 1])
                target_3 = generation_base + required_3
                flag_3 = other_3 * 64
                pl.system.cacheinvalid(progress, [1], [flag_3])
                initial_3 = pl.tensor.read(progress, [flag_3])
                for (ready_3, polls_3) in pl.while_(init_values=(initial_3, 0)):
                    pl.cond(ready_3 < target_3 and polls_3 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_3])
                    (after_3, spin_count_3) = pl.yield_(pl.tensor.read(progress, [flag_3]), polls_3 + 1)
                if after_3 < target_3:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 3)
                    pl.tensor.write(progress, [own_flag + 10], other_3)
                    pl.tensor.write(progress, [own_flag + 11], target_3)
                    pl.tensor.write(progress, [own_flag + 12], after_3)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_3)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_3 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_3)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_4 = 1728 + core_id * 6
    mapped_block_4 = pl.cast(pl.tensor.read(records, [record_4]), pl.INDEX)
    block_4 = pl.max(mapped_block_4, 0)
    if mapped_block_4 >= 0 and failed == 0:
        subslot_4 = pl.cast(pl.tensor.read(records, [record_4 + 1]), pl.INDEX)
        done_4 = pl.tensor.read(records, [record_4 + 2])
        wait_base_4 = pl.cast(pl.tensor.read(records, [record_4 + 3]), pl.INDEX)
        wait_count_4 = pl.cast(pl.tensor.read(records, [record_4 + 4]), pl.INDEX)
        publish_4 = pl.tensor.read(records, [record_4 + 5])
        for dependency_4 in pl.range(wait_count_4):
            if failed == 0:
                wait_index_4 = (wait_base_4 + dependency_4) * 2
                other_4 = pl.cast(pl.tensor.read(waits, [wait_index_4]), pl.INDEX)
                required_4 = pl.tensor.read(waits, [wait_index_4 + 1])
                target_4 = generation_base + required_4
                flag_4 = other_4 * 64
                pl.system.cacheinvalid(progress, [1], [flag_4])
                initial_4 = pl.tensor.read(progress, [flag_4])
                for (ready_4, polls_4) in pl.while_(init_values=(initial_4, 0)):
                    pl.cond(ready_4 < target_4 and polls_4 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_4])
                    (after_4, spin_count_4) = pl.yield_(pl.tensor.read(progress, [flag_4]), polls_4 + 1)
                if after_4 < target_4:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 4)
                    pl.tensor.write(progress, [own_flag + 10], other_4)
                    pl.tensor.write(progress, [own_flag + 11], target_4)
                    pl.tensor.write(progress, [own_flag + 12], after_4)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_4)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_4 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_4)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_5 = 2160 + core_id * 6
    mapped_block_5 = pl.cast(pl.tensor.read(records, [record_5]), pl.INDEX)
    block_5 = pl.max(mapped_block_5, 0)
    if mapped_block_5 >= 0 and failed == 0:
        subslot_5 = pl.cast(pl.tensor.read(records, [record_5 + 1]), pl.INDEX)
        done_5 = pl.tensor.read(records, [record_5 + 2])
        wait_base_5 = pl.cast(pl.tensor.read(records, [record_5 + 3]), pl.INDEX)
        wait_count_5 = pl.cast(pl.tensor.read(records, [record_5 + 4]), pl.INDEX)
        publish_5 = pl.tensor.read(records, [record_5 + 5])
        for dependency_5 in pl.range(wait_count_5):
            if failed == 0:
                wait_index_5 = (wait_base_5 + dependency_5) * 2
                other_5 = pl.cast(pl.tensor.read(waits, [wait_index_5]), pl.INDEX)
                required_5 = pl.tensor.read(waits, [wait_index_5 + 1])
                target_5 = generation_base + required_5
                flag_5 = other_5 * 64
                pl.system.cacheinvalid(progress, [1], [flag_5])
                initial_5 = pl.tensor.read(progress, [flag_5])
                for (ready_5, polls_5) in pl.while_(init_values=(initial_5, 0)):
                    pl.cond(ready_5 < target_5 and polls_5 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_5])
                    (after_5, spin_count_5) = pl.yield_(pl.tensor.read(progress, [flag_5]), polls_5 + 1)
                if after_5 < target_5:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 5)
                    pl.tensor.write(progress, [own_flag + 10], other_5)
                    pl.tensor.write(progress, [own_flag + 11], target_5)
                    pl.tensor.write(progress, [own_flag + 12], after_5)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_5)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_5 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_5)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_6 = 2592 + core_id * 6
    mapped_block_6 = pl.cast(pl.tensor.read(records, [record_6]), pl.INDEX)
    block_6 = pl.max(mapped_block_6, 0)
    if mapped_block_6 >= 0 and failed == 0:
        subslot_6 = pl.cast(pl.tensor.read(records, [record_6 + 1]), pl.INDEX)
        done_6 = pl.tensor.read(records, [record_6 + 2])
        wait_base_6 = pl.cast(pl.tensor.read(records, [record_6 + 3]), pl.INDEX)
        wait_count_6 = pl.cast(pl.tensor.read(records, [record_6 + 4]), pl.INDEX)
        publish_6 = pl.tensor.read(records, [record_6 + 5])
        for dependency_6 in pl.range(wait_count_6):
            if failed == 0:
                wait_index_6 = (wait_base_6 + dependency_6) * 2
                other_6 = pl.cast(pl.tensor.read(waits, [wait_index_6]), pl.INDEX)
                required_6 = pl.tensor.read(waits, [wait_index_6 + 1])
                target_6 = generation_base + required_6
                flag_6 = other_6 * 64
                pl.system.cacheinvalid(progress, [1], [flag_6])
                initial_6 = pl.tensor.read(progress, [flag_6])
                for (ready_6, polls_6) in pl.while_(init_values=(initial_6, 0)):
                    pl.cond(ready_6 < target_6 and polls_6 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_6])
                    (after_6, spin_count_6) = pl.yield_(pl.tensor.read(progress, [flag_6]), polls_6 + 1)
                if after_6 < target_6:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 6)
                    pl.tensor.write(progress, [own_flag + 10], other_6)
                    pl.tensor.write(progress, [own_flag + 11], target_6)
                    pl.tensor.write(progress, [own_flag + 12], after_6)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_6)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_6 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_6)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_7 = 3024 + core_id * 6
    mapped_block_7 = pl.cast(pl.tensor.read(records, [record_7]), pl.INDEX)
    block_7 = pl.max(mapped_block_7, 0)
    if mapped_block_7 >= 0 and failed == 0:
        subslot_7 = pl.cast(pl.tensor.read(records, [record_7 + 1]), pl.INDEX)
        done_7 = pl.tensor.read(records, [record_7 + 2])
        wait_base_7 = pl.cast(pl.tensor.read(records, [record_7 + 3]), pl.INDEX)
        wait_count_7 = pl.cast(pl.tensor.read(records, [record_7 + 4]), pl.INDEX)
        publish_7 = pl.tensor.read(records, [record_7 + 5])
        for dependency_7 in pl.range(wait_count_7):
            if failed == 0:
                wait_index_7 = (wait_base_7 + dependency_7) * 2
                other_7 = pl.cast(pl.tensor.read(waits, [wait_index_7]), pl.INDEX)
                required_7 = pl.tensor.read(waits, [wait_index_7 + 1])
                target_7 = generation_base + required_7
                flag_7 = other_7 * 64
                pl.system.cacheinvalid(progress, [1], [flag_7])
                initial_7 = pl.tensor.read(progress, [flag_7])
                for (ready_7, polls_7) in pl.while_(init_values=(initial_7, 0)):
                    pl.cond(ready_7 < target_7 and polls_7 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_7])
                    (after_7, spin_count_7) = pl.yield_(pl.tensor.read(progress, [flag_7]), polls_7 + 1)
                if after_7 < target_7:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 7)
                    pl.tensor.write(progress, [own_flag + 10], other_7)
                    pl.tensor.write(progress, [own_flag + 11], target_7)
                    pl.tensor.write(progress, [own_flag + 12], after_7)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_7)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_7 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_7)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_8 = 3456 + core_id * 6
    mapped_block_8 = pl.cast(pl.tensor.read(records, [record_8]), pl.INDEX)
    block_8 = pl.max(mapped_block_8, 0)
    if mapped_block_8 >= 0 and failed == 0:
        subslot_8 = pl.cast(pl.tensor.read(records, [record_8 + 1]), pl.INDEX)
        done_8 = pl.tensor.read(records, [record_8 + 2])
        wait_base_8 = pl.cast(pl.tensor.read(records, [record_8 + 3]), pl.INDEX)
        wait_count_8 = pl.cast(pl.tensor.read(records, [record_8 + 4]), pl.INDEX)
        publish_8 = pl.tensor.read(records, [record_8 + 5])
        for dependency_8 in pl.range(wait_count_8):
            if failed == 0:
                wait_index_8 = (wait_base_8 + dependency_8) * 2
                other_8 = pl.cast(pl.tensor.read(waits, [wait_index_8]), pl.INDEX)
                required_8 = pl.tensor.read(waits, [wait_index_8 + 1])
                target_8 = generation_base + required_8
                flag_8 = other_8 * 64
                pl.system.cacheinvalid(progress, [1], [flag_8])
                initial_8 = pl.tensor.read(progress, [flag_8])
                for (ready_8, polls_8) in pl.while_(init_values=(initial_8, 0)):
                    pl.cond(ready_8 < target_8 and polls_8 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_8])
                    (after_8, spin_count_8) = pl.yield_(pl.tensor.read(progress, [flag_8]), polls_8 + 1)
                if after_8 < target_8:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 8)
                    pl.tensor.write(progress, [own_flag + 10], other_8)
                    pl.tensor.write(progress, [own_flag + 11], target_8)
                    pl.tensor.write(progress, [own_flag + 12], after_8)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_8)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_8 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_8)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_9 = 3888 + core_id * 6
    mapped_block_9 = pl.cast(pl.tensor.read(records, [record_9]), pl.INDEX)
    block_9 = pl.max(mapped_block_9, 0)
    if mapped_block_9 >= 0 and failed == 0:
        subslot_9 = pl.cast(pl.tensor.read(records, [record_9 + 1]), pl.INDEX)
        done_9 = pl.tensor.read(records, [record_9 + 2])
        wait_base_9 = pl.cast(pl.tensor.read(records, [record_9 + 3]), pl.INDEX)
        wait_count_9 = pl.cast(pl.tensor.read(records, [record_9 + 4]), pl.INDEX)
        publish_9 = pl.tensor.read(records, [record_9 + 5])
        for dependency_9 in pl.range(wait_count_9):
            if failed == 0:
                wait_index_9 = (wait_base_9 + dependency_9) * 2
                other_9 = pl.cast(pl.tensor.read(waits, [wait_index_9]), pl.INDEX)
                required_9 = pl.tensor.read(waits, [wait_index_9 + 1])
                target_9 = generation_base + required_9
                flag_9 = other_9 * 64
                pl.system.cacheinvalid(progress, [1], [flag_9])
                initial_9 = pl.tensor.read(progress, [flag_9])
                for (ready_9, polls_9) in pl.while_(init_values=(initial_9, 0)):
                    pl.cond(ready_9 < target_9 and polls_9 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_9])
                    (after_9, spin_count_9) = pl.yield_(pl.tensor.read(progress, [flag_9]), polls_9 + 1)
                if after_9 < target_9:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 9)
                    pl.tensor.write(progress, [own_flag + 10], other_9)
                    pl.tensor.write(progress, [own_flag + 11], target_9)
                    pl.tensor.write(progress, [own_flag + 12], after_9)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_9)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_9 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_9)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_10 = 4320 + core_id * 6
    mapped_block_10 = pl.cast(pl.tensor.read(records, [record_10]), pl.INDEX)
    block_10 = pl.max(mapped_block_10, 0)
    if mapped_block_10 >= 0 and failed == 0:
        subslot_10 = pl.cast(pl.tensor.read(records, [record_10 + 1]), pl.INDEX)
        done_10 = pl.tensor.read(records, [record_10 + 2])
        wait_base_10 = pl.cast(pl.tensor.read(records, [record_10 + 3]), pl.INDEX)
        wait_count_10 = pl.cast(pl.tensor.read(records, [record_10 + 4]), pl.INDEX)
        publish_10 = pl.tensor.read(records, [record_10 + 5])
        for dependency_10 in pl.range(wait_count_10):
            if failed == 0:
                wait_index_10 = (wait_base_10 + dependency_10) * 2
                other_10 = pl.cast(pl.tensor.read(waits, [wait_index_10]), pl.INDEX)
                required_10 = pl.tensor.read(waits, [wait_index_10 + 1])
                target_10 = generation_base + required_10
                flag_10 = other_10 * 64
                pl.system.cacheinvalid(progress, [1], [flag_10])
                initial_10 = pl.tensor.read(progress, [flag_10])
                for (ready_10, polls_10) in pl.while_(init_values=(initial_10, 0)):
                    pl.cond(ready_10 < target_10 and polls_10 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_10])
                    (after_10, spin_count_10) = pl.yield_(pl.tensor.read(progress, [flag_10]), polls_10 + 1)
                if after_10 < target_10:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 10)
                    pl.tensor.write(progress, [own_flag + 10], other_10)
                    pl.tensor.write(progress, [own_flag + 11], target_10)
                    pl.tensor.write(progress, [own_flag + 12], after_10)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_10)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_10 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_10)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_11 = 4752 + core_id * 6
    mapped_block_11 = pl.cast(pl.tensor.read(records, [record_11]), pl.INDEX)
    block_11 = pl.max(mapped_block_11, 0)
    if mapped_block_11 >= 0 and failed == 0:
        subslot_11 = pl.cast(pl.tensor.read(records, [record_11 + 1]), pl.INDEX)
        done_11 = pl.tensor.read(records, [record_11 + 2])
        wait_base_11 = pl.cast(pl.tensor.read(records, [record_11 + 3]), pl.INDEX)
        wait_count_11 = pl.cast(pl.tensor.read(records, [record_11 + 4]), pl.INDEX)
        publish_11 = pl.tensor.read(records, [record_11 + 5])
        for dependency_11 in pl.range(wait_count_11):
            if failed == 0:
                wait_index_11 = (wait_base_11 + dependency_11) * 2
                other_11 = pl.cast(pl.tensor.read(waits, [wait_index_11]), pl.INDEX)
                required_11 = pl.tensor.read(waits, [wait_index_11 + 1])
                target_11 = generation_base + required_11
                flag_11 = other_11 * 64
                pl.system.cacheinvalid(progress, [1], [flag_11])
                initial_11 = pl.tensor.read(progress, [flag_11])
                for (ready_11, polls_11) in pl.while_(init_values=(initial_11, 0)):
                    pl.cond(ready_11 < target_11 and polls_11 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_11])
                    (after_11, spin_count_11) = pl.yield_(pl.tensor.read(progress, [flag_11]), polls_11 + 1)
                if after_11 < target_11:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 11)
                    pl.tensor.write(progress, [own_flag + 10], other_11)
                    pl.tensor.write(progress, [own_flag + 11], target_11)
                    pl.tensor.write(progress, [own_flag + 12], after_11)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_11)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_11 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_11)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_12 = 5184 + core_id * 6
    mapped_block_12 = pl.cast(pl.tensor.read(records, [record_12]), pl.INDEX)
    block_12 = pl.max(mapped_block_12, 0)
    if mapped_block_12 >= 0 and failed == 0:
        subslot_12 = pl.cast(pl.tensor.read(records, [record_12 + 1]), pl.INDEX)
        done_12 = pl.tensor.read(records, [record_12 + 2])
        wait_base_12 = pl.cast(pl.tensor.read(records, [record_12 + 3]), pl.INDEX)
        wait_count_12 = pl.cast(pl.tensor.read(records, [record_12 + 4]), pl.INDEX)
        publish_12 = pl.tensor.read(records, [record_12 + 5])
        for dependency_12 in pl.range(wait_count_12):
            if failed == 0:
                wait_index_12 = (wait_base_12 + dependency_12) * 2
                other_12 = pl.cast(pl.tensor.read(waits, [wait_index_12]), pl.INDEX)
                required_12 = pl.tensor.read(waits, [wait_index_12 + 1])
                target_12 = generation_base + required_12
                flag_12 = other_12 * 64
                pl.system.cacheinvalid(progress, [1], [flag_12])
                initial_12 = pl.tensor.read(progress, [flag_12])
                for (ready_12, polls_12) in pl.while_(init_values=(initial_12, 0)):
                    pl.cond(ready_12 < target_12 and polls_12 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_12])
                    (after_12, spin_count_12) = pl.yield_(pl.tensor.read(progress, [flag_12]), polls_12 + 1)
                if after_12 < target_12:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 12)
                    pl.tensor.write(progress, [own_flag + 10], other_12)
                    pl.tensor.write(progress, [own_flag + 11], target_12)
                    pl.tensor.write(progress, [own_flag + 12], after_12)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_12)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_12 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_12)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_13 = 5616 + core_id * 6
    mapped_block_13 = pl.cast(pl.tensor.read(records, [record_13]), pl.INDEX)
    block_13 = pl.max(mapped_block_13, 0)
    if mapped_block_13 >= 0 and failed == 0:
        subslot_13 = pl.cast(pl.tensor.read(records, [record_13 + 1]), pl.INDEX)
        done_13 = pl.tensor.read(records, [record_13 + 2])
        wait_base_13 = pl.cast(pl.tensor.read(records, [record_13 + 3]), pl.INDEX)
        wait_count_13 = pl.cast(pl.tensor.read(records, [record_13 + 4]), pl.INDEX)
        publish_13 = pl.tensor.read(records, [record_13 + 5])
        for dependency_13 in pl.range(wait_count_13):
            if failed == 0:
                wait_index_13 = (wait_base_13 + dependency_13) * 2
                other_13 = pl.cast(pl.tensor.read(waits, [wait_index_13]), pl.INDEX)
                required_13 = pl.tensor.read(waits, [wait_index_13 + 1])
                target_13 = generation_base + required_13
                flag_13 = other_13 * 64
                pl.system.cacheinvalid(progress, [1], [flag_13])
                initial_13 = pl.tensor.read(progress, [flag_13])
                for (ready_13, polls_13) in pl.while_(init_values=(initial_13, 0)):
                    pl.cond(ready_13 < target_13 and polls_13 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_13])
                    (after_13, spin_count_13) = pl.yield_(pl.tensor.read(progress, [flag_13]), polls_13 + 1)
                if after_13 < target_13:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 13)
                    pl.tensor.write(progress, [own_flag + 10], other_13)
                    pl.tensor.write(progress, [own_flag + 11], target_13)
                    pl.tensor.write(progress, [own_flag + 12], after_13)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_13)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_13 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_13)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_14 = 6048 + core_id * 6
    mapped_block_14 = pl.cast(pl.tensor.read(records, [record_14]), pl.INDEX)
    block_14 = pl.max(mapped_block_14, 0)
    if mapped_block_14 >= 0 and failed == 0:
        subslot_14 = pl.cast(pl.tensor.read(records, [record_14 + 1]), pl.INDEX)
        done_14 = pl.tensor.read(records, [record_14 + 2])
        wait_base_14 = pl.cast(pl.tensor.read(records, [record_14 + 3]), pl.INDEX)
        wait_count_14 = pl.cast(pl.tensor.read(records, [record_14 + 4]), pl.INDEX)
        publish_14 = pl.tensor.read(records, [record_14 + 5])
        for dependency_14 in pl.range(wait_count_14):
            if failed == 0:
                wait_index_14 = (wait_base_14 + dependency_14) * 2
                other_14 = pl.cast(pl.tensor.read(waits, [wait_index_14]), pl.INDEX)
                required_14 = pl.tensor.read(waits, [wait_index_14 + 1])
                target_14 = generation_base + required_14
                flag_14 = other_14 * 64
                pl.system.cacheinvalid(progress, [1], [flag_14])
                initial_14 = pl.tensor.read(progress, [flag_14])
                for (ready_14, polls_14) in pl.while_(init_values=(initial_14, 0)):
                    pl.cond(ready_14 < target_14 and polls_14 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_14])
                    (after_14, spin_count_14) = pl.yield_(pl.tensor.read(progress, [flag_14]), polls_14 + 1)
                if after_14 < target_14:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 14)
                    pl.tensor.write(progress, [own_flag + 10], other_14)
                    pl.tensor.write(progress, [own_flag + 11], target_14)
                    pl.tensor.write(progress, [own_flag + 12], after_14)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_14)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_14 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_14)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_15 = 6480 + core_id * 6
    mapped_block_15 = pl.cast(pl.tensor.read(records, [record_15]), pl.INDEX)
    block_15 = pl.max(mapped_block_15, 0)
    if mapped_block_15 >= 0 and failed == 0:
        subslot_15 = pl.cast(pl.tensor.read(records, [record_15 + 1]), pl.INDEX)
        done_15 = pl.tensor.read(records, [record_15 + 2])
        wait_base_15 = pl.cast(pl.tensor.read(records, [record_15 + 3]), pl.INDEX)
        wait_count_15 = pl.cast(pl.tensor.read(records, [record_15 + 4]), pl.INDEX)
        publish_15 = pl.tensor.read(records, [record_15 + 5])
        for dependency_15 in pl.range(wait_count_15):
            if failed == 0:
                wait_index_15 = (wait_base_15 + dependency_15) * 2
                other_15 = pl.cast(pl.tensor.read(waits, [wait_index_15]), pl.INDEX)
                required_15 = pl.tensor.read(waits, [wait_index_15 + 1])
                target_15 = generation_base + required_15
                flag_15 = other_15 * 64
                pl.system.cacheinvalid(progress, [1], [flag_15])
                initial_15 = pl.tensor.read(progress, [flag_15])
                for (ready_15, polls_15) in pl.while_(init_values=(initial_15, 0)):
                    pl.cond(ready_15 < target_15 and polls_15 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_15])
                    (after_15, spin_count_15) = pl.yield_(pl.tensor.read(progress, [flag_15]), polls_15 + 1)
                if after_15 < target_15:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 15)
                    pl.tensor.write(progress, [own_flag + 10], other_15)
                    pl.tensor.write(progress, [own_flag + 11], target_15)
                    pl.tensor.write(progress, [own_flag + 12], after_15)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_15)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_15 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_15)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_16 = 6912 + core_id * 6
    mapped_block_16 = pl.cast(pl.tensor.read(records, [record_16]), pl.INDEX)
    block_16 = pl.max(mapped_block_16, 0)
    if mapped_block_16 >= 0 and failed == 0:
        subslot_16 = pl.cast(pl.tensor.read(records, [record_16 + 1]), pl.INDEX)
        done_16 = pl.tensor.read(records, [record_16 + 2])
        wait_base_16 = pl.cast(pl.tensor.read(records, [record_16 + 3]), pl.INDEX)
        wait_count_16 = pl.cast(pl.tensor.read(records, [record_16 + 4]), pl.INDEX)
        publish_16 = pl.tensor.read(records, [record_16 + 5])
        for dependency_16 in pl.range(wait_count_16):
            if failed == 0:
                wait_index_16 = (wait_base_16 + dependency_16) * 2
                other_16 = pl.cast(pl.tensor.read(waits, [wait_index_16]), pl.INDEX)
                required_16 = pl.tensor.read(waits, [wait_index_16 + 1])
                target_16 = generation_base + required_16
                flag_16 = other_16 * 64
                pl.system.cacheinvalid(progress, [1], [flag_16])
                initial_16 = pl.tensor.read(progress, [flag_16])
                for (ready_16, polls_16) in pl.while_(init_values=(initial_16, 0)):
                    pl.cond(ready_16 < target_16 and polls_16 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_16])
                    (after_16, spin_count_16) = pl.yield_(pl.tensor.read(progress, [flag_16]), polls_16 + 1)
                if after_16 < target_16:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 16)
                    pl.tensor.write(progress, [own_flag + 10], other_16)
                    pl.tensor.write(progress, [own_flag + 11], target_16)
                    pl.tensor.write(progress, [own_flag + 12], after_16)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_16)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_16 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_16)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_17 = 7344 + core_id * 6
    mapped_block_17 = pl.cast(pl.tensor.read(records, [record_17]), pl.INDEX)
    block_17 = pl.max(mapped_block_17, 0)
    if mapped_block_17 >= 0 and failed == 0:
        subslot_17 = pl.cast(pl.tensor.read(records, [record_17 + 1]), pl.INDEX)
        done_17 = pl.tensor.read(records, [record_17 + 2])
        wait_base_17 = pl.cast(pl.tensor.read(records, [record_17 + 3]), pl.INDEX)
        wait_count_17 = pl.cast(pl.tensor.read(records, [record_17 + 4]), pl.INDEX)
        publish_17 = pl.tensor.read(records, [record_17 + 5])
        for dependency_17 in pl.range(wait_count_17):
            if failed == 0:
                wait_index_17 = (wait_base_17 + dependency_17) * 2
                other_17 = pl.cast(pl.tensor.read(waits, [wait_index_17]), pl.INDEX)
                required_17 = pl.tensor.read(waits, [wait_index_17 + 1])
                target_17 = generation_base + required_17
                flag_17 = other_17 * 64
                pl.system.cacheinvalid(progress, [1], [flag_17])
                initial_17 = pl.tensor.read(progress, [flag_17])
                for (ready_17, polls_17) in pl.while_(init_values=(initial_17, 0)):
                    pl.cond(ready_17 < target_17 and polls_17 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_17])
                    (after_17, spin_count_17) = pl.yield_(pl.tensor.read(progress, [flag_17]), polls_17 + 1)
                if after_17 < target_17:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 17)
                    pl.tensor.write(progress, [own_flag + 10], other_17)
                    pl.tensor.write(progress, [own_flag + 11], target_17)
                    pl.tensor.write(progress, [own_flag + 12], after_17)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_17)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_17 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_17)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_18 = 7776 + core_id * 6
    mapped_block_18 = pl.cast(pl.tensor.read(records, [record_18]), pl.INDEX)
    block_18 = pl.max(mapped_block_18, 0)
    if mapped_block_18 >= 0 and failed == 0:
        subslot_18 = pl.cast(pl.tensor.read(records, [record_18 + 1]), pl.INDEX)
        done_18 = pl.tensor.read(records, [record_18 + 2])
        wait_base_18 = pl.cast(pl.tensor.read(records, [record_18 + 3]), pl.INDEX)
        wait_count_18 = pl.cast(pl.tensor.read(records, [record_18 + 4]), pl.INDEX)
        publish_18 = pl.tensor.read(records, [record_18 + 5])
        for dependency_18 in pl.range(wait_count_18):
            if failed == 0:
                wait_index_18 = (wait_base_18 + dependency_18) * 2
                other_18 = pl.cast(pl.tensor.read(waits, [wait_index_18]), pl.INDEX)
                required_18 = pl.tensor.read(waits, [wait_index_18 + 1])
                target_18 = generation_base + required_18
                flag_18 = other_18 * 64
                pl.system.cacheinvalid(progress, [1], [flag_18])
                initial_18 = pl.tensor.read(progress, [flag_18])
                for (ready_18, polls_18) in pl.while_(init_values=(initial_18, 0)):
                    pl.cond(ready_18 < target_18 and polls_18 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_18])
                    (after_18, spin_count_18) = pl.yield_(pl.tensor.read(progress, [flag_18]), polls_18 + 1)
                if after_18 < target_18:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 18)
                    pl.tensor.write(progress, [own_flag + 10], other_18)
                    pl.tensor.write(progress, [own_flag + 11], target_18)
                    pl.tensor.write(progress, [own_flag + 12], after_18)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_18)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_18 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_18)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_19 = 8208 + core_id * 6
    mapped_block_19 = pl.cast(pl.tensor.read(records, [record_19]), pl.INDEX)
    block_19 = pl.max(mapped_block_19, 0)
    if mapped_block_19 >= 0 and failed == 0:
        subslot_19 = pl.cast(pl.tensor.read(records, [record_19 + 1]), pl.INDEX)
        done_19 = pl.tensor.read(records, [record_19 + 2])
        wait_base_19 = pl.cast(pl.tensor.read(records, [record_19 + 3]), pl.INDEX)
        wait_count_19 = pl.cast(pl.tensor.read(records, [record_19 + 4]), pl.INDEX)
        publish_19 = pl.tensor.read(records, [record_19 + 5])
        for dependency_19 in pl.range(wait_count_19):
            if failed == 0:
                wait_index_19 = (wait_base_19 + dependency_19) * 2
                other_19 = pl.cast(pl.tensor.read(waits, [wait_index_19]), pl.INDEX)
                required_19 = pl.tensor.read(waits, [wait_index_19 + 1])
                target_19 = generation_base + required_19
                flag_19 = other_19 * 64
                pl.system.cacheinvalid(progress, [1], [flag_19])
                initial_19 = pl.tensor.read(progress, [flag_19])
                for (ready_19, polls_19) in pl.while_(init_values=(initial_19, 0)):
                    pl.cond(ready_19 < target_19 and polls_19 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_19])
                    (after_19, spin_count_19) = pl.yield_(pl.tensor.read(progress, [flag_19]), polls_19 + 1)
                if after_19 < target_19:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 19)
                    pl.tensor.write(progress, [own_flag + 10], other_19)
                    pl.tensor.write(progress, [own_flag + 11], target_19)
                    pl.tensor.write(progress, [own_flag + 12], after_19)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_19)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_19 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_19)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_20 = 8640 + core_id * 6
    mapped_block_20 = pl.cast(pl.tensor.read(records, [record_20]), pl.INDEX)
    block_20 = pl.max(mapped_block_20, 0)
    if mapped_block_20 >= 0 and failed == 0:
        subslot_20 = pl.cast(pl.tensor.read(records, [record_20 + 1]), pl.INDEX)
        done_20 = pl.tensor.read(records, [record_20 + 2])
        wait_base_20 = pl.cast(pl.tensor.read(records, [record_20 + 3]), pl.INDEX)
        wait_count_20 = pl.cast(pl.tensor.read(records, [record_20 + 4]), pl.INDEX)
        publish_20 = pl.tensor.read(records, [record_20 + 5])
        for dependency_20 in pl.range(wait_count_20):
            if failed == 0:
                wait_index_20 = (wait_base_20 + dependency_20) * 2
                other_20 = pl.cast(pl.tensor.read(waits, [wait_index_20]), pl.INDEX)
                required_20 = pl.tensor.read(waits, [wait_index_20 + 1])
                target_20 = generation_base + required_20
                flag_20 = other_20 * 64
                pl.system.cacheinvalid(progress, [1], [flag_20])
                initial_20 = pl.tensor.read(progress, [flag_20])
                for (ready_20, polls_20) in pl.while_(init_values=(initial_20, 0)):
                    pl.cond(ready_20 < target_20 and polls_20 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_20])
                    (after_20, spin_count_20) = pl.yield_(pl.tensor.read(progress, [flag_20]), polls_20 + 1)
                if after_20 < target_20:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 20)
                    pl.tensor.write(progress, [own_flag + 10], other_20)
                    pl.tensor.write(progress, [own_flag + 11], target_20)
                    pl.tensor.write(progress, [own_flag + 12], after_20)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_20)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_20 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_20)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_21 = 9072 + core_id * 6
    mapped_block_21 = pl.cast(pl.tensor.read(records, [record_21]), pl.INDEX)
    block_21 = pl.max(mapped_block_21, 0)
    if mapped_block_21 >= 0 and failed == 0:
        subslot_21 = pl.cast(pl.tensor.read(records, [record_21 + 1]), pl.INDEX)
        done_21 = pl.tensor.read(records, [record_21 + 2])
        wait_base_21 = pl.cast(pl.tensor.read(records, [record_21 + 3]), pl.INDEX)
        wait_count_21 = pl.cast(pl.tensor.read(records, [record_21 + 4]), pl.INDEX)
        publish_21 = pl.tensor.read(records, [record_21 + 5])
        for dependency_21 in pl.range(wait_count_21):
            if failed == 0:
                wait_index_21 = (wait_base_21 + dependency_21) * 2
                other_21 = pl.cast(pl.tensor.read(waits, [wait_index_21]), pl.INDEX)
                required_21 = pl.tensor.read(waits, [wait_index_21 + 1])
                target_21 = generation_base + required_21
                flag_21 = other_21 * 64
                pl.system.cacheinvalid(progress, [1], [flag_21])
                initial_21 = pl.tensor.read(progress, [flag_21])
                for (ready_21, polls_21) in pl.while_(init_values=(initial_21, 0)):
                    pl.cond(ready_21 < target_21 and polls_21 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_21])
                    (after_21, spin_count_21) = pl.yield_(pl.tensor.read(progress, [flag_21]), polls_21 + 1)
                if after_21 < target_21:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 21)
                    pl.tensor.write(progress, [own_flag + 10], other_21)
                    pl.tensor.write(progress, [own_flag + 11], target_21)
                    pl.tensor.write(progress, [own_flag + 12], after_21)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_21)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_21 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_21)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_22 = 9504 + core_id * 6
    mapped_block_22 = pl.cast(pl.tensor.read(records, [record_22]), pl.INDEX)
    block_22 = pl.max(mapped_block_22, 0)
    if mapped_block_22 >= 0 and failed == 0:
        subslot_22 = pl.cast(pl.tensor.read(records, [record_22 + 1]), pl.INDEX)
        done_22 = pl.tensor.read(records, [record_22 + 2])
        wait_base_22 = pl.cast(pl.tensor.read(records, [record_22 + 3]), pl.INDEX)
        wait_count_22 = pl.cast(pl.tensor.read(records, [record_22 + 4]), pl.INDEX)
        publish_22 = pl.tensor.read(records, [record_22 + 5])
        for dependency_22 in pl.range(wait_count_22):
            if failed == 0:
                wait_index_22 = (wait_base_22 + dependency_22) * 2
                other_22 = pl.cast(pl.tensor.read(waits, [wait_index_22]), pl.INDEX)
                required_22 = pl.tensor.read(waits, [wait_index_22 + 1])
                target_22 = generation_base + required_22
                flag_22 = other_22 * 64
                pl.system.cacheinvalid(progress, [1], [flag_22])
                initial_22 = pl.tensor.read(progress, [flag_22])
                for (ready_22, polls_22) in pl.while_(init_values=(initial_22, 0)):
                    pl.cond(ready_22 < target_22 and polls_22 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_22])
                    (after_22, spin_count_22) = pl.yield_(pl.tensor.read(progress, [flag_22]), polls_22 + 1)
                if after_22 < target_22:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 22)
                    pl.tensor.write(progress, [own_flag + 10], other_22)
                    pl.tensor.write(progress, [own_flag + 11], target_22)
                    pl.tensor.write(progress, [own_flag + 12], after_22)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_22)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_22 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_22)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_23 = 9936 + core_id * 6
    mapped_block_23 = pl.cast(pl.tensor.read(records, [record_23]), pl.INDEX)
    block_23 = pl.max(mapped_block_23, 0)
    if mapped_block_23 >= 0 and failed == 0:
        subslot_23 = pl.cast(pl.tensor.read(records, [record_23 + 1]), pl.INDEX)
        done_23 = pl.tensor.read(records, [record_23 + 2])
        wait_base_23 = pl.cast(pl.tensor.read(records, [record_23 + 3]), pl.INDEX)
        wait_count_23 = pl.cast(pl.tensor.read(records, [record_23 + 4]), pl.INDEX)
        publish_23 = pl.tensor.read(records, [record_23 + 5])
        for dependency_23 in pl.range(wait_count_23):
            if failed == 0:
                wait_index_23 = (wait_base_23 + dependency_23) * 2
                other_23 = pl.cast(pl.tensor.read(waits, [wait_index_23]), pl.INDEX)
                required_23 = pl.tensor.read(waits, [wait_index_23 + 1])
                target_23 = generation_base + required_23
                flag_23 = other_23 * 64
                pl.system.cacheinvalid(progress, [1], [flag_23])
                initial_23 = pl.tensor.read(progress, [flag_23])
                for (ready_23, polls_23) in pl.while_(init_values=(initial_23, 0)):
                    pl.cond(ready_23 < target_23 and polls_23 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_23])
                    (after_23, spin_count_23) = pl.yield_(pl.tensor.read(progress, [flag_23]), polls_23 + 1)
                if after_23 < target_23:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 23)
                    pl.tensor.write(progress, [own_flag + 10], other_23)
                    pl.tensor.write(progress, [own_flag + 11], target_23)
                    pl.tensor.write(progress, [own_flag + 12], after_23)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_23)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_23 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_23)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_24 = 10368 + core_id * 6
    mapped_block_24 = pl.cast(pl.tensor.read(records, [record_24]), pl.INDEX)
    block_24 = pl.max(mapped_block_24, 0)
    if mapped_block_24 >= 0 and failed == 0:
        subslot_24 = pl.cast(pl.tensor.read(records, [record_24 + 1]), pl.INDEX)
        done_24 = pl.tensor.read(records, [record_24 + 2])
        wait_base_24 = pl.cast(pl.tensor.read(records, [record_24 + 3]), pl.INDEX)
        wait_count_24 = pl.cast(pl.tensor.read(records, [record_24 + 4]), pl.INDEX)
        publish_24 = pl.tensor.read(records, [record_24 + 5])
        for dependency_24 in pl.range(wait_count_24):
            if failed == 0:
                wait_index_24 = (wait_base_24 + dependency_24) * 2
                other_24 = pl.cast(pl.tensor.read(waits, [wait_index_24]), pl.INDEX)
                required_24 = pl.tensor.read(waits, [wait_index_24 + 1])
                target_24 = generation_base + required_24
                flag_24 = other_24 * 64
                pl.system.cacheinvalid(progress, [1], [flag_24])
                initial_24 = pl.tensor.read(progress, [flag_24])
                for (ready_24, polls_24) in pl.while_(init_values=(initial_24, 0)):
                    pl.cond(ready_24 < target_24 and polls_24 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_24])
                    (after_24, spin_count_24) = pl.yield_(pl.tensor.read(progress, [flag_24]), polls_24 + 1)
                if after_24 < target_24:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 24)
                    pl.tensor.write(progress, [own_flag + 10], other_24)
                    pl.tensor.write(progress, [own_flag + 11], target_24)
                    pl.tensor.write(progress, [own_flag + 12], after_24)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_24)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_24 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_24)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_25 = 10800 + core_id * 6
    mapped_block_25 = pl.cast(pl.tensor.read(records, [record_25]), pl.INDEX)
    block_25 = pl.max(mapped_block_25, 0)
    if mapped_block_25 >= 0 and failed == 0:
        subslot_25 = pl.cast(pl.tensor.read(records, [record_25 + 1]), pl.INDEX)
        done_25 = pl.tensor.read(records, [record_25 + 2])
        wait_base_25 = pl.cast(pl.tensor.read(records, [record_25 + 3]), pl.INDEX)
        wait_count_25 = pl.cast(pl.tensor.read(records, [record_25 + 4]), pl.INDEX)
        publish_25 = pl.tensor.read(records, [record_25 + 5])
        for dependency_25 in pl.range(wait_count_25):
            if failed == 0:
                wait_index_25 = (wait_base_25 + dependency_25) * 2
                other_25 = pl.cast(pl.tensor.read(waits, [wait_index_25]), pl.INDEX)
                required_25 = pl.tensor.read(waits, [wait_index_25 + 1])
                target_25 = generation_base + required_25
                flag_25 = other_25 * 64
                pl.system.cacheinvalid(progress, [1], [flag_25])
                initial_25 = pl.tensor.read(progress, [flag_25])
                for (ready_25, polls_25) in pl.while_(init_values=(initial_25, 0)):
                    pl.cond(ready_25 < target_25 and polls_25 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_25])
                    (after_25, spin_count_25) = pl.yield_(pl.tensor.read(progress, [flag_25]), polls_25 + 1)
                if after_25 < target_25:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 25)
                    pl.tensor.write(progress, [own_flag + 10], other_25)
                    pl.tensor.write(progress, [own_flag + 11], target_25)
                    pl.tensor.write(progress, [own_flag + 12], after_25)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_25)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_25 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_25)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_26 = 11232 + core_id * 6
    mapped_block_26 = pl.cast(pl.tensor.read(records, [record_26]), pl.INDEX)
    block_26 = pl.max(mapped_block_26, 0)
    if mapped_block_26 >= 0 and failed == 0:
        subslot_26 = pl.cast(pl.tensor.read(records, [record_26 + 1]), pl.INDEX)
        done_26 = pl.tensor.read(records, [record_26 + 2])
        wait_base_26 = pl.cast(pl.tensor.read(records, [record_26 + 3]), pl.INDEX)
        wait_count_26 = pl.cast(pl.tensor.read(records, [record_26 + 4]), pl.INDEX)
        publish_26 = pl.tensor.read(records, [record_26 + 5])
        for dependency_26 in pl.range(wait_count_26):
            if failed == 0:
                wait_index_26 = (wait_base_26 + dependency_26) * 2
                other_26 = pl.cast(pl.tensor.read(waits, [wait_index_26]), pl.INDEX)
                required_26 = pl.tensor.read(waits, [wait_index_26 + 1])
                target_26 = generation_base + required_26
                flag_26 = other_26 * 64
                pl.system.cacheinvalid(progress, [1], [flag_26])
                initial_26 = pl.tensor.read(progress, [flag_26])
                for (ready_26, polls_26) in pl.while_(init_values=(initial_26, 0)):
                    pl.cond(ready_26 < target_26 and polls_26 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_26])
                    (after_26, spin_count_26) = pl.yield_(pl.tensor.read(progress, [flag_26]), polls_26 + 1)
                if after_26 < target_26:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 26)
                    pl.tensor.write(progress, [own_flag + 10], other_26)
                    pl.tensor.write(progress, [own_flag + 11], target_26)
                    pl.tensor.write(progress, [own_flag + 12], after_26)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_26)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_26 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_26)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_27 = 11664 + core_id * 6
    mapped_block_27 = pl.cast(pl.tensor.read(records, [record_27]), pl.INDEX)
    block_27 = pl.max(mapped_block_27, 0)
    if mapped_block_27 >= 0 and failed == 0:
        subslot_27 = pl.cast(pl.tensor.read(records, [record_27 + 1]), pl.INDEX)
        done_27 = pl.tensor.read(records, [record_27 + 2])
        wait_base_27 = pl.cast(pl.tensor.read(records, [record_27 + 3]), pl.INDEX)
        wait_count_27 = pl.cast(pl.tensor.read(records, [record_27 + 4]), pl.INDEX)
        publish_27 = pl.tensor.read(records, [record_27 + 5])
        for dependency_27 in pl.range(wait_count_27):
            if failed == 0:
                wait_index_27 = (wait_base_27 + dependency_27) * 2
                other_27 = pl.cast(pl.tensor.read(waits, [wait_index_27]), pl.INDEX)
                required_27 = pl.tensor.read(waits, [wait_index_27 + 1])
                target_27 = generation_base + required_27
                flag_27 = other_27 * 64
                pl.system.cacheinvalid(progress, [1], [flag_27])
                initial_27 = pl.tensor.read(progress, [flag_27])
                for (ready_27, polls_27) in pl.while_(init_values=(initial_27, 0)):
                    pl.cond(ready_27 < target_27 and polls_27 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_27])
                    (after_27, spin_count_27) = pl.yield_(pl.tensor.read(progress, [flag_27]), polls_27 + 1)
                if after_27 < target_27:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 27)
                    pl.tensor.write(progress, [own_flag + 10], other_27)
                    pl.tensor.write(progress, [own_flag + 11], target_27)
                    pl.tensor.write(progress, [own_flag + 12], after_27)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_27)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_27 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_27)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_28 = 12096 + core_id * 6
    mapped_block_28 = pl.cast(pl.tensor.read(records, [record_28]), pl.INDEX)
    block_28 = pl.max(mapped_block_28, 0)
    if mapped_block_28 >= 0 and failed == 0:
        subslot_28 = pl.cast(pl.tensor.read(records, [record_28 + 1]), pl.INDEX)
        done_28 = pl.tensor.read(records, [record_28 + 2])
        wait_base_28 = pl.cast(pl.tensor.read(records, [record_28 + 3]), pl.INDEX)
        wait_count_28 = pl.cast(pl.tensor.read(records, [record_28 + 4]), pl.INDEX)
        publish_28 = pl.tensor.read(records, [record_28 + 5])
        for dependency_28 in pl.range(wait_count_28):
            if failed == 0:
                wait_index_28 = (wait_base_28 + dependency_28) * 2
                other_28 = pl.cast(pl.tensor.read(waits, [wait_index_28]), pl.INDEX)
                required_28 = pl.tensor.read(waits, [wait_index_28 + 1])
                target_28 = generation_base + required_28
                flag_28 = other_28 * 64
                pl.system.cacheinvalid(progress, [1], [flag_28])
                initial_28 = pl.tensor.read(progress, [flag_28])
                for (ready_28, polls_28) in pl.while_(init_values=(initial_28, 0)):
                    pl.cond(ready_28 < target_28 and polls_28 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_28])
                    (after_28, spin_count_28) = pl.yield_(pl.tensor.read(progress, [flag_28]), polls_28 + 1)
                if after_28 < target_28:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 28)
                    pl.tensor.write(progress, [own_flag + 10], other_28)
                    pl.tensor.write(progress, [own_flag + 11], target_28)
                    pl.tensor.write(progress, [own_flag + 12], after_28)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_28)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_28 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_28)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_29 = 12528 + core_id * 6
    mapped_block_29 = pl.cast(pl.tensor.read(records, [record_29]), pl.INDEX)
    block_29 = pl.max(mapped_block_29, 0)
    if mapped_block_29 >= 0 and failed == 0:
        subslot_29 = pl.cast(pl.tensor.read(records, [record_29 + 1]), pl.INDEX)
        done_29 = pl.tensor.read(records, [record_29 + 2])
        wait_base_29 = pl.cast(pl.tensor.read(records, [record_29 + 3]), pl.INDEX)
        wait_count_29 = pl.cast(pl.tensor.read(records, [record_29 + 4]), pl.INDEX)
        publish_29 = pl.tensor.read(records, [record_29 + 5])
        for dependency_29 in pl.range(wait_count_29):
            if failed == 0:
                wait_index_29 = (wait_base_29 + dependency_29) * 2
                other_29 = pl.cast(pl.tensor.read(waits, [wait_index_29]), pl.INDEX)
                required_29 = pl.tensor.read(waits, [wait_index_29 + 1])
                target_29 = generation_base + required_29
                flag_29 = other_29 * 64
                pl.system.cacheinvalid(progress, [1], [flag_29])
                initial_29 = pl.tensor.read(progress, [flag_29])
                for (ready_29, polls_29) in pl.while_(init_values=(initial_29, 0)):
                    pl.cond(ready_29 < target_29 and polls_29 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_29])
                    (after_29, spin_count_29) = pl.yield_(pl.tensor.read(progress, [flag_29]), polls_29 + 1)
                if after_29 < target_29:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 29)
                    pl.tensor.write(progress, [own_flag + 10], other_29)
                    pl.tensor.write(progress, [own_flag + 11], target_29)
                    pl.tensor.write(progress, [own_flag + 12], after_29)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_29)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_29 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_29)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_30 = 12960 + core_id * 6
    mapped_block_30 = pl.cast(pl.tensor.read(records, [record_30]), pl.INDEX)
    block_30 = pl.max(mapped_block_30, 0)
    if mapped_block_30 >= 0 and failed == 0:
        subslot_30 = pl.cast(pl.tensor.read(records, [record_30 + 1]), pl.INDEX)
        done_30 = pl.tensor.read(records, [record_30 + 2])
        wait_base_30 = pl.cast(pl.tensor.read(records, [record_30 + 3]), pl.INDEX)
        wait_count_30 = pl.cast(pl.tensor.read(records, [record_30 + 4]), pl.INDEX)
        publish_30 = pl.tensor.read(records, [record_30 + 5])
        for dependency_30 in pl.range(wait_count_30):
            if failed == 0:
                wait_index_30 = (wait_base_30 + dependency_30) * 2
                other_30 = pl.cast(pl.tensor.read(waits, [wait_index_30]), pl.INDEX)
                required_30 = pl.tensor.read(waits, [wait_index_30 + 1])
                target_30 = generation_base + required_30
                flag_30 = other_30 * 64
                pl.system.cacheinvalid(progress, [1], [flag_30])
                initial_30 = pl.tensor.read(progress, [flag_30])
                for (ready_30, polls_30) in pl.while_(init_values=(initial_30, 0)):
                    pl.cond(ready_30 < target_30 and polls_30 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_30])
                    (after_30, spin_count_30) = pl.yield_(pl.tensor.read(progress, [flag_30]), polls_30 + 1)
                if after_30 < target_30:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 30)
                    pl.tensor.write(progress, [own_flag + 10], other_30)
                    pl.tensor.write(progress, [own_flag + 11], target_30)
                    pl.tensor.write(progress, [own_flag + 12], after_30)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_30)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_30 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_30)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_31 = 13392 + core_id * 6
    mapped_block_31 = pl.cast(pl.tensor.read(records, [record_31]), pl.INDEX)
    block_31 = pl.max(mapped_block_31, 0)
    if mapped_block_31 >= 0 and failed == 0:
        subslot_31 = pl.cast(pl.tensor.read(records, [record_31 + 1]), pl.INDEX)
        done_31 = pl.tensor.read(records, [record_31 + 2])
        wait_base_31 = pl.cast(pl.tensor.read(records, [record_31 + 3]), pl.INDEX)
        wait_count_31 = pl.cast(pl.tensor.read(records, [record_31 + 4]), pl.INDEX)
        publish_31 = pl.tensor.read(records, [record_31 + 5])
        for dependency_31 in pl.range(wait_count_31):
            if failed == 0:
                wait_index_31 = (wait_base_31 + dependency_31) * 2
                other_31 = pl.cast(pl.tensor.read(waits, [wait_index_31]), pl.INDEX)
                required_31 = pl.tensor.read(waits, [wait_index_31 + 1])
                target_31 = generation_base + required_31
                flag_31 = other_31 * 64
                pl.system.cacheinvalid(progress, [1], [flag_31])
                initial_31 = pl.tensor.read(progress, [flag_31])
                for (ready_31, polls_31) in pl.while_(init_values=(initial_31, 0)):
                    pl.cond(ready_31 < target_31 and polls_31 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_31])
                    (after_31, spin_count_31) = pl.yield_(pl.tensor.read(progress, [flag_31]), polls_31 + 1)
                if after_31 < target_31:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 31)
                    pl.tensor.write(progress, [own_flag + 10], other_31)
                    pl.tensor.write(progress, [own_flag + 11], target_31)
                    pl.tensor.write(progress, [own_flag + 12], after_31)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_31)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_31 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_31)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_32 = 13824 + core_id * 6
    mapped_block_32 = pl.cast(pl.tensor.read(records, [record_32]), pl.INDEX)
    block_32 = pl.max(mapped_block_32, 0)
    if mapped_block_32 >= 0 and failed == 0:
        subslot_32 = pl.cast(pl.tensor.read(records, [record_32 + 1]), pl.INDEX)
        done_32 = pl.tensor.read(records, [record_32 + 2])
        wait_base_32 = pl.cast(pl.tensor.read(records, [record_32 + 3]), pl.INDEX)
        wait_count_32 = pl.cast(pl.tensor.read(records, [record_32 + 4]), pl.INDEX)
        publish_32 = pl.tensor.read(records, [record_32 + 5])
        for dependency_32 in pl.range(wait_count_32):
            if failed == 0:
                wait_index_32 = (wait_base_32 + dependency_32) * 2
                other_32 = pl.cast(pl.tensor.read(waits, [wait_index_32]), pl.INDEX)
                required_32 = pl.tensor.read(waits, [wait_index_32 + 1])
                target_32 = generation_base + required_32
                flag_32 = other_32 * 64
                pl.system.cacheinvalid(progress, [1], [flag_32])
                initial_32 = pl.tensor.read(progress, [flag_32])
                for (ready_32, polls_32) in pl.while_(init_values=(initial_32, 0)):
                    pl.cond(ready_32 < target_32 and polls_32 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_32])
                    (after_32, spin_count_32) = pl.yield_(pl.tensor.read(progress, [flag_32]), polls_32 + 1)
                if after_32 < target_32:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 32)
                    pl.tensor.write(progress, [own_flag + 10], other_32)
                    pl.tensor.write(progress, [own_flag + 11], target_32)
                    pl.tensor.write(progress, [own_flag + 12], after_32)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_32)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_32 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_32)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_33 = 14256 + core_id * 6
    mapped_block_33 = pl.cast(pl.tensor.read(records, [record_33]), pl.INDEX)
    block_33 = pl.max(mapped_block_33, 0)
    if mapped_block_33 >= 0 and failed == 0:
        subslot_33 = pl.cast(pl.tensor.read(records, [record_33 + 1]), pl.INDEX)
        done_33 = pl.tensor.read(records, [record_33 + 2])
        wait_base_33 = pl.cast(pl.tensor.read(records, [record_33 + 3]), pl.INDEX)
        wait_count_33 = pl.cast(pl.tensor.read(records, [record_33 + 4]), pl.INDEX)
        publish_33 = pl.tensor.read(records, [record_33 + 5])
        for dependency_33 in pl.range(wait_count_33):
            if failed == 0:
                wait_index_33 = (wait_base_33 + dependency_33) * 2
                other_33 = pl.cast(pl.tensor.read(waits, [wait_index_33]), pl.INDEX)
                required_33 = pl.tensor.read(waits, [wait_index_33 + 1])
                target_33 = generation_base + required_33
                flag_33 = other_33 * 64
                pl.system.cacheinvalid(progress, [1], [flag_33])
                initial_33 = pl.tensor.read(progress, [flag_33])
                for (ready_33, polls_33) in pl.while_(init_values=(initial_33, 0)):
                    pl.cond(ready_33 < target_33 and polls_33 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_33])
                    (after_33, spin_count_33) = pl.yield_(pl.tensor.read(progress, [flag_33]), polls_33 + 1)
                if after_33 < target_33:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 33)
                    pl.tensor.write(progress, [own_flag + 10], other_33)
                    pl.tensor.write(progress, [own_flag + 11], target_33)
                    pl.tensor.write(progress, [own_flag + 12], after_33)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_33)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_33 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_33)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_34 = 14688 + core_id * 6
    mapped_block_34 = pl.cast(pl.tensor.read(records, [record_34]), pl.INDEX)
    block_34 = pl.max(mapped_block_34, 0)
    if mapped_block_34 >= 0 and failed == 0:
        subslot_34 = pl.cast(pl.tensor.read(records, [record_34 + 1]), pl.INDEX)
        done_34 = pl.tensor.read(records, [record_34 + 2])
        wait_base_34 = pl.cast(pl.tensor.read(records, [record_34 + 3]), pl.INDEX)
        wait_count_34 = pl.cast(pl.tensor.read(records, [record_34 + 4]), pl.INDEX)
        publish_34 = pl.tensor.read(records, [record_34 + 5])
        for dependency_34 in pl.range(wait_count_34):
            if failed == 0:
                wait_index_34 = (wait_base_34 + dependency_34) * 2
                other_34 = pl.cast(pl.tensor.read(waits, [wait_index_34]), pl.INDEX)
                required_34 = pl.tensor.read(waits, [wait_index_34 + 1])
                target_34 = generation_base + required_34
                flag_34 = other_34 * 64
                pl.system.cacheinvalid(progress, [1], [flag_34])
                initial_34 = pl.tensor.read(progress, [flag_34])
                for (ready_34, polls_34) in pl.while_(init_values=(initial_34, 0)):
                    pl.cond(ready_34 < target_34 and polls_34 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_34])
                    (after_34, spin_count_34) = pl.yield_(pl.tensor.read(progress, [flag_34]), polls_34 + 1)
                if after_34 < target_34:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 34)
                    pl.tensor.write(progress, [own_flag + 10], other_34)
                    pl.tensor.write(progress, [own_flag + 11], target_34)
                    pl.tensor.write(progress, [own_flag + 12], after_34)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_34)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_34 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_34)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_35 = 15120 + core_id * 6
    mapped_block_35 = pl.cast(pl.tensor.read(records, [record_35]), pl.INDEX)
    block_35 = pl.max(mapped_block_35, 0)
    if mapped_block_35 >= 0 and failed == 0:
        subslot_35 = pl.cast(pl.tensor.read(records, [record_35 + 1]), pl.INDEX)
        done_35 = pl.tensor.read(records, [record_35 + 2])
        wait_base_35 = pl.cast(pl.tensor.read(records, [record_35 + 3]), pl.INDEX)
        wait_count_35 = pl.cast(pl.tensor.read(records, [record_35 + 4]), pl.INDEX)
        publish_35 = pl.tensor.read(records, [record_35 + 5])
        for dependency_35 in pl.range(wait_count_35):
            if failed == 0:
                wait_index_35 = (wait_base_35 + dependency_35) * 2
                other_35 = pl.cast(pl.tensor.read(waits, [wait_index_35]), pl.INDEX)
                required_35 = pl.tensor.read(waits, [wait_index_35 + 1])
                target_35 = generation_base + required_35
                flag_35 = other_35 * 64
                pl.system.cacheinvalid(progress, [1], [flag_35])
                initial_35 = pl.tensor.read(progress, [flag_35])
                for (ready_35, polls_35) in pl.while_(init_values=(initial_35, 0)):
                    pl.cond(ready_35 < target_35 and polls_35 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_35])
                    (after_35, spin_count_35) = pl.yield_(pl.tensor.read(progress, [flag_35]), polls_35 + 1)
                if after_35 < target_35:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 35)
                    pl.tensor.write(progress, [own_flag + 10], other_35)
                    pl.tensor.write(progress, [own_flag + 11], target_35)
                    pl.tensor.write(progress, [own_flag + 12], after_35)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_35)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_35 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_35)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_36 = 15552 + core_id * 6
    mapped_block_36 = pl.cast(pl.tensor.read(records, [record_36]), pl.INDEX)
    block_36 = pl.max(mapped_block_36, 0)
    if mapped_block_36 >= 0 and failed == 0:
        subslot_36 = pl.cast(pl.tensor.read(records, [record_36 + 1]), pl.INDEX)
        done_36 = pl.tensor.read(records, [record_36 + 2])
        wait_base_36 = pl.cast(pl.tensor.read(records, [record_36 + 3]), pl.INDEX)
        wait_count_36 = pl.cast(pl.tensor.read(records, [record_36 + 4]), pl.INDEX)
        publish_36 = pl.tensor.read(records, [record_36 + 5])
        for dependency_36 in pl.range(wait_count_36):
            if failed == 0:
                wait_index_36 = (wait_base_36 + dependency_36) * 2
                other_36 = pl.cast(pl.tensor.read(waits, [wait_index_36]), pl.INDEX)
                required_36 = pl.tensor.read(waits, [wait_index_36 + 1])
                target_36 = generation_base + required_36
                flag_36 = other_36 * 64
                pl.system.cacheinvalid(progress, [1], [flag_36])
                initial_36 = pl.tensor.read(progress, [flag_36])
                for (ready_36, polls_36) in pl.while_(init_values=(initial_36, 0)):
                    pl.cond(ready_36 < target_36 and polls_36 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_36])
                    (after_36, spin_count_36) = pl.yield_(pl.tensor.read(progress, [flag_36]), polls_36 + 1)
                if after_36 < target_36:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 36)
                    pl.tensor.write(progress, [own_flag + 10], other_36)
                    pl.tensor.write(progress, [own_flag + 11], target_36)
                    pl.tensor.write(progress, [own_flag + 12], after_36)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_36)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_36 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_36)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_37 = 15984 + core_id * 6
    mapped_block_37 = pl.cast(pl.tensor.read(records, [record_37]), pl.INDEX)
    block_37 = pl.max(mapped_block_37, 0)
    if mapped_block_37 >= 0 and failed == 0:
        subslot_37 = pl.cast(pl.tensor.read(records, [record_37 + 1]), pl.INDEX)
        done_37 = pl.tensor.read(records, [record_37 + 2])
        wait_base_37 = pl.cast(pl.tensor.read(records, [record_37 + 3]), pl.INDEX)
        wait_count_37 = pl.cast(pl.tensor.read(records, [record_37 + 4]), pl.INDEX)
        publish_37 = pl.tensor.read(records, [record_37 + 5])
        for dependency_37 in pl.range(wait_count_37):
            if failed == 0:
                wait_index_37 = (wait_base_37 + dependency_37) * 2
                other_37 = pl.cast(pl.tensor.read(waits, [wait_index_37]), pl.INDEX)
                required_37 = pl.tensor.read(waits, [wait_index_37 + 1])
                target_37 = generation_base + required_37
                flag_37 = other_37 * 64
                pl.system.cacheinvalid(progress, [1], [flag_37])
                initial_37 = pl.tensor.read(progress, [flag_37])
                for (ready_37, polls_37) in pl.while_(init_values=(initial_37, 0)):
                    pl.cond(ready_37 < target_37 and polls_37 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_37])
                    (after_37, spin_count_37) = pl.yield_(pl.tensor.read(progress, [flag_37]), polls_37 + 1)
                if after_37 < target_37:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 37)
                    pl.tensor.write(progress, [own_flag + 10], other_37)
                    pl.tensor.write(progress, [own_flag + 11], target_37)
                    pl.tensor.write(progress, [own_flag + 12], after_37)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_37)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_37 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_37)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_38 = 16416 + core_id * 6
    mapped_block_38 = pl.cast(pl.tensor.read(records, [record_38]), pl.INDEX)
    block_38 = pl.max(mapped_block_38, 0)
    if mapped_block_38 >= 0 and failed == 0:
        subslot_38 = pl.cast(pl.tensor.read(records, [record_38 + 1]), pl.INDEX)
        done_38 = pl.tensor.read(records, [record_38 + 2])
        wait_base_38 = pl.cast(pl.tensor.read(records, [record_38 + 3]), pl.INDEX)
        wait_count_38 = pl.cast(pl.tensor.read(records, [record_38 + 4]), pl.INDEX)
        publish_38 = pl.tensor.read(records, [record_38 + 5])
        for dependency_38 in pl.range(wait_count_38):
            if failed == 0:
                wait_index_38 = (wait_base_38 + dependency_38) * 2
                other_38 = pl.cast(pl.tensor.read(waits, [wait_index_38]), pl.INDEX)
                required_38 = pl.tensor.read(waits, [wait_index_38 + 1])
                target_38 = generation_base + required_38
                flag_38 = other_38 * 64
                pl.system.cacheinvalid(progress, [1], [flag_38])
                initial_38 = pl.tensor.read(progress, [flag_38])
                for (ready_38, polls_38) in pl.while_(init_values=(initial_38, 0)):
                    pl.cond(ready_38 < target_38 and polls_38 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_38])
                    (after_38, spin_count_38) = pl.yield_(pl.tensor.read(progress, [flag_38]), polls_38 + 1)
                if after_38 < target_38:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 38)
                    pl.tensor.write(progress, [own_flag + 10], other_38)
                    pl.tensor.write(progress, [own_flag + 11], target_38)
                    pl.tensor.write(progress, [own_flag + 12], after_38)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_38)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_38 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_38)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_39 = 16848 + core_id * 6
    mapped_block_39 = pl.cast(pl.tensor.read(records, [record_39]), pl.INDEX)
    block_39 = pl.max(mapped_block_39, 0)
    if mapped_block_39 >= 0 and failed == 0:
        subslot_39 = pl.cast(pl.tensor.read(records, [record_39 + 1]), pl.INDEX)
        done_39 = pl.tensor.read(records, [record_39 + 2])
        wait_base_39 = pl.cast(pl.tensor.read(records, [record_39 + 3]), pl.INDEX)
        wait_count_39 = pl.cast(pl.tensor.read(records, [record_39 + 4]), pl.INDEX)
        publish_39 = pl.tensor.read(records, [record_39 + 5])
        for dependency_39 in pl.range(wait_count_39):
            if failed == 0:
                wait_index_39 = (wait_base_39 + dependency_39) * 2
                other_39 = pl.cast(pl.tensor.read(waits, [wait_index_39]), pl.INDEX)
                required_39 = pl.tensor.read(waits, [wait_index_39 + 1])
                target_39 = generation_base + required_39
                flag_39 = other_39 * 64
                pl.system.cacheinvalid(progress, [1], [flag_39])
                initial_39 = pl.tensor.read(progress, [flag_39])
                for (ready_39, polls_39) in pl.while_(init_values=(initial_39, 0)):
                    pl.cond(ready_39 < target_39 and polls_39 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_39])
                    (after_39, spin_count_39) = pl.yield_(pl.tensor.read(progress, [flag_39]), polls_39 + 1)
                if after_39 < target_39:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 39)
                    pl.tensor.write(progress, [own_flag + 10], other_39)
                    pl.tensor.write(progress, [own_flag + 11], target_39)
                    pl.tensor.write(progress, [own_flag + 12], after_39)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_39)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_39 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_39)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    return progress

@pl.jit.incore
def csa_static_aiv(progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> pl.Tensor[[4608], pl.INT64]:
    pl.func_attr({'mx_tensor_views_blocked': True, 'split_aiv': True})
    failed = 0
    group_id = pl.tile.get_block_idx()
    lane_id = pl.tile.get_subblock_idx()
    core_id = 24 + group_id * 2 + lane_id
    own_flag = core_id * 64
    pl.system.cacheinvalid(progress, [1], [own_flag])
    prior_progress = pl.tensor.read(progress, [own_flag])
    generation_base = (prior_progress // 1024 + 1) * 1024
    pl.tensor.write(progress, [own_flag], generation_base)
    pl.system.cacheinvalid(progress, [1], [own_flag])
    pl.system.fence()
    record_0 = 17280 + core_id * 6
    mapped_block_0 = pl.cast(pl.tensor.read(records, [record_0]), pl.INDEX)
    block_0 = pl.max(mapped_block_0, 0)
    if mapped_block_0 >= 0 and failed == 0:
        subslot_0 = pl.cast(pl.tensor.read(records, [record_0 + 1]), pl.INDEX)
        done_0 = pl.tensor.read(records, [record_0 + 2])
        wait_base_0 = pl.cast(pl.tensor.read(records, [record_0 + 3]), pl.INDEX)
        wait_count_0 = pl.cast(pl.tensor.read(records, [record_0 + 4]), pl.INDEX)
        publish_0 = pl.tensor.read(records, [record_0 + 5])
        for dependency_0 in pl.range(wait_count_0):
            if failed == 0:
                wait_index_0 = (wait_base_0 + dependency_0) * 2
                other_0 = pl.cast(pl.tensor.read(waits, [wait_index_0]), pl.INDEX)
                required_0 = pl.tensor.read(waits, [wait_index_0 + 1])
                target_0 = generation_base + required_0
                flag_0 = other_0 * 64
                pl.system.cacheinvalid(progress, [1], [flag_0])
                initial_0 = pl.tensor.read(progress, [flag_0])
                for (ready_0, polls_0) in pl.while_(init_values=(initial_0, 0)):
                    pl.cond(ready_0 < target_0 and polls_0 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_0])
                    (after_0, spin_count_0) = pl.yield_(pl.tensor.read(progress, [flag_0]), polls_0 + 1)
                if after_0 < target_0:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 0)
                    pl.tensor.write(progress, [own_flag + 10], other_0)
                    pl.tensor.write(progress, [own_flag + 11], target_0)
                    pl.tensor.write(progress, [own_flag + 12], after_0)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_0)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_0 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_0)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_1 = 17712 + core_id * 6
    mapped_block_1 = pl.cast(pl.tensor.read(records, [record_1]), pl.INDEX)
    block_1 = pl.max(mapped_block_1, 0)
    if mapped_block_1 >= 0 and failed == 0:
        subslot_1 = pl.cast(pl.tensor.read(records, [record_1 + 1]), pl.INDEX)
        done_1 = pl.tensor.read(records, [record_1 + 2])
        wait_base_1 = pl.cast(pl.tensor.read(records, [record_1 + 3]), pl.INDEX)
        wait_count_1 = pl.cast(pl.tensor.read(records, [record_1 + 4]), pl.INDEX)
        publish_1 = pl.tensor.read(records, [record_1 + 5])
        for dependency_1 in pl.range(wait_count_1):
            if failed == 0:
                wait_index_1 = (wait_base_1 + dependency_1) * 2
                other_1 = pl.cast(pl.tensor.read(waits, [wait_index_1]), pl.INDEX)
                required_1 = pl.tensor.read(waits, [wait_index_1 + 1])
                target_1 = generation_base + required_1
                flag_1 = other_1 * 64
                pl.system.cacheinvalid(progress, [1], [flag_1])
                initial_1 = pl.tensor.read(progress, [flag_1])
                for (ready_1, polls_1) in pl.while_(init_values=(initial_1, 0)):
                    pl.cond(ready_1 < target_1 and polls_1 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_1])
                    (after_1, spin_count_1) = pl.yield_(pl.tensor.read(progress, [flag_1]), polls_1 + 1)
                if after_1 < target_1:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 1)
                    pl.tensor.write(progress, [own_flag + 10], other_1)
                    pl.tensor.write(progress, [own_flag + 11], target_1)
                    pl.tensor.write(progress, [own_flag + 12], after_1)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_1)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_1 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_1)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_2 = 18144 + core_id * 6
    mapped_block_2 = pl.cast(pl.tensor.read(records, [record_2]), pl.INDEX)
    block_2 = pl.max(mapped_block_2, 0)
    if mapped_block_2 >= 0 and failed == 0:
        subslot_2 = pl.cast(pl.tensor.read(records, [record_2 + 1]), pl.INDEX)
        done_2 = pl.tensor.read(records, [record_2 + 2])
        wait_base_2 = pl.cast(pl.tensor.read(records, [record_2 + 3]), pl.INDEX)
        wait_count_2 = pl.cast(pl.tensor.read(records, [record_2 + 4]), pl.INDEX)
        publish_2 = pl.tensor.read(records, [record_2 + 5])
        for dependency_2 in pl.range(wait_count_2):
            if failed == 0:
                wait_index_2 = (wait_base_2 + dependency_2) * 2
                other_2 = pl.cast(pl.tensor.read(waits, [wait_index_2]), pl.INDEX)
                required_2 = pl.tensor.read(waits, [wait_index_2 + 1])
                target_2 = generation_base + required_2
                flag_2 = other_2 * 64
                pl.system.cacheinvalid(progress, [1], [flag_2])
                initial_2 = pl.tensor.read(progress, [flag_2])
                for (ready_2, polls_2) in pl.while_(init_values=(initial_2, 0)):
                    pl.cond(ready_2 < target_2 and polls_2 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_2])
                    (after_2, spin_count_2) = pl.yield_(pl.tensor.read(progress, [flag_2]), polls_2 + 1)
                if after_2 < target_2:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 2)
                    pl.tensor.write(progress, [own_flag + 10], other_2)
                    pl.tensor.write(progress, [own_flag + 11], target_2)
                    pl.tensor.write(progress, [own_flag + 12], after_2)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_2)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_2 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_2)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_3 = 18576 + core_id * 6
    mapped_block_3 = pl.cast(pl.tensor.read(records, [record_3]), pl.INDEX)
    block_3 = pl.max(mapped_block_3, 0)
    if mapped_block_3 >= 0 and failed == 0:
        subslot_3 = pl.cast(pl.tensor.read(records, [record_3 + 1]), pl.INDEX)
        done_3 = pl.tensor.read(records, [record_3 + 2])
        wait_base_3 = pl.cast(pl.tensor.read(records, [record_3 + 3]), pl.INDEX)
        wait_count_3 = pl.cast(pl.tensor.read(records, [record_3 + 4]), pl.INDEX)
        publish_3 = pl.tensor.read(records, [record_3 + 5])
        for dependency_3 in pl.range(wait_count_3):
            if failed == 0:
                wait_index_3 = (wait_base_3 + dependency_3) * 2
                other_3 = pl.cast(pl.tensor.read(waits, [wait_index_3]), pl.INDEX)
                required_3 = pl.tensor.read(waits, [wait_index_3 + 1])
                target_3 = generation_base + required_3
                flag_3 = other_3 * 64
                pl.system.cacheinvalid(progress, [1], [flag_3])
                initial_3 = pl.tensor.read(progress, [flag_3])
                for (ready_3, polls_3) in pl.while_(init_values=(initial_3, 0)):
                    pl.cond(ready_3 < target_3 and polls_3 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_3])
                    (after_3, spin_count_3) = pl.yield_(pl.tensor.read(progress, [flag_3]), polls_3 + 1)
                if after_3 < target_3:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 3)
                    pl.tensor.write(progress, [own_flag + 10], other_3)
                    pl.tensor.write(progress, [own_flag + 11], target_3)
                    pl.tensor.write(progress, [own_flag + 12], after_3)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_3)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_3 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_3)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_4 = 19008 + core_id * 6
    mapped_block_4 = pl.cast(pl.tensor.read(records, [record_4]), pl.INDEX)
    block_4 = pl.max(mapped_block_4, 0)
    if mapped_block_4 >= 0 and failed == 0:
        subslot_4 = pl.cast(pl.tensor.read(records, [record_4 + 1]), pl.INDEX)
        done_4 = pl.tensor.read(records, [record_4 + 2])
        wait_base_4 = pl.cast(pl.tensor.read(records, [record_4 + 3]), pl.INDEX)
        wait_count_4 = pl.cast(pl.tensor.read(records, [record_4 + 4]), pl.INDEX)
        publish_4 = pl.tensor.read(records, [record_4 + 5])
        for dependency_4 in pl.range(wait_count_4):
            if failed == 0:
                wait_index_4 = (wait_base_4 + dependency_4) * 2
                other_4 = pl.cast(pl.tensor.read(waits, [wait_index_4]), pl.INDEX)
                required_4 = pl.tensor.read(waits, [wait_index_4 + 1])
                target_4 = generation_base + required_4
                flag_4 = other_4 * 64
                pl.system.cacheinvalid(progress, [1], [flag_4])
                initial_4 = pl.tensor.read(progress, [flag_4])
                for (ready_4, polls_4) in pl.while_(init_values=(initial_4, 0)):
                    pl.cond(ready_4 < target_4 and polls_4 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_4])
                    (after_4, spin_count_4) = pl.yield_(pl.tensor.read(progress, [flag_4]), polls_4 + 1)
                if after_4 < target_4:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 4)
                    pl.tensor.write(progress, [own_flag + 10], other_4)
                    pl.tensor.write(progress, [own_flag + 11], target_4)
                    pl.tensor.write(progress, [own_flag + 12], after_4)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_4)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_4 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_4)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_5 = 19440 + core_id * 6
    mapped_block_5 = pl.cast(pl.tensor.read(records, [record_5]), pl.INDEX)
    block_5 = pl.max(mapped_block_5, 0)
    if mapped_block_5 >= 0 and failed == 0:
        subslot_5 = pl.cast(pl.tensor.read(records, [record_5 + 1]), pl.INDEX)
        done_5 = pl.tensor.read(records, [record_5 + 2])
        wait_base_5 = pl.cast(pl.tensor.read(records, [record_5 + 3]), pl.INDEX)
        wait_count_5 = pl.cast(pl.tensor.read(records, [record_5 + 4]), pl.INDEX)
        publish_5 = pl.tensor.read(records, [record_5 + 5])
        for dependency_5 in pl.range(wait_count_5):
            if failed == 0:
                wait_index_5 = (wait_base_5 + dependency_5) * 2
                other_5 = pl.cast(pl.tensor.read(waits, [wait_index_5]), pl.INDEX)
                required_5 = pl.tensor.read(waits, [wait_index_5 + 1])
                target_5 = generation_base + required_5
                flag_5 = other_5 * 64
                pl.system.cacheinvalid(progress, [1], [flag_5])
                initial_5 = pl.tensor.read(progress, [flag_5])
                for (ready_5, polls_5) in pl.while_(init_values=(initial_5, 0)):
                    pl.cond(ready_5 < target_5 and polls_5 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_5])
                    (after_5, spin_count_5) = pl.yield_(pl.tensor.read(progress, [flag_5]), polls_5 + 1)
                if after_5 < target_5:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 5)
                    pl.tensor.write(progress, [own_flag + 10], other_5)
                    pl.tensor.write(progress, [own_flag + 11], target_5)
                    pl.tensor.write(progress, [own_flag + 12], after_5)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_5)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_5 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_5)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_6 = 19872 + core_id * 6
    mapped_block_6 = pl.cast(pl.tensor.read(records, [record_6]), pl.INDEX)
    block_6 = pl.max(mapped_block_6, 0)
    if mapped_block_6 >= 0 and failed == 0:
        subslot_6 = pl.cast(pl.tensor.read(records, [record_6 + 1]), pl.INDEX)
        done_6 = pl.tensor.read(records, [record_6 + 2])
        wait_base_6 = pl.cast(pl.tensor.read(records, [record_6 + 3]), pl.INDEX)
        wait_count_6 = pl.cast(pl.tensor.read(records, [record_6 + 4]), pl.INDEX)
        publish_6 = pl.tensor.read(records, [record_6 + 5])
        for dependency_6 in pl.range(wait_count_6):
            if failed == 0:
                wait_index_6 = (wait_base_6 + dependency_6) * 2
                other_6 = pl.cast(pl.tensor.read(waits, [wait_index_6]), pl.INDEX)
                required_6 = pl.tensor.read(waits, [wait_index_6 + 1])
                target_6 = generation_base + required_6
                flag_6 = other_6 * 64
                pl.system.cacheinvalid(progress, [1], [flag_6])
                initial_6 = pl.tensor.read(progress, [flag_6])
                for (ready_6, polls_6) in pl.while_(init_values=(initial_6, 0)):
                    pl.cond(ready_6 < target_6 and polls_6 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_6])
                    (after_6, spin_count_6) = pl.yield_(pl.tensor.read(progress, [flag_6]), polls_6 + 1)
                if after_6 < target_6:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 6)
                    pl.tensor.write(progress, [own_flag + 10], other_6)
                    pl.tensor.write(progress, [own_flag + 11], target_6)
                    pl.tensor.write(progress, [own_flag + 12], after_6)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_6)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_6 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_6)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_7 = 20304 + core_id * 6
    mapped_block_7 = pl.cast(pl.tensor.read(records, [record_7]), pl.INDEX)
    block_7 = pl.max(mapped_block_7, 0)
    if mapped_block_7 >= 0 and failed == 0:
        subslot_7 = pl.cast(pl.tensor.read(records, [record_7 + 1]), pl.INDEX)
        done_7 = pl.tensor.read(records, [record_7 + 2])
        wait_base_7 = pl.cast(pl.tensor.read(records, [record_7 + 3]), pl.INDEX)
        wait_count_7 = pl.cast(pl.tensor.read(records, [record_7 + 4]), pl.INDEX)
        publish_7 = pl.tensor.read(records, [record_7 + 5])
        for dependency_7 in pl.range(wait_count_7):
            if failed == 0:
                wait_index_7 = (wait_base_7 + dependency_7) * 2
                other_7 = pl.cast(pl.tensor.read(waits, [wait_index_7]), pl.INDEX)
                required_7 = pl.tensor.read(waits, [wait_index_7 + 1])
                target_7 = generation_base + required_7
                flag_7 = other_7 * 64
                pl.system.cacheinvalid(progress, [1], [flag_7])
                initial_7 = pl.tensor.read(progress, [flag_7])
                for (ready_7, polls_7) in pl.while_(init_values=(initial_7, 0)):
                    pl.cond(ready_7 < target_7 and polls_7 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_7])
                    (after_7, spin_count_7) = pl.yield_(pl.tensor.read(progress, [flag_7]), polls_7 + 1)
                if after_7 < target_7:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 7)
                    pl.tensor.write(progress, [own_flag + 10], other_7)
                    pl.tensor.write(progress, [own_flag + 11], target_7)
                    pl.tensor.write(progress, [own_flag + 12], after_7)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_7)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_7 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_7)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_8 = 20736 + core_id * 6
    mapped_block_8 = pl.cast(pl.tensor.read(records, [record_8]), pl.INDEX)
    block_8 = pl.max(mapped_block_8, 0)
    if mapped_block_8 >= 0 and failed == 0:
        subslot_8 = pl.cast(pl.tensor.read(records, [record_8 + 1]), pl.INDEX)
        done_8 = pl.tensor.read(records, [record_8 + 2])
        wait_base_8 = pl.cast(pl.tensor.read(records, [record_8 + 3]), pl.INDEX)
        wait_count_8 = pl.cast(pl.tensor.read(records, [record_8 + 4]), pl.INDEX)
        publish_8 = pl.tensor.read(records, [record_8 + 5])
        for dependency_8 in pl.range(wait_count_8):
            if failed == 0:
                wait_index_8 = (wait_base_8 + dependency_8) * 2
                other_8 = pl.cast(pl.tensor.read(waits, [wait_index_8]), pl.INDEX)
                required_8 = pl.tensor.read(waits, [wait_index_8 + 1])
                target_8 = generation_base + required_8
                flag_8 = other_8 * 64
                pl.system.cacheinvalid(progress, [1], [flag_8])
                initial_8 = pl.tensor.read(progress, [flag_8])
                for (ready_8, polls_8) in pl.while_(init_values=(initial_8, 0)):
                    pl.cond(ready_8 < target_8 and polls_8 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_8])
                    (after_8, spin_count_8) = pl.yield_(pl.tensor.read(progress, [flag_8]), polls_8 + 1)
                if after_8 < target_8:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 8)
                    pl.tensor.write(progress, [own_flag + 10], other_8)
                    pl.tensor.write(progress, [own_flag + 11], target_8)
                    pl.tensor.write(progress, [own_flag + 12], after_8)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_8)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_8 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_8)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_9 = 21168 + core_id * 6
    mapped_block_9 = pl.cast(pl.tensor.read(records, [record_9]), pl.INDEX)
    block_9 = pl.max(mapped_block_9, 0)
    if mapped_block_9 >= 0 and failed == 0:
        subslot_9 = pl.cast(pl.tensor.read(records, [record_9 + 1]), pl.INDEX)
        done_9 = pl.tensor.read(records, [record_9 + 2])
        wait_base_9 = pl.cast(pl.tensor.read(records, [record_9 + 3]), pl.INDEX)
        wait_count_9 = pl.cast(pl.tensor.read(records, [record_9 + 4]), pl.INDEX)
        publish_9 = pl.tensor.read(records, [record_9 + 5])
        for dependency_9 in pl.range(wait_count_9):
            if failed == 0:
                wait_index_9 = (wait_base_9 + dependency_9) * 2
                other_9 = pl.cast(pl.tensor.read(waits, [wait_index_9]), pl.INDEX)
                required_9 = pl.tensor.read(waits, [wait_index_9 + 1])
                target_9 = generation_base + required_9
                flag_9 = other_9 * 64
                pl.system.cacheinvalid(progress, [1], [flag_9])
                initial_9 = pl.tensor.read(progress, [flag_9])
                for (ready_9, polls_9) in pl.while_(init_values=(initial_9, 0)):
                    pl.cond(ready_9 < target_9 and polls_9 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_9])
                    (after_9, spin_count_9) = pl.yield_(pl.tensor.read(progress, [flag_9]), polls_9 + 1)
                if after_9 < target_9:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 9)
                    pl.tensor.write(progress, [own_flag + 10], other_9)
                    pl.tensor.write(progress, [own_flag + 11], target_9)
                    pl.tensor.write(progress, [own_flag + 12], after_9)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_9)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_9 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_9)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_10 = 21600 + core_id * 6
    mapped_block_10 = pl.cast(pl.tensor.read(records, [record_10]), pl.INDEX)
    block_10 = pl.max(mapped_block_10, 0)
    if mapped_block_10 >= 0 and failed == 0:
        subslot_10 = pl.cast(pl.tensor.read(records, [record_10 + 1]), pl.INDEX)
        done_10 = pl.tensor.read(records, [record_10 + 2])
        wait_base_10 = pl.cast(pl.tensor.read(records, [record_10 + 3]), pl.INDEX)
        wait_count_10 = pl.cast(pl.tensor.read(records, [record_10 + 4]), pl.INDEX)
        publish_10 = pl.tensor.read(records, [record_10 + 5])
        for dependency_10 in pl.range(wait_count_10):
            if failed == 0:
                wait_index_10 = (wait_base_10 + dependency_10) * 2
                other_10 = pl.cast(pl.tensor.read(waits, [wait_index_10]), pl.INDEX)
                required_10 = pl.tensor.read(waits, [wait_index_10 + 1])
                target_10 = generation_base + required_10
                flag_10 = other_10 * 64
                pl.system.cacheinvalid(progress, [1], [flag_10])
                initial_10 = pl.tensor.read(progress, [flag_10])
                for (ready_10, polls_10) in pl.while_(init_values=(initial_10, 0)):
                    pl.cond(ready_10 < target_10 and polls_10 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_10])
                    (after_10, spin_count_10) = pl.yield_(pl.tensor.read(progress, [flag_10]), polls_10 + 1)
                if after_10 < target_10:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 10)
                    pl.tensor.write(progress, [own_flag + 10], other_10)
                    pl.tensor.write(progress, [own_flag + 11], target_10)
                    pl.tensor.write(progress, [own_flag + 12], after_10)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_10)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_10 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_10)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_11 = 22032 + core_id * 6
    mapped_block_11 = pl.cast(pl.tensor.read(records, [record_11]), pl.INDEX)
    block_11 = pl.max(mapped_block_11, 0)
    if mapped_block_11 >= 0 and failed == 0:
        subslot_11 = pl.cast(pl.tensor.read(records, [record_11 + 1]), pl.INDEX)
        done_11 = pl.tensor.read(records, [record_11 + 2])
        wait_base_11 = pl.cast(pl.tensor.read(records, [record_11 + 3]), pl.INDEX)
        wait_count_11 = pl.cast(pl.tensor.read(records, [record_11 + 4]), pl.INDEX)
        publish_11 = pl.tensor.read(records, [record_11 + 5])
        for dependency_11 in pl.range(wait_count_11):
            if failed == 0:
                wait_index_11 = (wait_base_11 + dependency_11) * 2
                other_11 = pl.cast(pl.tensor.read(waits, [wait_index_11]), pl.INDEX)
                required_11 = pl.tensor.read(waits, [wait_index_11 + 1])
                target_11 = generation_base + required_11
                flag_11 = other_11 * 64
                pl.system.cacheinvalid(progress, [1], [flag_11])
                initial_11 = pl.tensor.read(progress, [flag_11])
                for (ready_11, polls_11) in pl.while_(init_values=(initial_11, 0)):
                    pl.cond(ready_11 < target_11 and polls_11 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_11])
                    (after_11, spin_count_11) = pl.yield_(pl.tensor.read(progress, [flag_11]), polls_11 + 1)
                if after_11 < target_11:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 11)
                    pl.tensor.write(progress, [own_flag + 10], other_11)
                    pl.tensor.write(progress, [own_flag + 11], target_11)
                    pl.tensor.write(progress, [own_flag + 12], after_11)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_11)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_11 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_11)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_12 = 22464 + core_id * 6
    mapped_block_12 = pl.cast(pl.tensor.read(records, [record_12]), pl.INDEX)
    block_12 = pl.max(mapped_block_12, 0)
    if mapped_block_12 >= 0 and failed == 0:
        subslot_12 = pl.cast(pl.tensor.read(records, [record_12 + 1]), pl.INDEX)
        done_12 = pl.tensor.read(records, [record_12 + 2])
        wait_base_12 = pl.cast(pl.tensor.read(records, [record_12 + 3]), pl.INDEX)
        wait_count_12 = pl.cast(pl.tensor.read(records, [record_12 + 4]), pl.INDEX)
        publish_12 = pl.tensor.read(records, [record_12 + 5])
        for dependency_12 in pl.range(wait_count_12):
            if failed == 0:
                wait_index_12 = (wait_base_12 + dependency_12) * 2
                other_12 = pl.cast(pl.tensor.read(waits, [wait_index_12]), pl.INDEX)
                required_12 = pl.tensor.read(waits, [wait_index_12 + 1])
                target_12 = generation_base + required_12
                flag_12 = other_12 * 64
                pl.system.cacheinvalid(progress, [1], [flag_12])
                initial_12 = pl.tensor.read(progress, [flag_12])
                for (ready_12, polls_12) in pl.while_(init_values=(initial_12, 0)):
                    pl.cond(ready_12 < target_12 and polls_12 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_12])
                    (after_12, spin_count_12) = pl.yield_(pl.tensor.read(progress, [flag_12]), polls_12 + 1)
                if after_12 < target_12:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 12)
                    pl.tensor.write(progress, [own_flag + 10], other_12)
                    pl.tensor.write(progress, [own_flag + 11], target_12)
                    pl.tensor.write(progress, [own_flag + 12], after_12)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_12)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_12 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_12)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_13 = 22896 + core_id * 6
    mapped_block_13 = pl.cast(pl.tensor.read(records, [record_13]), pl.INDEX)
    block_13 = pl.max(mapped_block_13, 0)
    if mapped_block_13 >= 0 and failed == 0:
        subslot_13 = pl.cast(pl.tensor.read(records, [record_13 + 1]), pl.INDEX)
        done_13 = pl.tensor.read(records, [record_13 + 2])
        wait_base_13 = pl.cast(pl.tensor.read(records, [record_13 + 3]), pl.INDEX)
        wait_count_13 = pl.cast(pl.tensor.read(records, [record_13 + 4]), pl.INDEX)
        publish_13 = pl.tensor.read(records, [record_13 + 5])
        for dependency_13 in pl.range(wait_count_13):
            if failed == 0:
                wait_index_13 = (wait_base_13 + dependency_13) * 2
                other_13 = pl.cast(pl.tensor.read(waits, [wait_index_13]), pl.INDEX)
                required_13 = pl.tensor.read(waits, [wait_index_13 + 1])
                target_13 = generation_base + required_13
                flag_13 = other_13 * 64
                pl.system.cacheinvalid(progress, [1], [flag_13])
                initial_13 = pl.tensor.read(progress, [flag_13])
                for (ready_13, polls_13) in pl.while_(init_values=(initial_13, 0)):
                    pl.cond(ready_13 < target_13 and polls_13 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_13])
                    (after_13, spin_count_13) = pl.yield_(pl.tensor.read(progress, [flag_13]), polls_13 + 1)
                if after_13 < target_13:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 13)
                    pl.tensor.write(progress, [own_flag + 10], other_13)
                    pl.tensor.write(progress, [own_flag + 11], target_13)
                    pl.tensor.write(progress, [own_flag + 12], after_13)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_13)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_13 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_13)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_14 = 23328 + core_id * 6
    mapped_block_14 = pl.cast(pl.tensor.read(records, [record_14]), pl.INDEX)
    block_14 = pl.max(mapped_block_14, 0)
    if mapped_block_14 >= 0 and failed == 0:
        subslot_14 = pl.cast(pl.tensor.read(records, [record_14 + 1]), pl.INDEX)
        done_14 = pl.tensor.read(records, [record_14 + 2])
        wait_base_14 = pl.cast(pl.tensor.read(records, [record_14 + 3]), pl.INDEX)
        wait_count_14 = pl.cast(pl.tensor.read(records, [record_14 + 4]), pl.INDEX)
        publish_14 = pl.tensor.read(records, [record_14 + 5])
        for dependency_14 in pl.range(wait_count_14):
            if failed == 0:
                wait_index_14 = (wait_base_14 + dependency_14) * 2
                other_14 = pl.cast(pl.tensor.read(waits, [wait_index_14]), pl.INDEX)
                required_14 = pl.tensor.read(waits, [wait_index_14 + 1])
                target_14 = generation_base + required_14
                flag_14 = other_14 * 64
                pl.system.cacheinvalid(progress, [1], [flag_14])
                initial_14 = pl.tensor.read(progress, [flag_14])
                for (ready_14, polls_14) in pl.while_(init_values=(initial_14, 0)):
                    pl.cond(ready_14 < target_14 and polls_14 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_14])
                    (after_14, spin_count_14) = pl.yield_(pl.tensor.read(progress, [flag_14]), polls_14 + 1)
                if after_14 < target_14:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 14)
                    pl.tensor.write(progress, [own_flag + 10], other_14)
                    pl.tensor.write(progress, [own_flag + 11], target_14)
                    pl.tensor.write(progress, [own_flag + 12], after_14)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_14)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_14 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_14)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_15 = 23760 + core_id * 6
    mapped_block_15 = pl.cast(pl.tensor.read(records, [record_15]), pl.INDEX)
    block_15 = pl.max(mapped_block_15, 0)
    if mapped_block_15 >= 0 and failed == 0:
        subslot_15 = pl.cast(pl.tensor.read(records, [record_15 + 1]), pl.INDEX)
        done_15 = pl.tensor.read(records, [record_15 + 2])
        wait_base_15 = pl.cast(pl.tensor.read(records, [record_15 + 3]), pl.INDEX)
        wait_count_15 = pl.cast(pl.tensor.read(records, [record_15 + 4]), pl.INDEX)
        publish_15 = pl.tensor.read(records, [record_15 + 5])
        for dependency_15 in pl.range(wait_count_15):
            if failed == 0:
                wait_index_15 = (wait_base_15 + dependency_15) * 2
                other_15 = pl.cast(pl.tensor.read(waits, [wait_index_15]), pl.INDEX)
                required_15 = pl.tensor.read(waits, [wait_index_15 + 1])
                target_15 = generation_base + required_15
                flag_15 = other_15 * 64
                pl.system.cacheinvalid(progress, [1], [flag_15])
                initial_15 = pl.tensor.read(progress, [flag_15])
                for (ready_15, polls_15) in pl.while_(init_values=(initial_15, 0)):
                    pl.cond(ready_15 < target_15 and polls_15 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_15])
                    (after_15, spin_count_15) = pl.yield_(pl.tensor.read(progress, [flag_15]), polls_15 + 1)
                if after_15 < target_15:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 15)
                    pl.tensor.write(progress, [own_flag + 10], other_15)
                    pl.tensor.write(progress, [own_flag + 11], target_15)
                    pl.tensor.write(progress, [own_flag + 12], after_15)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_15)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_15 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_15)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_16 = 24192 + core_id * 6
    mapped_block_16 = pl.cast(pl.tensor.read(records, [record_16]), pl.INDEX)
    block_16 = pl.max(mapped_block_16, 0)
    if mapped_block_16 >= 0 and failed == 0:
        subslot_16 = pl.cast(pl.tensor.read(records, [record_16 + 1]), pl.INDEX)
        done_16 = pl.tensor.read(records, [record_16 + 2])
        wait_base_16 = pl.cast(pl.tensor.read(records, [record_16 + 3]), pl.INDEX)
        wait_count_16 = pl.cast(pl.tensor.read(records, [record_16 + 4]), pl.INDEX)
        publish_16 = pl.tensor.read(records, [record_16 + 5])
        for dependency_16 in pl.range(wait_count_16):
            if failed == 0:
                wait_index_16 = (wait_base_16 + dependency_16) * 2
                other_16 = pl.cast(pl.tensor.read(waits, [wait_index_16]), pl.INDEX)
                required_16 = pl.tensor.read(waits, [wait_index_16 + 1])
                target_16 = generation_base + required_16
                flag_16 = other_16 * 64
                pl.system.cacheinvalid(progress, [1], [flag_16])
                initial_16 = pl.tensor.read(progress, [flag_16])
                for (ready_16, polls_16) in pl.while_(init_values=(initial_16, 0)):
                    pl.cond(ready_16 < target_16 and polls_16 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_16])
                    (after_16, spin_count_16) = pl.yield_(pl.tensor.read(progress, [flag_16]), polls_16 + 1)
                if after_16 < target_16:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 16)
                    pl.tensor.write(progress, [own_flag + 10], other_16)
                    pl.tensor.write(progress, [own_flag + 11], target_16)
                    pl.tensor.write(progress, [own_flag + 12], after_16)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_16)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_16 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_16)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_17 = 24624 + core_id * 6
    mapped_block_17 = pl.cast(pl.tensor.read(records, [record_17]), pl.INDEX)
    block_17 = pl.max(mapped_block_17, 0)
    if mapped_block_17 >= 0 and failed == 0:
        subslot_17 = pl.cast(pl.tensor.read(records, [record_17 + 1]), pl.INDEX)
        done_17 = pl.tensor.read(records, [record_17 + 2])
        wait_base_17 = pl.cast(pl.tensor.read(records, [record_17 + 3]), pl.INDEX)
        wait_count_17 = pl.cast(pl.tensor.read(records, [record_17 + 4]), pl.INDEX)
        publish_17 = pl.tensor.read(records, [record_17 + 5])
        for dependency_17 in pl.range(wait_count_17):
            if failed == 0:
                wait_index_17 = (wait_base_17 + dependency_17) * 2
                other_17 = pl.cast(pl.tensor.read(waits, [wait_index_17]), pl.INDEX)
                required_17 = pl.tensor.read(waits, [wait_index_17 + 1])
                target_17 = generation_base + required_17
                flag_17 = other_17 * 64
                pl.system.cacheinvalid(progress, [1], [flag_17])
                initial_17 = pl.tensor.read(progress, [flag_17])
                for (ready_17, polls_17) in pl.while_(init_values=(initial_17, 0)):
                    pl.cond(ready_17 < target_17 and polls_17 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_17])
                    (after_17, spin_count_17) = pl.yield_(pl.tensor.read(progress, [flag_17]), polls_17 + 1)
                if after_17 < target_17:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 17)
                    pl.tensor.write(progress, [own_flag + 10], other_17)
                    pl.tensor.write(progress, [own_flag + 11], target_17)
                    pl.tensor.write(progress, [own_flag + 12], after_17)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_17)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_17 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_17)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_18 = 25056 + core_id * 6
    mapped_block_18 = pl.cast(pl.tensor.read(records, [record_18]), pl.INDEX)
    block_18 = pl.max(mapped_block_18, 0)
    if mapped_block_18 >= 0 and failed == 0:
        subslot_18 = pl.cast(pl.tensor.read(records, [record_18 + 1]), pl.INDEX)
        done_18 = pl.tensor.read(records, [record_18 + 2])
        wait_base_18 = pl.cast(pl.tensor.read(records, [record_18 + 3]), pl.INDEX)
        wait_count_18 = pl.cast(pl.tensor.read(records, [record_18 + 4]), pl.INDEX)
        publish_18 = pl.tensor.read(records, [record_18 + 5])
        for dependency_18 in pl.range(wait_count_18):
            if failed == 0:
                wait_index_18 = (wait_base_18 + dependency_18) * 2
                other_18 = pl.cast(pl.tensor.read(waits, [wait_index_18]), pl.INDEX)
                required_18 = pl.tensor.read(waits, [wait_index_18 + 1])
                target_18 = generation_base + required_18
                flag_18 = other_18 * 64
                pl.system.cacheinvalid(progress, [1], [flag_18])
                initial_18 = pl.tensor.read(progress, [flag_18])
                for (ready_18, polls_18) in pl.while_(init_values=(initial_18, 0)):
                    pl.cond(ready_18 < target_18 and polls_18 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_18])
                    (after_18, spin_count_18) = pl.yield_(pl.tensor.read(progress, [flag_18]), polls_18 + 1)
                if after_18 < target_18:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 18)
                    pl.tensor.write(progress, [own_flag + 10], other_18)
                    pl.tensor.write(progress, [own_flag + 11], target_18)
                    pl.tensor.write(progress, [own_flag + 12], after_18)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_18)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_18 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_18)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_19 = 25488 + core_id * 6
    mapped_block_19 = pl.cast(pl.tensor.read(records, [record_19]), pl.INDEX)
    block_19 = pl.max(mapped_block_19, 0)
    if mapped_block_19 >= 0 and failed == 0:
        subslot_19 = pl.cast(pl.tensor.read(records, [record_19 + 1]), pl.INDEX)
        done_19 = pl.tensor.read(records, [record_19 + 2])
        wait_base_19 = pl.cast(pl.tensor.read(records, [record_19 + 3]), pl.INDEX)
        wait_count_19 = pl.cast(pl.tensor.read(records, [record_19 + 4]), pl.INDEX)
        publish_19 = pl.tensor.read(records, [record_19 + 5])
        for dependency_19 in pl.range(wait_count_19):
            if failed == 0:
                wait_index_19 = (wait_base_19 + dependency_19) * 2
                other_19 = pl.cast(pl.tensor.read(waits, [wait_index_19]), pl.INDEX)
                required_19 = pl.tensor.read(waits, [wait_index_19 + 1])
                target_19 = generation_base + required_19
                flag_19 = other_19 * 64
                pl.system.cacheinvalid(progress, [1], [flag_19])
                initial_19 = pl.tensor.read(progress, [flag_19])
                for (ready_19, polls_19) in pl.while_(init_values=(initial_19, 0)):
                    pl.cond(ready_19 < target_19 and polls_19 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_19])
                    (after_19, spin_count_19) = pl.yield_(pl.tensor.read(progress, [flag_19]), polls_19 + 1)
                if after_19 < target_19:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 19)
                    pl.tensor.write(progress, [own_flag + 10], other_19)
                    pl.tensor.write(progress, [own_flag + 11], target_19)
                    pl.tensor.write(progress, [own_flag + 12], after_19)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_19)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_19 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_19)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_20 = 25920 + core_id * 6
    mapped_block_20 = pl.cast(pl.tensor.read(records, [record_20]), pl.INDEX)
    block_20 = pl.max(mapped_block_20, 0)
    if mapped_block_20 >= 0 and failed == 0:
        subslot_20 = pl.cast(pl.tensor.read(records, [record_20 + 1]), pl.INDEX)
        done_20 = pl.tensor.read(records, [record_20 + 2])
        wait_base_20 = pl.cast(pl.tensor.read(records, [record_20 + 3]), pl.INDEX)
        wait_count_20 = pl.cast(pl.tensor.read(records, [record_20 + 4]), pl.INDEX)
        publish_20 = pl.tensor.read(records, [record_20 + 5])
        for dependency_20 in pl.range(wait_count_20):
            if failed == 0:
                wait_index_20 = (wait_base_20 + dependency_20) * 2
                other_20 = pl.cast(pl.tensor.read(waits, [wait_index_20]), pl.INDEX)
                required_20 = pl.tensor.read(waits, [wait_index_20 + 1])
                target_20 = generation_base + required_20
                flag_20 = other_20 * 64
                pl.system.cacheinvalid(progress, [1], [flag_20])
                initial_20 = pl.tensor.read(progress, [flag_20])
                for (ready_20, polls_20) in pl.while_(init_values=(initial_20, 0)):
                    pl.cond(ready_20 < target_20 and polls_20 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_20])
                    (after_20, spin_count_20) = pl.yield_(pl.tensor.read(progress, [flag_20]), polls_20 + 1)
                if after_20 < target_20:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 20)
                    pl.tensor.write(progress, [own_flag + 10], other_20)
                    pl.tensor.write(progress, [own_flag + 11], target_20)
                    pl.tensor.write(progress, [own_flag + 12], after_20)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_20)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_20 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_20)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_21 = 26352 + core_id * 6
    mapped_block_21 = pl.cast(pl.tensor.read(records, [record_21]), pl.INDEX)
    block_21 = pl.max(mapped_block_21, 0)
    if mapped_block_21 >= 0 and failed == 0:
        subslot_21 = pl.cast(pl.tensor.read(records, [record_21 + 1]), pl.INDEX)
        done_21 = pl.tensor.read(records, [record_21 + 2])
        wait_base_21 = pl.cast(pl.tensor.read(records, [record_21 + 3]), pl.INDEX)
        wait_count_21 = pl.cast(pl.tensor.read(records, [record_21 + 4]), pl.INDEX)
        publish_21 = pl.tensor.read(records, [record_21 + 5])
        for dependency_21 in pl.range(wait_count_21):
            if failed == 0:
                wait_index_21 = (wait_base_21 + dependency_21) * 2
                other_21 = pl.cast(pl.tensor.read(waits, [wait_index_21]), pl.INDEX)
                required_21 = pl.tensor.read(waits, [wait_index_21 + 1])
                target_21 = generation_base + required_21
                flag_21 = other_21 * 64
                pl.system.cacheinvalid(progress, [1], [flag_21])
                initial_21 = pl.tensor.read(progress, [flag_21])
                for (ready_21, polls_21) in pl.while_(init_values=(initial_21, 0)):
                    pl.cond(ready_21 < target_21 and polls_21 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_21])
                    (after_21, spin_count_21) = pl.yield_(pl.tensor.read(progress, [flag_21]), polls_21 + 1)
                if after_21 < target_21:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 21)
                    pl.tensor.write(progress, [own_flag + 10], other_21)
                    pl.tensor.write(progress, [own_flag + 11], target_21)
                    pl.tensor.write(progress, [own_flag + 12], after_21)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_21)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_21 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_21)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_22 = 26784 + core_id * 6
    mapped_block_22 = pl.cast(pl.tensor.read(records, [record_22]), pl.INDEX)
    block_22 = pl.max(mapped_block_22, 0)
    if mapped_block_22 >= 0 and failed == 0:
        subslot_22 = pl.cast(pl.tensor.read(records, [record_22 + 1]), pl.INDEX)
        done_22 = pl.tensor.read(records, [record_22 + 2])
        wait_base_22 = pl.cast(pl.tensor.read(records, [record_22 + 3]), pl.INDEX)
        wait_count_22 = pl.cast(pl.tensor.read(records, [record_22 + 4]), pl.INDEX)
        publish_22 = pl.tensor.read(records, [record_22 + 5])
        for dependency_22 in pl.range(wait_count_22):
            if failed == 0:
                wait_index_22 = (wait_base_22 + dependency_22) * 2
                other_22 = pl.cast(pl.tensor.read(waits, [wait_index_22]), pl.INDEX)
                required_22 = pl.tensor.read(waits, [wait_index_22 + 1])
                target_22 = generation_base + required_22
                flag_22 = other_22 * 64
                pl.system.cacheinvalid(progress, [1], [flag_22])
                initial_22 = pl.tensor.read(progress, [flag_22])
                for (ready_22, polls_22) in pl.while_(init_values=(initial_22, 0)):
                    pl.cond(ready_22 < target_22 and polls_22 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_22])
                    (after_22, spin_count_22) = pl.yield_(pl.tensor.read(progress, [flag_22]), polls_22 + 1)
                if after_22 < target_22:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 22)
                    pl.tensor.write(progress, [own_flag + 10], other_22)
                    pl.tensor.write(progress, [own_flag + 11], target_22)
                    pl.tensor.write(progress, [own_flag + 12], after_22)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_22)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_22 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_22)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_23 = 27216 + core_id * 6
    mapped_block_23 = pl.cast(pl.tensor.read(records, [record_23]), pl.INDEX)
    block_23 = pl.max(mapped_block_23, 0)
    if mapped_block_23 >= 0 and failed == 0:
        subslot_23 = pl.cast(pl.tensor.read(records, [record_23 + 1]), pl.INDEX)
        done_23 = pl.tensor.read(records, [record_23 + 2])
        wait_base_23 = pl.cast(pl.tensor.read(records, [record_23 + 3]), pl.INDEX)
        wait_count_23 = pl.cast(pl.tensor.read(records, [record_23 + 4]), pl.INDEX)
        publish_23 = pl.tensor.read(records, [record_23 + 5])
        for dependency_23 in pl.range(wait_count_23):
            if failed == 0:
                wait_index_23 = (wait_base_23 + dependency_23) * 2
                other_23 = pl.cast(pl.tensor.read(waits, [wait_index_23]), pl.INDEX)
                required_23 = pl.tensor.read(waits, [wait_index_23 + 1])
                target_23 = generation_base + required_23
                flag_23 = other_23 * 64
                pl.system.cacheinvalid(progress, [1], [flag_23])
                initial_23 = pl.tensor.read(progress, [flag_23])
                for (ready_23, polls_23) in pl.while_(init_values=(initial_23, 0)):
                    pl.cond(ready_23 < target_23 and polls_23 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_23])
                    (after_23, spin_count_23) = pl.yield_(pl.tensor.read(progress, [flag_23]), polls_23 + 1)
                if after_23 < target_23:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 23)
                    pl.tensor.write(progress, [own_flag + 10], other_23)
                    pl.tensor.write(progress, [own_flag + 11], target_23)
                    pl.tensor.write(progress, [own_flag + 12], after_23)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_23)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_23 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_23)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_24 = 27648 + core_id * 6
    mapped_block_24 = pl.cast(pl.tensor.read(records, [record_24]), pl.INDEX)
    block_24 = pl.max(mapped_block_24, 0)
    if mapped_block_24 >= 0 and failed == 0:
        subslot_24 = pl.cast(pl.tensor.read(records, [record_24 + 1]), pl.INDEX)
        done_24 = pl.tensor.read(records, [record_24 + 2])
        wait_base_24 = pl.cast(pl.tensor.read(records, [record_24 + 3]), pl.INDEX)
        wait_count_24 = pl.cast(pl.tensor.read(records, [record_24 + 4]), pl.INDEX)
        publish_24 = pl.tensor.read(records, [record_24 + 5])
        for dependency_24 in pl.range(wait_count_24):
            if failed == 0:
                wait_index_24 = (wait_base_24 + dependency_24) * 2
                other_24 = pl.cast(pl.tensor.read(waits, [wait_index_24]), pl.INDEX)
                required_24 = pl.tensor.read(waits, [wait_index_24 + 1])
                target_24 = generation_base + required_24
                flag_24 = other_24 * 64
                pl.system.cacheinvalid(progress, [1], [flag_24])
                initial_24 = pl.tensor.read(progress, [flag_24])
                for (ready_24, polls_24) in pl.while_(init_values=(initial_24, 0)):
                    pl.cond(ready_24 < target_24 and polls_24 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_24])
                    (after_24, spin_count_24) = pl.yield_(pl.tensor.read(progress, [flag_24]), polls_24 + 1)
                if after_24 < target_24:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 24)
                    pl.tensor.write(progress, [own_flag + 10], other_24)
                    pl.tensor.write(progress, [own_flag + 11], target_24)
                    pl.tensor.write(progress, [own_flag + 12], after_24)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_24)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_24 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_24)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_25 = 28080 + core_id * 6
    mapped_block_25 = pl.cast(pl.tensor.read(records, [record_25]), pl.INDEX)
    block_25 = pl.max(mapped_block_25, 0)
    if mapped_block_25 >= 0 and failed == 0:
        subslot_25 = pl.cast(pl.tensor.read(records, [record_25 + 1]), pl.INDEX)
        done_25 = pl.tensor.read(records, [record_25 + 2])
        wait_base_25 = pl.cast(pl.tensor.read(records, [record_25 + 3]), pl.INDEX)
        wait_count_25 = pl.cast(pl.tensor.read(records, [record_25 + 4]), pl.INDEX)
        publish_25 = pl.tensor.read(records, [record_25 + 5])
        for dependency_25 in pl.range(wait_count_25):
            if failed == 0:
                wait_index_25 = (wait_base_25 + dependency_25) * 2
                other_25 = pl.cast(pl.tensor.read(waits, [wait_index_25]), pl.INDEX)
                required_25 = pl.tensor.read(waits, [wait_index_25 + 1])
                target_25 = generation_base + required_25
                flag_25 = other_25 * 64
                pl.system.cacheinvalid(progress, [1], [flag_25])
                initial_25 = pl.tensor.read(progress, [flag_25])
                for (ready_25, polls_25) in pl.while_(init_values=(initial_25, 0)):
                    pl.cond(ready_25 < target_25 and polls_25 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_25])
                    (after_25, spin_count_25) = pl.yield_(pl.tensor.read(progress, [flag_25]), polls_25 + 1)
                if after_25 < target_25:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 25)
                    pl.tensor.write(progress, [own_flag + 10], other_25)
                    pl.tensor.write(progress, [own_flag + 11], target_25)
                    pl.tensor.write(progress, [own_flag + 12], after_25)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_25)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_25 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_25)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_26 = 28512 + core_id * 6
    mapped_block_26 = pl.cast(pl.tensor.read(records, [record_26]), pl.INDEX)
    block_26 = pl.max(mapped_block_26, 0)
    if mapped_block_26 >= 0 and failed == 0:
        subslot_26 = pl.cast(pl.tensor.read(records, [record_26 + 1]), pl.INDEX)
        done_26 = pl.tensor.read(records, [record_26 + 2])
        wait_base_26 = pl.cast(pl.tensor.read(records, [record_26 + 3]), pl.INDEX)
        wait_count_26 = pl.cast(pl.tensor.read(records, [record_26 + 4]), pl.INDEX)
        publish_26 = pl.tensor.read(records, [record_26 + 5])
        for dependency_26 in pl.range(wait_count_26):
            if failed == 0:
                wait_index_26 = (wait_base_26 + dependency_26) * 2
                other_26 = pl.cast(pl.tensor.read(waits, [wait_index_26]), pl.INDEX)
                required_26 = pl.tensor.read(waits, [wait_index_26 + 1])
                target_26 = generation_base + required_26
                flag_26 = other_26 * 64
                pl.system.cacheinvalid(progress, [1], [flag_26])
                initial_26 = pl.tensor.read(progress, [flag_26])
                for (ready_26, polls_26) in pl.while_(init_values=(initial_26, 0)):
                    pl.cond(ready_26 < target_26 and polls_26 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_26])
                    (after_26, spin_count_26) = pl.yield_(pl.tensor.read(progress, [flag_26]), polls_26 + 1)
                if after_26 < target_26:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 26)
                    pl.tensor.write(progress, [own_flag + 10], other_26)
                    pl.tensor.write(progress, [own_flag + 11], target_26)
                    pl.tensor.write(progress, [own_flag + 12], after_26)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_26)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_26 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_26)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_27 = 28944 + core_id * 6
    mapped_block_27 = pl.cast(pl.tensor.read(records, [record_27]), pl.INDEX)
    block_27 = pl.max(mapped_block_27, 0)
    if mapped_block_27 >= 0 and failed == 0:
        subslot_27 = pl.cast(pl.tensor.read(records, [record_27 + 1]), pl.INDEX)
        done_27 = pl.tensor.read(records, [record_27 + 2])
        wait_base_27 = pl.cast(pl.tensor.read(records, [record_27 + 3]), pl.INDEX)
        wait_count_27 = pl.cast(pl.tensor.read(records, [record_27 + 4]), pl.INDEX)
        publish_27 = pl.tensor.read(records, [record_27 + 5])
        for dependency_27 in pl.range(wait_count_27):
            if failed == 0:
                wait_index_27 = (wait_base_27 + dependency_27) * 2
                other_27 = pl.cast(pl.tensor.read(waits, [wait_index_27]), pl.INDEX)
                required_27 = pl.tensor.read(waits, [wait_index_27 + 1])
                target_27 = generation_base + required_27
                flag_27 = other_27 * 64
                pl.system.cacheinvalid(progress, [1], [flag_27])
                initial_27 = pl.tensor.read(progress, [flag_27])
                for (ready_27, polls_27) in pl.while_(init_values=(initial_27, 0)):
                    pl.cond(ready_27 < target_27 and polls_27 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_27])
                    (after_27, spin_count_27) = pl.yield_(pl.tensor.read(progress, [flag_27]), polls_27 + 1)
                if after_27 < target_27:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 27)
                    pl.tensor.write(progress, [own_flag + 10], other_27)
                    pl.tensor.write(progress, [own_flag + 11], target_27)
                    pl.tensor.write(progress, [own_flag + 12], after_27)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_27)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_27 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_27)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_28 = 29376 + core_id * 6
    mapped_block_28 = pl.cast(pl.tensor.read(records, [record_28]), pl.INDEX)
    block_28 = pl.max(mapped_block_28, 0)
    if mapped_block_28 >= 0 and failed == 0:
        subslot_28 = pl.cast(pl.tensor.read(records, [record_28 + 1]), pl.INDEX)
        done_28 = pl.tensor.read(records, [record_28 + 2])
        wait_base_28 = pl.cast(pl.tensor.read(records, [record_28 + 3]), pl.INDEX)
        wait_count_28 = pl.cast(pl.tensor.read(records, [record_28 + 4]), pl.INDEX)
        publish_28 = pl.tensor.read(records, [record_28 + 5])
        for dependency_28 in pl.range(wait_count_28):
            if failed == 0:
                wait_index_28 = (wait_base_28 + dependency_28) * 2
                other_28 = pl.cast(pl.tensor.read(waits, [wait_index_28]), pl.INDEX)
                required_28 = pl.tensor.read(waits, [wait_index_28 + 1])
                target_28 = generation_base + required_28
                flag_28 = other_28 * 64
                pl.system.cacheinvalid(progress, [1], [flag_28])
                initial_28 = pl.tensor.read(progress, [flag_28])
                for (ready_28, polls_28) in pl.while_(init_values=(initial_28, 0)):
                    pl.cond(ready_28 < target_28 and polls_28 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_28])
                    (after_28, spin_count_28) = pl.yield_(pl.tensor.read(progress, [flag_28]), polls_28 + 1)
                if after_28 < target_28:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 28)
                    pl.tensor.write(progress, [own_flag + 10], other_28)
                    pl.tensor.write(progress, [own_flag + 11], target_28)
                    pl.tensor.write(progress, [own_flag + 12], after_28)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_28)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_28 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_28)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_29 = 29808 + core_id * 6
    mapped_block_29 = pl.cast(pl.tensor.read(records, [record_29]), pl.INDEX)
    block_29 = pl.max(mapped_block_29, 0)
    if mapped_block_29 >= 0 and failed == 0:
        subslot_29 = pl.cast(pl.tensor.read(records, [record_29 + 1]), pl.INDEX)
        done_29 = pl.tensor.read(records, [record_29 + 2])
        wait_base_29 = pl.cast(pl.tensor.read(records, [record_29 + 3]), pl.INDEX)
        wait_count_29 = pl.cast(pl.tensor.read(records, [record_29 + 4]), pl.INDEX)
        publish_29 = pl.tensor.read(records, [record_29 + 5])
        for dependency_29 in pl.range(wait_count_29):
            if failed == 0:
                wait_index_29 = (wait_base_29 + dependency_29) * 2
                other_29 = pl.cast(pl.tensor.read(waits, [wait_index_29]), pl.INDEX)
                required_29 = pl.tensor.read(waits, [wait_index_29 + 1])
                target_29 = generation_base + required_29
                flag_29 = other_29 * 64
                pl.system.cacheinvalid(progress, [1], [flag_29])
                initial_29 = pl.tensor.read(progress, [flag_29])
                for (ready_29, polls_29) in pl.while_(init_values=(initial_29, 0)):
                    pl.cond(ready_29 < target_29 and polls_29 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_29])
                    (after_29, spin_count_29) = pl.yield_(pl.tensor.read(progress, [flag_29]), polls_29 + 1)
                if after_29 < target_29:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 29)
                    pl.tensor.write(progress, [own_flag + 10], other_29)
                    pl.tensor.write(progress, [own_flag + 11], target_29)
                    pl.tensor.write(progress, [own_flag + 12], after_29)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_29)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_29 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_29)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_30 = 30240 + core_id * 6
    mapped_block_30 = pl.cast(pl.tensor.read(records, [record_30]), pl.INDEX)
    block_30 = pl.max(mapped_block_30, 0)
    if mapped_block_30 >= 0 and failed == 0:
        subslot_30 = pl.cast(pl.tensor.read(records, [record_30 + 1]), pl.INDEX)
        done_30 = pl.tensor.read(records, [record_30 + 2])
        wait_base_30 = pl.cast(pl.tensor.read(records, [record_30 + 3]), pl.INDEX)
        wait_count_30 = pl.cast(pl.tensor.read(records, [record_30 + 4]), pl.INDEX)
        publish_30 = pl.tensor.read(records, [record_30 + 5])
        for dependency_30 in pl.range(wait_count_30):
            if failed == 0:
                wait_index_30 = (wait_base_30 + dependency_30) * 2
                other_30 = pl.cast(pl.tensor.read(waits, [wait_index_30]), pl.INDEX)
                required_30 = pl.tensor.read(waits, [wait_index_30 + 1])
                target_30 = generation_base + required_30
                flag_30 = other_30 * 64
                pl.system.cacheinvalid(progress, [1], [flag_30])
                initial_30 = pl.tensor.read(progress, [flag_30])
                for (ready_30, polls_30) in pl.while_(init_values=(initial_30, 0)):
                    pl.cond(ready_30 < target_30 and polls_30 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_30])
                    (after_30, spin_count_30) = pl.yield_(pl.tensor.read(progress, [flag_30]), polls_30 + 1)
                if after_30 < target_30:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 30)
                    pl.tensor.write(progress, [own_flag + 10], other_30)
                    pl.tensor.write(progress, [own_flag + 11], target_30)
                    pl.tensor.write(progress, [own_flag + 12], after_30)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_30)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_30 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_30)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_31 = 30672 + core_id * 6
    mapped_block_31 = pl.cast(pl.tensor.read(records, [record_31]), pl.INDEX)
    block_31 = pl.max(mapped_block_31, 0)
    if mapped_block_31 >= 0 and failed == 0:
        subslot_31 = pl.cast(pl.tensor.read(records, [record_31 + 1]), pl.INDEX)
        done_31 = pl.tensor.read(records, [record_31 + 2])
        wait_base_31 = pl.cast(pl.tensor.read(records, [record_31 + 3]), pl.INDEX)
        wait_count_31 = pl.cast(pl.tensor.read(records, [record_31 + 4]), pl.INDEX)
        publish_31 = pl.tensor.read(records, [record_31 + 5])
        for dependency_31 in pl.range(wait_count_31):
            if failed == 0:
                wait_index_31 = (wait_base_31 + dependency_31) * 2
                other_31 = pl.cast(pl.tensor.read(waits, [wait_index_31]), pl.INDEX)
                required_31 = pl.tensor.read(waits, [wait_index_31 + 1])
                target_31 = generation_base + required_31
                flag_31 = other_31 * 64
                pl.system.cacheinvalid(progress, [1], [flag_31])
                initial_31 = pl.tensor.read(progress, [flag_31])
                for (ready_31, polls_31) in pl.while_(init_values=(initial_31, 0)):
                    pl.cond(ready_31 < target_31 and polls_31 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_31])
                    (after_31, spin_count_31) = pl.yield_(pl.tensor.read(progress, [flag_31]), polls_31 + 1)
                if after_31 < target_31:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 31)
                    pl.tensor.write(progress, [own_flag + 10], other_31)
                    pl.tensor.write(progress, [own_flag + 11], target_31)
                    pl.tensor.write(progress, [own_flag + 12], after_31)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_31)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_31 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_31)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_32 = 31104 + core_id * 6
    mapped_block_32 = pl.cast(pl.tensor.read(records, [record_32]), pl.INDEX)
    block_32 = pl.max(mapped_block_32, 0)
    if mapped_block_32 >= 0 and failed == 0:
        subslot_32 = pl.cast(pl.tensor.read(records, [record_32 + 1]), pl.INDEX)
        done_32 = pl.tensor.read(records, [record_32 + 2])
        wait_base_32 = pl.cast(pl.tensor.read(records, [record_32 + 3]), pl.INDEX)
        wait_count_32 = pl.cast(pl.tensor.read(records, [record_32 + 4]), pl.INDEX)
        publish_32 = pl.tensor.read(records, [record_32 + 5])
        for dependency_32 in pl.range(wait_count_32):
            if failed == 0:
                wait_index_32 = (wait_base_32 + dependency_32) * 2
                other_32 = pl.cast(pl.tensor.read(waits, [wait_index_32]), pl.INDEX)
                required_32 = pl.tensor.read(waits, [wait_index_32 + 1])
                target_32 = generation_base + required_32
                flag_32 = other_32 * 64
                pl.system.cacheinvalid(progress, [1], [flag_32])
                initial_32 = pl.tensor.read(progress, [flag_32])
                for (ready_32, polls_32) in pl.while_(init_values=(initial_32, 0)):
                    pl.cond(ready_32 < target_32 and polls_32 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_32])
                    (after_32, spin_count_32) = pl.yield_(pl.tensor.read(progress, [flag_32]), polls_32 + 1)
                if after_32 < target_32:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 32)
                    pl.tensor.write(progress, [own_flag + 10], other_32)
                    pl.tensor.write(progress, [own_flag + 11], target_32)
                    pl.tensor.write(progress, [own_flag + 12], after_32)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_32)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_32 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_32)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_33 = 31536 + core_id * 6
    mapped_block_33 = pl.cast(pl.tensor.read(records, [record_33]), pl.INDEX)
    block_33 = pl.max(mapped_block_33, 0)
    if mapped_block_33 >= 0 and failed == 0:
        subslot_33 = pl.cast(pl.tensor.read(records, [record_33 + 1]), pl.INDEX)
        done_33 = pl.tensor.read(records, [record_33 + 2])
        wait_base_33 = pl.cast(pl.tensor.read(records, [record_33 + 3]), pl.INDEX)
        wait_count_33 = pl.cast(pl.tensor.read(records, [record_33 + 4]), pl.INDEX)
        publish_33 = pl.tensor.read(records, [record_33 + 5])
        for dependency_33 in pl.range(wait_count_33):
            if failed == 0:
                wait_index_33 = (wait_base_33 + dependency_33) * 2
                other_33 = pl.cast(pl.tensor.read(waits, [wait_index_33]), pl.INDEX)
                required_33 = pl.tensor.read(waits, [wait_index_33 + 1])
                target_33 = generation_base + required_33
                flag_33 = other_33 * 64
                pl.system.cacheinvalid(progress, [1], [flag_33])
                initial_33 = pl.tensor.read(progress, [flag_33])
                for (ready_33, polls_33) in pl.while_(init_values=(initial_33, 0)):
                    pl.cond(ready_33 < target_33 and polls_33 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_33])
                    (after_33, spin_count_33) = pl.yield_(pl.tensor.read(progress, [flag_33]), polls_33 + 1)
                if after_33 < target_33:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 33)
                    pl.tensor.write(progress, [own_flag + 10], other_33)
                    pl.tensor.write(progress, [own_flag + 11], target_33)
                    pl.tensor.write(progress, [own_flag + 12], after_33)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_33)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_33 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_33)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_34 = 31968 + core_id * 6
    mapped_block_34 = pl.cast(pl.tensor.read(records, [record_34]), pl.INDEX)
    block_34 = pl.max(mapped_block_34, 0)
    if mapped_block_34 >= 0 and failed == 0:
        subslot_34 = pl.cast(pl.tensor.read(records, [record_34 + 1]), pl.INDEX)
        done_34 = pl.tensor.read(records, [record_34 + 2])
        wait_base_34 = pl.cast(pl.tensor.read(records, [record_34 + 3]), pl.INDEX)
        wait_count_34 = pl.cast(pl.tensor.read(records, [record_34 + 4]), pl.INDEX)
        publish_34 = pl.tensor.read(records, [record_34 + 5])
        for dependency_34 in pl.range(wait_count_34):
            if failed == 0:
                wait_index_34 = (wait_base_34 + dependency_34) * 2
                other_34 = pl.cast(pl.tensor.read(waits, [wait_index_34]), pl.INDEX)
                required_34 = pl.tensor.read(waits, [wait_index_34 + 1])
                target_34 = generation_base + required_34
                flag_34 = other_34 * 64
                pl.system.cacheinvalid(progress, [1], [flag_34])
                initial_34 = pl.tensor.read(progress, [flag_34])
                for (ready_34, polls_34) in pl.while_(init_values=(initial_34, 0)):
                    pl.cond(ready_34 < target_34 and polls_34 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_34])
                    (after_34, spin_count_34) = pl.yield_(pl.tensor.read(progress, [flag_34]), polls_34 + 1)
                if after_34 < target_34:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 34)
                    pl.tensor.write(progress, [own_flag + 10], other_34)
                    pl.tensor.write(progress, [own_flag + 11], target_34)
                    pl.tensor.write(progress, [own_flag + 12], after_34)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_34)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_34 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_34)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_35 = 32400 + core_id * 6
    mapped_block_35 = pl.cast(pl.tensor.read(records, [record_35]), pl.INDEX)
    block_35 = pl.max(mapped_block_35, 0)
    if mapped_block_35 >= 0 and failed == 0:
        subslot_35 = pl.cast(pl.tensor.read(records, [record_35 + 1]), pl.INDEX)
        done_35 = pl.tensor.read(records, [record_35 + 2])
        wait_base_35 = pl.cast(pl.tensor.read(records, [record_35 + 3]), pl.INDEX)
        wait_count_35 = pl.cast(pl.tensor.read(records, [record_35 + 4]), pl.INDEX)
        publish_35 = pl.tensor.read(records, [record_35 + 5])
        for dependency_35 in pl.range(wait_count_35):
            if failed == 0:
                wait_index_35 = (wait_base_35 + dependency_35) * 2
                other_35 = pl.cast(pl.tensor.read(waits, [wait_index_35]), pl.INDEX)
                required_35 = pl.tensor.read(waits, [wait_index_35 + 1])
                target_35 = generation_base + required_35
                flag_35 = other_35 * 64
                pl.system.cacheinvalid(progress, [1], [flag_35])
                initial_35 = pl.tensor.read(progress, [flag_35])
                for (ready_35, polls_35) in pl.while_(init_values=(initial_35, 0)):
                    pl.cond(ready_35 < target_35 and polls_35 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_35])
                    (after_35, spin_count_35) = pl.yield_(pl.tensor.read(progress, [flag_35]), polls_35 + 1)
                if after_35 < target_35:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 35)
                    pl.tensor.write(progress, [own_flag + 10], other_35)
                    pl.tensor.write(progress, [own_flag + 11], target_35)
                    pl.tensor.write(progress, [own_flag + 12], after_35)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_35)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_35 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_35)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_36 = 32832 + core_id * 6
    mapped_block_36 = pl.cast(pl.tensor.read(records, [record_36]), pl.INDEX)
    block_36 = pl.max(mapped_block_36, 0)
    if mapped_block_36 >= 0 and failed == 0:
        subslot_36 = pl.cast(pl.tensor.read(records, [record_36 + 1]), pl.INDEX)
        done_36 = pl.tensor.read(records, [record_36 + 2])
        wait_base_36 = pl.cast(pl.tensor.read(records, [record_36 + 3]), pl.INDEX)
        wait_count_36 = pl.cast(pl.tensor.read(records, [record_36 + 4]), pl.INDEX)
        publish_36 = pl.tensor.read(records, [record_36 + 5])
        for dependency_36 in pl.range(wait_count_36):
            if failed == 0:
                wait_index_36 = (wait_base_36 + dependency_36) * 2
                other_36 = pl.cast(pl.tensor.read(waits, [wait_index_36]), pl.INDEX)
                required_36 = pl.tensor.read(waits, [wait_index_36 + 1])
                target_36 = generation_base + required_36
                flag_36 = other_36 * 64
                pl.system.cacheinvalid(progress, [1], [flag_36])
                initial_36 = pl.tensor.read(progress, [flag_36])
                for (ready_36, polls_36) in pl.while_(init_values=(initial_36, 0)):
                    pl.cond(ready_36 < target_36 and polls_36 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_36])
                    (after_36, spin_count_36) = pl.yield_(pl.tensor.read(progress, [flag_36]), polls_36 + 1)
                if after_36 < target_36:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 36)
                    pl.tensor.write(progress, [own_flag + 10], other_36)
                    pl.tensor.write(progress, [own_flag + 11], target_36)
                    pl.tensor.write(progress, [own_flag + 12], after_36)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_36)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_36 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_36)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_37 = 33264 + core_id * 6
    mapped_block_37 = pl.cast(pl.tensor.read(records, [record_37]), pl.INDEX)
    block_37 = pl.max(mapped_block_37, 0)
    if mapped_block_37 >= 0 and failed == 0:
        subslot_37 = pl.cast(pl.tensor.read(records, [record_37 + 1]), pl.INDEX)
        done_37 = pl.tensor.read(records, [record_37 + 2])
        wait_base_37 = pl.cast(pl.tensor.read(records, [record_37 + 3]), pl.INDEX)
        wait_count_37 = pl.cast(pl.tensor.read(records, [record_37 + 4]), pl.INDEX)
        publish_37 = pl.tensor.read(records, [record_37 + 5])
        for dependency_37 in pl.range(wait_count_37):
            if failed == 0:
                wait_index_37 = (wait_base_37 + dependency_37) * 2
                other_37 = pl.cast(pl.tensor.read(waits, [wait_index_37]), pl.INDEX)
                required_37 = pl.tensor.read(waits, [wait_index_37 + 1])
                target_37 = generation_base + required_37
                flag_37 = other_37 * 64
                pl.system.cacheinvalid(progress, [1], [flag_37])
                initial_37 = pl.tensor.read(progress, [flag_37])
                for (ready_37, polls_37) in pl.while_(init_values=(initial_37, 0)):
                    pl.cond(ready_37 < target_37 and polls_37 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_37])
                    (after_37, spin_count_37) = pl.yield_(pl.tensor.read(progress, [flag_37]), polls_37 + 1)
                if after_37 < target_37:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 37)
                    pl.tensor.write(progress, [own_flag + 10], other_37)
                    pl.tensor.write(progress, [own_flag + 11], target_37)
                    pl.tensor.write(progress, [own_flag + 12], after_37)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_37)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_37 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_37)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_38 = 33696 + core_id * 6
    mapped_block_38 = pl.cast(pl.tensor.read(records, [record_38]), pl.INDEX)
    block_38 = pl.max(mapped_block_38, 0)
    if mapped_block_38 >= 0 and failed == 0:
        subslot_38 = pl.cast(pl.tensor.read(records, [record_38 + 1]), pl.INDEX)
        done_38 = pl.tensor.read(records, [record_38 + 2])
        wait_base_38 = pl.cast(pl.tensor.read(records, [record_38 + 3]), pl.INDEX)
        wait_count_38 = pl.cast(pl.tensor.read(records, [record_38 + 4]), pl.INDEX)
        publish_38 = pl.tensor.read(records, [record_38 + 5])
        for dependency_38 in pl.range(wait_count_38):
            if failed == 0:
                wait_index_38 = (wait_base_38 + dependency_38) * 2
                other_38 = pl.cast(pl.tensor.read(waits, [wait_index_38]), pl.INDEX)
                required_38 = pl.tensor.read(waits, [wait_index_38 + 1])
                target_38 = generation_base + required_38
                flag_38 = other_38 * 64
                pl.system.cacheinvalid(progress, [1], [flag_38])
                initial_38 = pl.tensor.read(progress, [flag_38])
                for (ready_38, polls_38) in pl.while_(init_values=(initial_38, 0)):
                    pl.cond(ready_38 < target_38 and polls_38 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_38])
                    (after_38, spin_count_38) = pl.yield_(pl.tensor.read(progress, [flag_38]), polls_38 + 1)
                if after_38 < target_38:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 38)
                    pl.tensor.write(progress, [own_flag + 10], other_38)
                    pl.tensor.write(progress, [own_flag + 11], target_38)
                    pl.tensor.write(progress, [own_flag + 12], after_38)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_38)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_38 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_38)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_39 = 34128 + core_id * 6
    mapped_block_39 = pl.cast(pl.tensor.read(records, [record_39]), pl.INDEX)
    block_39 = pl.max(mapped_block_39, 0)
    if mapped_block_39 >= 0 and failed == 0:
        subslot_39 = pl.cast(pl.tensor.read(records, [record_39 + 1]), pl.INDEX)
        done_39 = pl.tensor.read(records, [record_39 + 2])
        wait_base_39 = pl.cast(pl.tensor.read(records, [record_39 + 3]), pl.INDEX)
        wait_count_39 = pl.cast(pl.tensor.read(records, [record_39 + 4]), pl.INDEX)
        publish_39 = pl.tensor.read(records, [record_39 + 5])
        for dependency_39 in pl.range(wait_count_39):
            if failed == 0:
                wait_index_39 = (wait_base_39 + dependency_39) * 2
                other_39 = pl.cast(pl.tensor.read(waits, [wait_index_39]), pl.INDEX)
                required_39 = pl.tensor.read(waits, [wait_index_39 + 1])
                target_39 = generation_base + required_39
                flag_39 = other_39 * 64
                pl.system.cacheinvalid(progress, [1], [flag_39])
                initial_39 = pl.tensor.read(progress, [flag_39])
                for (ready_39, polls_39) in pl.while_(init_values=(initial_39, 0)):
                    pl.cond(ready_39 < target_39 and polls_39 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_39])
                    (after_39, spin_count_39) = pl.yield_(pl.tensor.read(progress, [flag_39]), polls_39 + 1)
                if after_39 < target_39:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 39)
                    pl.tensor.write(progress, [own_flag + 10], other_39)
                    pl.tensor.write(progress, [own_flag + 11], target_39)
                    pl.tensor.write(progress, [own_flag + 12], after_39)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_39)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_39 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_39)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_40 = 34560 + core_id * 6
    mapped_block_40 = pl.cast(pl.tensor.read(records, [record_40]), pl.INDEX)
    block_40 = pl.max(mapped_block_40, 0)
    if mapped_block_40 >= 0 and failed == 0:
        subslot_40 = pl.cast(pl.tensor.read(records, [record_40 + 1]), pl.INDEX)
        done_40 = pl.tensor.read(records, [record_40 + 2])
        wait_base_40 = pl.cast(pl.tensor.read(records, [record_40 + 3]), pl.INDEX)
        wait_count_40 = pl.cast(pl.tensor.read(records, [record_40 + 4]), pl.INDEX)
        publish_40 = pl.tensor.read(records, [record_40 + 5])
        for dependency_40 in pl.range(wait_count_40):
            if failed == 0:
                wait_index_40 = (wait_base_40 + dependency_40) * 2
                other_40 = pl.cast(pl.tensor.read(waits, [wait_index_40]), pl.INDEX)
                required_40 = pl.tensor.read(waits, [wait_index_40 + 1])
                target_40 = generation_base + required_40
                flag_40 = other_40 * 64
                pl.system.cacheinvalid(progress, [1], [flag_40])
                initial_40 = pl.tensor.read(progress, [flag_40])
                for (ready_40, polls_40) in pl.while_(init_values=(initial_40, 0)):
                    pl.cond(ready_40 < target_40 and polls_40 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_40])
                    (after_40, spin_count_40) = pl.yield_(pl.tensor.read(progress, [flag_40]), polls_40 + 1)
                if after_40 < target_40:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 40)
                    pl.tensor.write(progress, [own_flag + 10], other_40)
                    pl.tensor.write(progress, [own_flag + 11], target_40)
                    pl.tensor.write(progress, [own_flag + 12], after_40)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_40)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_40 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_40)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_41 = 34992 + core_id * 6
    mapped_block_41 = pl.cast(pl.tensor.read(records, [record_41]), pl.INDEX)
    block_41 = pl.max(mapped_block_41, 0)
    if mapped_block_41 >= 0 and failed == 0:
        subslot_41 = pl.cast(pl.tensor.read(records, [record_41 + 1]), pl.INDEX)
        done_41 = pl.tensor.read(records, [record_41 + 2])
        wait_base_41 = pl.cast(pl.tensor.read(records, [record_41 + 3]), pl.INDEX)
        wait_count_41 = pl.cast(pl.tensor.read(records, [record_41 + 4]), pl.INDEX)
        publish_41 = pl.tensor.read(records, [record_41 + 5])
        for dependency_41 in pl.range(wait_count_41):
            if failed == 0:
                wait_index_41 = (wait_base_41 + dependency_41) * 2
                other_41 = pl.cast(pl.tensor.read(waits, [wait_index_41]), pl.INDEX)
                required_41 = pl.tensor.read(waits, [wait_index_41 + 1])
                target_41 = generation_base + required_41
                flag_41 = other_41 * 64
                pl.system.cacheinvalid(progress, [1], [flag_41])
                initial_41 = pl.tensor.read(progress, [flag_41])
                for (ready_41, polls_41) in pl.while_(init_values=(initial_41, 0)):
                    pl.cond(ready_41 < target_41 and polls_41 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_41])
                    (after_41, spin_count_41) = pl.yield_(pl.tensor.read(progress, [flag_41]), polls_41 + 1)
                if after_41 < target_41:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 41)
                    pl.tensor.write(progress, [own_flag + 10], other_41)
                    pl.tensor.write(progress, [own_flag + 11], target_41)
                    pl.tensor.write(progress, [own_flag + 12], after_41)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_41)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_41 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_41)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_42 = 35424 + core_id * 6
    mapped_block_42 = pl.cast(pl.tensor.read(records, [record_42]), pl.INDEX)
    block_42 = pl.max(mapped_block_42, 0)
    if mapped_block_42 >= 0 and failed == 0:
        subslot_42 = pl.cast(pl.tensor.read(records, [record_42 + 1]), pl.INDEX)
        done_42 = pl.tensor.read(records, [record_42 + 2])
        wait_base_42 = pl.cast(pl.tensor.read(records, [record_42 + 3]), pl.INDEX)
        wait_count_42 = pl.cast(pl.tensor.read(records, [record_42 + 4]), pl.INDEX)
        publish_42 = pl.tensor.read(records, [record_42 + 5])
        for dependency_42 in pl.range(wait_count_42):
            if failed == 0:
                wait_index_42 = (wait_base_42 + dependency_42) * 2
                other_42 = pl.cast(pl.tensor.read(waits, [wait_index_42]), pl.INDEX)
                required_42 = pl.tensor.read(waits, [wait_index_42 + 1])
                target_42 = generation_base + required_42
                flag_42 = other_42 * 64
                pl.system.cacheinvalid(progress, [1], [flag_42])
                initial_42 = pl.tensor.read(progress, [flag_42])
                for (ready_42, polls_42) in pl.while_(init_values=(initial_42, 0)):
                    pl.cond(ready_42 < target_42 and polls_42 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_42])
                    (after_42, spin_count_42) = pl.yield_(pl.tensor.read(progress, [flag_42]), polls_42 + 1)
                if after_42 < target_42:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 42)
                    pl.tensor.write(progress, [own_flag + 10], other_42)
                    pl.tensor.write(progress, [own_flag + 11], target_42)
                    pl.tensor.write(progress, [own_flag + 12], after_42)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_42)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_42 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_42)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    record_43 = 35856 + core_id * 6
    mapped_block_43 = pl.cast(pl.tensor.read(records, [record_43]), pl.INDEX)
    block_43 = pl.max(mapped_block_43, 0)
    if mapped_block_43 >= 0 and failed == 0:
        subslot_43 = pl.cast(pl.tensor.read(records, [record_43 + 1]), pl.INDEX)
        done_43 = pl.tensor.read(records, [record_43 + 2])
        wait_base_43 = pl.cast(pl.tensor.read(records, [record_43 + 3]), pl.INDEX)
        wait_count_43 = pl.cast(pl.tensor.read(records, [record_43 + 4]), pl.INDEX)
        publish_43 = pl.tensor.read(records, [record_43 + 5])
        for dependency_43 in pl.range(wait_count_43):
            if failed == 0:
                wait_index_43 = (wait_base_43 + dependency_43) * 2
                other_43 = pl.cast(pl.tensor.read(waits, [wait_index_43]), pl.INDEX)
                required_43 = pl.tensor.read(waits, [wait_index_43 + 1])
                target_43 = generation_base + required_43
                flag_43 = other_43 * 64
                pl.system.cacheinvalid(progress, [1], [flag_43])
                initial_43 = pl.tensor.read(progress, [flag_43])
                for (ready_43, polls_43) in pl.while_(init_values=(initial_43, 0)):
                    pl.cond(ready_43 < target_43 and polls_43 < 200000)
                    pl.system.cacheinvalid(progress, [1], [flag_43])
                    (after_43, spin_count_43) = pl.yield_(pl.tensor.read(progress, [flag_43]), polls_43 + 1)
                if after_43 < target_43:
                    failed = 1
                    pl.tensor.write(progress, [own_flag + 8], 1)
                    pl.tensor.write(progress, [own_flag + 9], 43)
                    pl.tensor.write(progress, [own_flag + 10], other_43)
                    pl.tensor.write(progress, [own_flag + 11], target_43)
                    pl.tensor.write(progress, [own_flag + 12], after_43)
                    pl.tensor.write(progress, [own_flag + 13], spin_count_43)
                    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
                    pl.system.fence()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        if publish_43 != 0 and failed == 0:
            pl.tensor.write(progress, [own_flag], generation_base + done_43)
            pl.system.cacheinvalid(progress, [1], [own_flag])
            pl.system.fence()
    return progress

@pl.jit.inline
def csa_static_group(progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> pl.Tensor[[4608], pl.INT64]:
    with pl.cluster(name_hint='csa_static_protocol'):
        progress = csa_static_aic(progress, records, waits)
        progress = csa_static_aiv(progress, records, waits)
    return progress

@pl.jit
def entry(progress: pl.InOut[pl.Tensor[[4608], pl.INT64]], records: pl.Tensor[[36288], pl.INT64], waits: pl.Tensor[[33112], pl.INT64]) -> pl.Tensor[[4608], pl.INT64]:
    with pl.spmd(24, sync_start=True):
        progress = csa_static_group(progress, records, waits)
    return progress
