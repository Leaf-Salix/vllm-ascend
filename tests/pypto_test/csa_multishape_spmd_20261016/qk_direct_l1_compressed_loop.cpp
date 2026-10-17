// 真实NPU C++摘录，仅证据，不单独编译。
for (int64_t v3867 = v354; v3867 < v333; v3867 += v318) {
              // pto: %2171, %2172, %2168, %2169, %2170, %2167
              ;
              int32_t v3868 = (v105)[(int64_t) ((uint64_t) ((int64_t) (uint64_t) v3801 * (uint64_t) v332) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v3800 - (uint64_t) v318) * (uint64_t) v333) + (uint64_t) v3867))];
              // pto: %2173
              ;
              int64_t v3869 = (int64_t) v3868;
              // pto: %2174
              ;
              if (v3869 >= v354) {
                // pto: %2176
                ;
                Tile<TileType::Mat, bfloat16_t, 384, 512, BLayout::ColMajor, 1, 512, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v3870;
                // pto: %2175, %2176
                ;
                uint64_t v3871 = (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v3816 + (uint64_t) v3867) * (uint64_t) v205) + (uint64_t) v294);
                TASSIGN(v3870, v3871);
                // pto: %2177
                ;
                int64_t v3872 = v3869 < v354 ? v354 : v3869;
                // pto: %view_61__ssa_v0_pview
                ;
                __gm__ bfloat16_t* v3873 = PTOAS__GLOBAL_TENSOR_DATA(v554);
                // pto: %view_61__ssa_v0_pview
                ;
                const int64_t v3874 = 0;
                // pto: %view_61__ssa_v0_pview
                ;
                const int64_t v3875 = 512;
                // pto: %view_61__ssa_v0_pview
                ;
                pto::Shape<1, 1, 1, 1, 512> v3876 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %view_61__ssa_v0_pview
                ;
                pto::Stride<512, 512, 512, 512, 1> v3877 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %view_61__ssa_v0_pview
                ;
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v3878 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v3873 + (v3874 + v3872 * v3875), v3876, v3877);
                TLOAD(v3870, v3878);
              } else {
                // pto: %2179
                ;
                Tile<TileType::Mat, bfloat16_t, 384, 512, BLayout::ColMajor, 1, 512, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v3879;
                // pto: %2178, %2179
                ;
                uint64_t v3880 = (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v3816 + (uint64_t) v3867) * (uint64_t) v205) + (uint64_t) v294);
                TASSIGN(v3879, v3880);
                // pto: %2180
                ;
                int64_t v3881 = (int64_t) ((uint64_t) v3771 * (uint64_t) v321);
                // pto: %2181
                ;
                int64_t v3882 = v3881 < v354 ? v354 : v3881;
                // pto: %2182
                ;
                __gm__ bfloat16_t* v3883 = PTOAS__GLOBAL_TENSOR_DATA(v714);
                // pto: %2182
                ;
                const int64_t v3884 = 0;
                // pto: %2182
                ;
                const int64_t v3885 = 512;
                // pto: %2182
                ;
                pto::Shape<1, 1, 1, 1, 512> v3886 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %2182
                ;
                pto::Stride<512, 512, 512, 512, 1> v3887 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %2182
                ;
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v3888 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v3883 + (v3884 + v3882 * v3885), v3886, v3887);
                TLOAD(v3879, v3888);
              };
            }
