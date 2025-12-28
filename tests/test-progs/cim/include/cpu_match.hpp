#pragma once
#include "workload.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace lab {

// 倒排表：table[d][v] = 所有 map_id 使得 map_desc[map_id][d] == v
// 另外 zero_table[d] = 所有 map_id 使得 map_desc[map_id][d] == 0 （目前 match 不用它，但你要求要保留）
struct CpuInvIndex {
    std::array<std::array<std::vector<uint32_t>, 16>, kDims> table;
    std::array<std::vector<uint32_t>, kDims> zero_table;
    uint32_t M = 0;
};

struct CpuOne {
    int32_t  best_lane = -1;        // best map id
    uint16_t best_match = 0;        // argmax match 分數（只看 query 非 0 dims）
    uint16_t best_mismatch = 0;     // mismatch = (#query_nonzero_dims - best_match)
    uint16_t compared_dims = 0;     // query 非 0 的維度數
};

struct CpuResults {
    std::vector<CpuOne> per_query;
};

CpuInvIndex build_inv_index(const Workload& wl);

// ROI 內跑 query：argmax match（只看 query 非 0）
// tie-break：best_match 相同時選 map_id 最小者
CpuResults run_cpu_inverted_match(const Workload& wl, const CpuInvIndex& idx);

} // namespace lab