#pragma once
#include "workload.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace lab {

// Inverted index: table[d][v] = all map_id values where map_desc[map_id][d] == v
// zero_table[d] stores all map_id values where map_desc[map_id][d] == 0; keep it for completeness
struct CpuInvIndex {
    std::array<std::array<std::vector<uint32_t>, 16>, kDims> table;
    std::array<std::vector<uint32_t>, kDims> zero_table;
    uint32_t M = 0;
};

struct CpuOne {
    int32_t  best_lane = -1;        // best map id
    uint16_t best_match = 0;        // argmax match score over non-zero query dims
    uint16_t best_mismatch = 0;     // mismatch = (#query_nonzero_dims - best_match)
    uint16_t compared_dims = 0;     // number of non-zero query dims
};

struct CpuResults {
    std::vector<CpuOne> per_query;
};

CpuInvIndex build_inv_index(const Workload& wl);

// Run queries in the ROI: argmax match over non-zero query dims
// Tie-break: choose the smaller map_id when best_match ties
CpuResults run_cpu_inverted_match(const Workload& wl, const CpuInvIndex& idx);

} // namespace lab