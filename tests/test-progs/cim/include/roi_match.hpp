#pragma once
#include <cstdint>
#include <vector>

#include "workload.hpp"
#include "nvm_writer.hpp"

class CimModule;

namespace lab {

struct RoiPerQuery {
    int64_t best_global = -1;      // best candidate index in [0, M)
    uint32_t best_mismatch = 0xFFFFFFFFu;
};

struct RoiResults {
    std::vector<RoiPerQuery> per_query;
};

// Full scan over all groups (512 candidates per group), with geo gating from bank_pos,
// then masked argmin over mismatch counter planes.
RoiResults run_roi_match(CimModule& cim, const Workload& wl, const Placement& place);

} // namespace lab
