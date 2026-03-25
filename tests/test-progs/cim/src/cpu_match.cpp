#include "cpu_match.hpp"

#include <algorithm>
#include <cstring>

namespace lab {

CpuInvIndex build_inv_index(const Workload& wl)
{
    CpuInvIndex idx;
    idx.M = static_cast<uint32_t>(wl.M);

    // Clear posting lists before filling the index
    for (int d = 0; d < kDims; ++d) {
        for (int v = 0; v < 16; ++v) idx.table[d][v].clear();
        idx.zero_table[d].clear();
    }

    for (uint32_t mi = 0; mi < static_cast<uint32_t>(wl.M); ++mi) {
        for (int d = 0; d < kDims; ++d) {
            const uint8_t mv = static_cast<uint8_t>(wl.map_desc[mi][static_cast<size_t>(d)] & 0x0Fu);
            if (mv == 0) {
                idx.zero_table[d].push_back(mi);
            } else {
                idx.table[d][mv].push_back(mi);
            }
        }
    }
    return idx;
}

CpuResults run_cpu_inverted_match(const Workload& wl, const CpuInvIndex& idx)
{
    CpuResults out;
    out.per_query.resize(static_cast<size_t>(wl.Q));

    const uint32_t M = static_cast<uint32_t>(wl.M);

    // Each query uses a score array: score[map_id]++
    // M=512 is small, so a full max scan is cheap
    std::vector<uint16_t> score(M, 0);

    for (int qi = 0; qi < wl.Q; ++qi) {
        std::fill(score.begin(), score.end(), 0);

        uint16_t compared = 0;

        // Only consider non-zero query dims
        for (int d = 0; d < kDims; ++d) {
            const uint8_t qv = static_cast<uint8_t>(wl.query_desc[static_cast<size_t>(qi)][static_cast<size_t>(d)] & 0x0Fu);
            if (qv == 0) continue;
            ++compared;

            const auto& posting = idx.table[d][qv];
            for (uint32_t mi : posting) {
                // M is small; uint16_t is enough here (compared <= 64)
                score[mi] = static_cast<uint16_t>(score[mi] + 1);
            }
        }

        // Argmax match; ties go to the smaller map_id
        int32_t best_id = -1;
        uint16_t best_match = 0;

        for (uint32_t mi = 0; mi < M; ++mi) {
            const uint16_t s = score[mi];
            if (best_id < 0 || s > best_match || (s == best_match && static_cast<int32_t>(mi) < best_id)) {
                best_id = static_cast<int32_t>(mi);
                best_match = s;
            }
        }

        CpuOne r;
        r.best_lane = best_id;
        r.best_match = best_match;
        r.compared_dims = compared;
        r.best_mismatch = static_cast<uint16_t>((best_id >= 0) ? (compared - best_match) : compared);

        out.per_query[static_cast<size_t>(qi)] = r;
    }

    return out;
}

} // namespace lab