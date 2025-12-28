#include "cpu_match.hpp"

#include <algorithm>
#include <cstring>

namespace lab {

CpuInvIndex build_inv_index(const Workload& wl)
{
    CpuInvIndex idx;
    idx.M = static_cast<uint32_t>(wl.M);

    // reserve（可選）：降低 push_back 擴容
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

    // 每個 query 需要一個計分陣列：score[map_id]++
    // M=512 很小，掃一遍找 max 也不貴
    std::vector<uint16_t> score(M, 0);

    for (int qi = 0; qi < wl.Q; ++qi) {
        std::fill(score.begin(), score.end(), 0);

        uint16_t compared = 0;

        // 只看 query 非 0 維度
        for (int d = 0; d < kDims; ++d) {
            const uint8_t qv = static_cast<uint8_t>(wl.query_desc[static_cast<size_t>(qi)][static_cast<size_t>(d)] & 0x0Fu);
            if (qv == 0) continue;
            ++compared;

            const auto& posting = idx.table[d][qv];
            for (uint32_t mi : posting) {
                // M 很小，uint16_t 足夠（最多加到 compared<=64）
                score[mi] = static_cast<uint16_t>(score[mi] + 1);
            }
        }

        // argmax match，tie -> 最小 map_id
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