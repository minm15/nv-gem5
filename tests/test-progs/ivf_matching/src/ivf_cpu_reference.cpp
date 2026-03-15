#include "ivf_cpu_reference.hpp"

#include "msim_config.hpp"

#include <algorithm>
#include <stdexcept>

namespace msim {

namespace {

static inline uint8_t clamp_u8(int value, int lo, int hi)
{
    if (value < lo) return static_cast<uint8_t>(lo);
    if (value > hi) return static_cast<uint8_t>(hi);
    return static_cast<uint8_t>(value);
}

static std::vector<uint32_t> select_lists_cpu_reference(const IvfMapBins& map,
                                                        const uint8_t* qdesc64,
                                                        uint32_t nprobe)
{
    const uint32_t K = map.model.nlist;
    if (K == 0u) return {};
    if (nprobe == 0u || nprobe > K) nprobe = K;

    const uint32_t dim = map.model.dim;
    if (dim == 0u || map.model.centroids.size() != static_cast<size_t>(K) * dim) {
        throw std::runtime_error("select_lists_cpu_reference: invalid centroids");
    }

    std::vector<float> q(dim, 0.0f);
    const float alpha = map.model.alpha_value;
    const float beta = map.model.beta_presence;
    const bool use_idf = (map.model.use_idf != 0);

    auto val_scale = [&](int d) -> float {
        return map.model.val_scale.empty() ? 1.0f : map.model.val_scale.at(static_cast<size_t>(d));
    };
    auto idf_weight = [&](int d) -> float {
        if (!use_idf) return 1.0f;
        return map.model.idf_w.empty() ? 1.0f : map.model.idf_w.at(static_cast<size_t>(d));
    };

    if (dim >= 128u) {
        for (int d = 0; d < 64; ++d) {
            const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
            const float idf = idf_weight(d);
            q[static_cast<size_t>(d)] = alpha * idf * val_scale(d) * static_cast<float>(qv);
            q[static_cast<size_t>(64 + d)] = beta * idf * (qv != 0u ? 1.0f : 0.0f);
        }
    } else {
        const uint32_t D = std::min<uint32_t>(dim, 64u);
        for (uint32_t d = 0; d < D; ++d) {
            q[d] = static_cast<float>(qdesc64[d] & 0x0Fu);
        }
    }

    struct Pair {
        float dist;
        uint32_t id;
    };

    std::vector<Pair> best;
    best.reserve(nprobe);

    for (uint32_t k = 0; k < K; ++k) {
        const float* centroid = &map.model.centroids[static_cast<size_t>(k) * dim];
        float dist = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            const float diff = q[i] - centroid[i];
            dist += diff * diff;
        }

        if (best.size() < nprobe) {
            best.push_back({dist, k});
            if (best.size() == nprobe) {
                std::nth_element(best.begin(), best.end() - 1, best.end(),
                                 [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }
        } else if (dist < best.back().dist) {
            best.back() = {dist, k};
            std::nth_element(best.begin(), best.end() - 1, best.end(),
                             [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
        }
    }

    std::sort(best.begin(), best.end(),
              [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

    std::vector<uint32_t> out;
    out.reserve(best.size());
    for (const auto& item : best) out.push_back(item.id);
    return out;
}

static bool geo_matches_cross(uint8_t map_gx, uint8_t map_gy,
                              uint8_t qgx, uint8_t qgy,
                              uint8_t geo_max)
{
    if (map_gy == qgy) {
        for (int dx = -kGeoProbeRadius; dx <= kGeoProbeRadius; ++dx) {
            if (map_gx == clamp_u8(static_cast<int>(qgx) + dx, 0, geo_max)) return true;
        }
    }

    if (map_gx == qgx) {
        for (int dy = -kGeoProbeRadius; dy <= kGeoProbeRadius; ++dy) {
            if (map_gy == clamp_u8(static_cast<int>(qgy) + dy, 0, geo_max)) return true;
        }
    }

    return false;
}

static uint8_t descriptor_mismatch(const uint8_t* map_desc64, const uint8_t* qdesc64)
{
    uint8_t mismatch = 0u;
    for (size_t d = 0; d < 64u; ++d) {
        const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
        if (qv == 0u) continue;
        const uint8_t mv = static_cast<uint8_t>(map_desc64[d] & 0x0Fu);
        mismatch = static_cast<uint8_t>(mismatch + (mv == qv ? 0u : 1u));
    }
    return mismatch;
}

} // namespace

MatchResult match_one_query_desc_cpu_reference(const IvfMapBins& map,
                                               uint8_t qgx,
                                               uint8_t qgy,
                                               const uint8_t* qdesc64,
                                               uint32_t nprobe,
                                               uint32_t max_groups_per_list)
{
    const auto results = match_one_step_cpu_reference(map, qgx, qgy, qdesc64, 1u, nprobe, max_groups_per_list);
    return results.empty() ? MatchResult{} : results.front();
}

std::vector<MatchResult> match_one_step_cpu_reference(const IvfMapBins& map,
                                                      uint8_t qgx,
                                                      uint8_t qgy,
                                                      const uint8_t* desc_ptr,
                                                      size_t m,
                                                      uint32_t nprobe,
                                                      uint32_t max_groups_per_list)
{
    std::vector<MatchResult> results(m);
    if (m == 0u) return results;

    for (size_t qi = 0; qi < m; ++qi) {
        const uint8_t* qdesc64 = desc_ptr + qi * 64u;
        const std::vector<uint32_t> lists = select_lists_cpu_reference(map, qdesc64, nprobe);

        for (uint32_t lid : lists) {
            const uint32_t off0 = map.postings.offsets.at(static_cast<size_t>(lid));
            const uint32_t off1 = map.postings.offsets.at(static_cast<size_t>(lid + 1u));
            const uint32_t map_count = off1 - off0;
            const uint32_t groups_total =
                (map_count + static_cast<uint32_t>(kLanesPerGroup) - 1u) /
                static_cast<uint32_t>(kLanesPerGroup);
            const uint32_t groups = (max_groups_per_list == 0u)
                ? groups_total
                : std::min<uint32_t>(groups_total, max_groups_per_list);

            for (uint32_t g = 0; g < groups; ++g) {
                const uint32_t base_off = off0 + g * static_cast<uint32_t>(kLanesPerGroup);
                const uint32_t remain = map_count > g * static_cast<uint32_t>(kLanesPerGroup)
                    ? (map_count - g * static_cast<uint32_t>(kLanesPerGroup))
                    : 0u;
                const uint32_t valid = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

                for (uint32_t lane = 0; lane < valid; ++lane) {
                    const int32_t map_id = map.postings.map_ids.at(static_cast<size_t>(base_off + lane));
                    const size_t geo_off = static_cast<size_t>(map_id) * 2u;

                    const uint8_t map_gx = map.geo_grid.at(geo_off + 0u);
                    const uint8_t map_gy = map.geo_grid.at(geo_off + 1u);
                    if (!geo_matches_cross(map_gx, map_gy, qgx, qgy, map.geo_max)) continue;

                    const uint8_t mismatch =
                        descriptor_mismatch(&map.desc_q4.at(static_cast<size_t>(map_id) * 64u), qdesc64);

                    auto& best = results[qi];
                    if (mismatch < best.best_mismatch ||
                        (mismatch == best.best_mismatch && map_id < best.best_map_id)) {
                        best.best_mismatch = mismatch;
                        best.best_map_id = map_id;
                    }
                }
            }
        }
    }

    return results;
}

} // namespace msim
