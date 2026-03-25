#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(inCIM)
#include "gem5/m5ops.h"
#endif

#include "cim_api.hpp"
#include "ivf_cpu_reference.hpp"
#include "ivf_layout.hpp"
#include "ivf_map_writer.hpp"
#include "ivf_matcher.hpp"
#include "layout.hpp"
#include "map_loader.hpp"
#include "msim_config.hpp"
#include "query_loader.hpp"

namespace {

#if defined(inCIM)
static inline void
begin_roi(uint64_t step_idx)
{
    m5_reset_stats(0, 0);
    m5_work_begin(1, step_idx);
}

static inline void
end_roi(uint64_t step_idx)
{
    m5_work_end(1, step_idx);
    m5_dump_stats(0, 0);
}
#else
static inline void
begin_roi(uint64_t)
{
}

static inline void
end_roi(uint64_t)
{
}
#endif

static std::vector<uint32_t> build_order_centroid_nn(const msim::IvfMapBins& map, uint32_t start = 0u)
{
    const uint32_t K = map.model.nlist;
    const uint32_t dim = map.model.dim;
    if (K == 0u) return {};
    if (map.model.centroids.size() != static_cast<size_t>(K) * dim) {
        throw std::runtime_error("build_order_centroid_nn: invalid centroids");
    }

    start = K == 0u ? 0u : (start % K);

    auto dist2 = [&](uint32_t a, uint32_t b) -> float {
        const float* ca = &map.model.centroids[static_cast<size_t>(a) * dim];
        const float* cb = &map.model.centroids[static_cast<size_t>(b) * dim];
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            const float diff = ca[i] - cb[i];
            sum += diff * diff;
        }
        return sum;
    };

    std::vector<uint32_t> order;
    order.reserve(K);
    std::vector<uint8_t> used(K, 0u);

    uint32_t current = start;
    used[current] = 1u;
    order.push_back(current);

    for (uint32_t t = 1; t < K; ++t) {
        uint32_t best = UINT32_MAX;
        float best_dist = 0.0f;

        for (uint32_t cand = 0; cand < K; ++cand) {
            if (used[cand] != 0u) continue;
            const float dist = dist2(current, cand);
            if (best == UINT32_MAX || dist < best_dist) {
                best = cand;
                best_dist = dist;
            }
        }

        if (best == UINT32_MAX) break;
        used[best] = 1u;
        order.push_back(best);
        current = best;
    }

    for (uint32_t i = 0; i < K; ++i) {
        if (used[i] == 0u) order.push_back(i);
    }

    return order;
}

static msim::IvfMapBins load_ivf_map(const std::string& map_dir)
{
    msim::MapLoader loader;
    loader.load_from_folder(map_dir);
    if (!loader.ivf().enabled) {
        throw std::runtime_error("ivf_verify: map.ivf is not enabled in map.json");
    }

    msim::IvfMapBins map;
    map.N = static_cast<uint32_t>(loader.n());
    map.geo_max = loader.geo().max;
    map.desc_q4 = loader.desc_q4_u8();
    map.geo_grid = loader.geo_grid_u8();

    map.postings.nlist = loader.ivf().nlist;
    map.postings.offsets = loader.ivf().postings_offsets_u32;
    map.postings.map_ids = loader.ivf().postings_map_ids_i32;

    map.model.nlist = loader.ivf().nlist;
    map.model.dim = loader.ivf().dim;
    map.model.centroids = loader.ivf().centroids_f32;
    map.model.alpha_value = loader.ivf().alpha_value;
    map.model.beta_presence = loader.ivf().beta_presence;
    map.model.use_idf = loader.ivf().use_idf ? 1 : 0;
    map.model.val_scale = loader.ivf().val_scale_f32;
    map.model.idf_w = loader.ivf().idf_w_f32;
    return map;
}

static std::vector<uint32_t> select_lists_cpu_debug(const msim::IvfMapBins& map,
                                                    const uint8_t* qdesc64,
                                                    uint32_t nprobe)
{
    const uint32_t K = map.model.nlist;
    if (K == 0u) return {};
    if (nprobe == 0u || nprobe > K) nprobe = K;

    const uint32_t dim = map.model.dim;
    std::vector<float> q(dim, 0.0f);
    const float alpha = map.model.alpha_value;
    const float beta = map.model.beta_presence;
    const bool use_idf = (map.model.use_idf != 0);

    auto vs = [&](int d) -> float {
        return map.model.val_scale.empty() ? 1.0f : map.model.val_scale.at(static_cast<size_t>(d));
    };
    auto iw = [&](int d) -> float {
        if (!use_idf) return 1.0f;
        return map.model.idf_w.empty() ? 1.0f : map.model.idf_w.at(static_cast<size_t>(d));
    };

    if (dim >= 128u) {
        for (int d = 0; d < 64; ++d) {
            const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
            const float idf = iw(d);
            q[static_cast<size_t>(d)] = alpha * idf * vs(d) * static_cast<float>(qv);
            q[static_cast<size_t>(64 + d)] = beta * idf * (qv != 0u ? 1.0f : 0.0f);
        }
    } else {
        const uint32_t D = std::min<uint32_t>(dim, 64u);
        for (uint32_t d = 0; d < D; ++d) q[d] = static_cast<float>(qdesc64[d] & 0x0Fu);
    }

    struct Pair {
        float dist;
        uint32_t id;
    };

    std::vector<Pair> best;
    best.reserve(nprobe);

    for (uint32_t k = 0; k < K; ++k) {
        const float* c = &map.model.centroids[static_cast<size_t>(k) * dim];
        float s = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            const float diff = q[i] - c[i];
            s += diff * diff;
        }

        if (best.size() < nprobe) {
            best.push_back({s, k});
            if (best.size() == nprobe) {
                std::nth_element(best.begin(), best.end() - 1, best.end(),
                                 [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }
        } else if (s < best.back().dist) {
            best.back() = {s, k};
            std::nth_element(best.begin(), best.end() - 1, best.end(),
                             [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
        }
    }

    std::sort(best.begin(), best.end(),
              [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

    std::vector<uint32_t> out;
    out.reserve(best.size());
    for (const auto& p : best) out.push_back(p.id);
    return out;
}

static uint8_t clamp_u8(int v, int lo, int hi)
{
    if (v < lo) return static_cast<uint8_t>(lo);
    if (v > hi) return static_cast<uint8_t>(hi);
    return static_cast<uint8_t>(v);
}

static bool geo_matches_cross(uint8_t map_gx, uint8_t map_gy,
                              uint8_t qgx, uint8_t qgy,
                              uint8_t geo_max)
{
    if (map_gy == qgy) {
        for (int dx = -msim::kGeoProbeRadius; dx <= msim::kGeoProbeRadius; ++dx) {
            if (map_gx == clamp_u8(static_cast<int>(qgx) + dx, 0, geo_max)) return true;
        }
    }

    if (map_gx == qgx) {
        for (int dy = -msim::kGeoProbeRadius; dy <= msim::kGeoProbeRadius; ++dy) {
            if (map_gy == clamp_u8(static_cast<int>(qgy) + dy, 0, geo_max)) return true;
        }
    }

    return false;
}

static uint8_t descriptor_mismatch(const msim::IvfMapBins& map,
                                   int32_t map_id,
                                   const uint8_t* qdesc64)
{
    uint8_t mismatch = 0u;
    const uint8_t* mdesc = &map.desc_q4.at(static_cast<size_t>(map_id) * 64u);
    for (size_t d = 0; d < 64u; ++d) {
        const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
        if (qv == 0u) continue;
        const uint8_t mv = static_cast<uint8_t>(mdesc[d] & 0x0Fu);
        mismatch = static_cast<uint8_t>(mismatch + (mv == qv ? 0u : 1u));
    }
    return mismatch;
}

using MaskRow = std::array<uint8_t, msim::kRowBytes>;

static void mask_and_inplace(MaskRow& a, const MaskRow& b)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(a[i] & b[i]);
}

static void mask_or_inplace(MaskRow& a, const MaskRow& b)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(a[i] | b[i]);
}

static void mask_not_inplace(MaskRow& a)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(~a[i]);
}

static void geo_batch_eq_masks_axis_block_debug(
    CimModule& cim,
    const msim::IvfPlacement& place,
    uint32_t bucket_id,
    uint32_t group_in_bucket,
    bool is_x,
    const std::vector<uint8_t>& vals,
    std::vector<MaskRow>& out_eq_masks,
    uint16_t temp_base_row
)
{
    out_eq_masks.clear();
    out_eq_masks.resize(vals.size());
    if (vals.empty()) return;

    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    msim::map_geo_bucket_group_to_region(place, bucket_id, group_in_bucket, bank, mat, array, row_base);

    std::vector<uint16_t> rows;
    rows.reserve(4);

    for (size_t i = 0; i < vals.size(); ++i) {
        const uint8_t v = vals[i];
        const uint8_t lo = static_cast<uint8_t>(v & 0x0Fu);
        const uint8_t hi = static_cast<uint8_t>((v >> 4) & 0x0Fu);

        auto issue_nibble = [&](bool high, uint8_t nib, uint16_t temp_row) {
            rows.clear();
            for (int b = 0; b < 4; ++b) {
                const int qbit = (nib >> b) & 1u;
                const bool inv = (qbit != 0);
                uint16_t r = 0;
                if (is_x) {
                    r = static_cast<uint16_t>(
                        row_base + (!high ? msim::geo_x_row_low(b, inv) : msim::geo_x_row_high(b, inv)));
                } else {
                    r = static_cast<uint16_t>(
                        row_base + (!high ? msim::geo_y_row_low(b, inv) : msim::geo_y_row_high(b, inv)));
                }
                rows.push_back(r);
            }

            cim.OR(rows,
                   0xffu,
                   CimModule::Mask::bank(bank),
                   CimModule::Mask::colsAll(),
                   temp_row,
                   CimModule::Mask::mat(mat),
                   CimModule::Mask::array(array));
        };

        issue_nibble(false, lo, static_cast<uint16_t>(temp_base_row + 2u * i + 0u));
        issue_nibble(true, hi, static_cast<uint16_t>(temp_base_row + 2u * i + 1u));
    }

    std::vector<uint8_t> block(static_cast<size_t>(2u * vals.size()) * msim::kRowBytes, 0u);
    cim.copy_temp_block_to_cpu(block.data(), bank, mat, array, temp_base_row, static_cast<size_t>(2u * vals.size()));

    for (size_t i = 0; i < vals.size(); ++i) {
        const uint8_t* mis_lo = block.data() + static_cast<size_t>(2u * i + 0u) * msim::kRowBytes;
        const uint8_t* mis_hi = block.data() + static_cast<size_t>(2u * i + 1u) * msim::kRowBytes;
        MaskRow eq{};
        for (size_t j = 0; j < eq.size(); ++j) {
            eq[j] = static_cast<uint8_t>(static_cast<uint8_t>(~mis_lo[j]) & static_cast<uint8_t>(~mis_hi[j]));
        }
        out_eq_masks[i] = eq;
    }
}

static MaskRow compute_geo_mask_xy_debug(CimModule& cim,
                                         const msim::IvfMapBins& map,
                                         const msim::IvfPlacement& place,
                                         uint32_t lid,
                                         uint32_t group_in_bucket,
                                         uint8_t qgx,
                                         uint8_t qgy)
{
    static constexpr uint16_t kGeoTempBase = 0;
    MaskRow out{};
    out.fill(0);

    std::vector<MaskRow> eqs;
    geo_batch_eq_masks_axis_block_debug(cim, place, lid, group_in_bucket,
                                        false, {qgy}, eqs, kGeoTempBase);
    MaskRow my_fixed = eqs[0];

    geo_batch_eq_masks_axis_block_debug(cim, place, lid, group_in_bucket,
                                        true, {qgx}, eqs, kGeoTempBase);
    MaskRow mx_fixed = eqs[0];

    std::vector<uint8_t> gx_vals;
    for (int dx = -msim::kGeoProbeRadius; dx <= msim::kGeoProbeRadius; ++dx) {
        gx_vals.push_back(clamp_u8(static_cast<int>(qgx) + dx, 0, map.geo_max));
    }
    geo_batch_eq_masks_axis_block_debug(cim, place, lid, group_in_bucket,
                                        true, gx_vals, eqs, kGeoTempBase);
    for (const auto& mx : eqs) {
        MaskRow tmp = mx;
        mask_and_inplace(tmp, my_fixed);
        mask_or_inplace(out, tmp);
    }

    std::vector<uint8_t> gy_vals;
    for (int dy = -msim::kGeoProbeRadius; dy <= msim::kGeoProbeRadius; ++dy) {
        gy_vals.push_back(clamp_u8(static_cast<int>(qgy) + dy, 0, map.geo_max));
    }
    geo_batch_eq_masks_axis_block_debug(cim, place, lid, group_in_bucket,
                                        false, gy_vals, eqs, kGeoTempBase);
    for (const auto& my : eqs) {
        MaskRow tmp = my;
        mask_and_inplace(tmp, mx_fixed);
        mask_or_inplace(out, tmp);
    }

    return out;
}

static int32_t list_of_map_id(const msim::IvfMapBins& map, int32_t map_id)
{
    for (uint32_t lid = 0; lid < map.postings.nlist; ++lid) {
        const uint32_t off0 = map.postings.offsets.at(static_cast<size_t>(lid));
        const uint32_t off1 = map.postings.offsets.at(static_cast<size_t>(lid + 1u));
        for (uint32_t off = off0; off < off1; ++off) {
            if (map.postings.map_ids.at(static_cast<size_t>(off)) == map_id) {
                return static_cast<int32_t>(lid);
            }
        }
    }
    return -1;
}

static void dump_row_desc(const msim::IvfMapBins& map, int32_t map_id, size_t begin, size_t end)
{
    std::cout << "    desc[" << begin << ":" << end << "]=";
    const uint8_t* mdesc = &map.desc_q4.at(static_cast<size_t>(map_id) * 64u);
    for (size_t d = begin; d < end; ++d) {
        if (d != begin) std::cout << " ";
        std::cout << static_cast<int>(mdesc[d]);
    }
    std::cout << "\n";
}

static void dump_cim_geo_mask_debug(CimModule& cim,
                                    const msim::IvfMapBins& map,
                                    const msim::IvfPlacement& place,
                                    uint32_t lid,
                                    uint8_t qgx,
                                    uint8_t qgy)
{
    const auto& bp = place.bucket.at(lid);
    const uint32_t groups = bp.desc_groups;
    for (uint32_t g = 0; g < groups; ++g) {
        const MaskRow mask = compute_geo_mask_xy_debug(cim, map, place, lid, g, qgx, qgy);
        const uint32_t base_off = bp.posting_off + g * static_cast<uint32_t>(msim::kLanesPerGroup);
        const uint32_t remain = bp.map_count > g * static_cast<uint32_t>(msim::kLanesPerGroup)
            ? (bp.map_count - g * static_cast<uint32_t>(msim::kLanesPerGroup))
            : 0u;
        const uint32_t valid = std::min<uint32_t>(static_cast<uint32_t>(msim::kLanesPerGroup), remain);

        std::cout << "[debug] cim_geo_mask lid=" << lid << " group=" << g << " set_lanes=";
        bool any = false;
        for (uint32_t lane = 0; lane < valid; ++lane) {
            if (msim::lane_get_bit(mask.data(), static_cast<int>(lane)) == 0u) continue;
            const int32_t mid = map.postings.map_ids.at(static_cast<size_t>(base_off + lane));
            if (any) std::cout << ",";
            any = true;
            std::cout << lane << "(map=" << mid << ")";
        }
        if (!any) std::cout << "none";
        std::cout << "\n";
    }
}

static void dump_mismatch_debug(CimModule& cim,
                                const msim::IvfMapBins& map,
                                const msim::IvfPlacement& place,
                                const msim::QueryStepView& step,
                                size_t step_idx,
                                size_t qi,
                                const uint8_t* qdesc64,
                                uint32_t nprobe,
                                uint32_t max_groups_per_list,
                                const msim::MatchResult& cim_r,
                                const msim::MatchResult& cpu_r)
{
    const std::vector<uint32_t> lists = select_lists_cpu_debug(map, qdesc64, nprobe);
    std::cout << "[debug] step=" << step_idx
              << " query_in_step=" << qi
              << " gx=" << static_cast<int>(step.gx)
              << " gy=" << static_cast<int>(step.gy)
              << " selected_lists=";
    for (size_t i = 0; i < lists.size(); ++i) {
        if (i != 0u) std::cout << ",";
        std::cout << lists[i];
    }
    std::cout << "\n";

    std::cout << "[debug] query desc[0:16]=";
    for (size_t d = 0; d < 16u; ++d) {
        if (d != 0u) std::cout << " ";
        std::cout << static_cast<int>(qdesc64[d]);
    }
    std::cout << " desc[32:48]=";
    for (size_t d = 32u; d < 48u; ++d) {
        if (d != 32u) std::cout << " ";
        std::cout << static_cast<int>(qdesc64[d]);
    }
    std::cout << "\n";

    for (const int32_t map_id : {6, 9, cim_r.best_map_id, cpu_r.best_map_id}) {
        if (map_id < 0) continue;
        const int32_t lid = list_of_map_id(map, map_id);
        const size_t geo_off = static_cast<size_t>(map_id) * 2u;
        const uint8_t mgx = map.geo_grid.at(geo_off + 0u);
        const uint8_t mgy = map.geo_grid.at(geo_off + 1u);
        const bool geo_ok = geo_matches_cross(mgx, mgy, step.gx, step.gy, map.geo_max);
        const uint8_t mis = descriptor_mismatch(map, map_id, qdesc64);
        const bool list_selected = std::find(lists.begin(), lists.end(), static_cast<uint32_t>(lid)) != lists.end();

        std::cout << "[debug] map_id=" << map_id
                  << " lid=" << lid
                  << " selected_list=" << (list_selected ? "yes" : "no")
                  << " geo=(" << static_cast<int>(mgx) << "," << static_cast<int>(mgy) << ")"
                  << " geo_ok=" << (geo_ok ? "yes" : "no")
                  << " mismatch=" << static_cast<int>(mis)
                  << "\n";
        dump_row_desc(map, map_id, 0u, 8u);
        dump_row_desc(map, map_id, 32u, 40u);
    }

    for (uint32_t lid : {1u, 2u}) {
        if (lid < map.postings.nlist) {
            dump_cim_geo_mask_debug(cim, map, place, lid, step.gx, step.gy);
        }
    }

    std::cout << "[debug] qualifying candidates in selected lists:\n";
    for (uint32_t lid : lists) {
        const uint32_t off0 = map.postings.offsets.at(static_cast<size_t>(lid));
        const uint32_t off1 = map.postings.offsets.at(static_cast<size_t>(lid + 1u));
        const uint32_t map_count = off1 - off0;
        const uint32_t groups_total =
            (map_count + static_cast<uint32_t>(msim::kLanesPerGroup) - 1u) /
            static_cast<uint32_t>(msim::kLanesPerGroup);
        const uint32_t groups = (max_groups_per_list == 0u)
            ? groups_total
            : std::min<uint32_t>(groups_total, max_groups_per_list);

        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t base_off = off0 + g * static_cast<uint32_t>(msim::kLanesPerGroup);
            const uint32_t remain = map_count > g * static_cast<uint32_t>(msim::kLanesPerGroup)
                ? (map_count - g * static_cast<uint32_t>(msim::kLanesPerGroup))
                : 0u;
            const uint32_t valid = std::min<uint32_t>(static_cast<uint32_t>(msim::kLanesPerGroup), remain);

            for (uint32_t lane = 0; lane < valid; ++lane) {
                const int32_t mid = map.postings.map_ids.at(static_cast<size_t>(base_off + lane));
                const size_t geo_off = static_cast<size_t>(mid) * 2u;
                const uint8_t mgx = map.geo_grid.at(geo_off + 0u);
                const uint8_t mgy = map.geo_grid.at(geo_off + 1u);
                const bool geo_ok = geo_matches_cross(mgx, mgy, step.gx, step.gy, map.geo_max);
                if (!geo_ok) continue;

                const uint8_t mis = descriptor_mismatch(map, mid, qdesc64);
                std::cout << "  lid=" << lid
                          << " group=" << g
                          << " lane=" << lane
                          << " map_id=" << mid
                          << " geo=(" << static_cast<int>(mgx) << "," << static_cast<int>(mgy) << ")"
                          << " mismatch=" << static_cast<int>(mis)
                          << "\n";
            }
        }
    }
}

static uint64_t checksum_results(const std::vector<msim::MatchResult>& results)
{
    uint64_t checksum = 0u;
    for (const auto& r : results) {
        checksum += static_cast<uint64_t>(static_cast<uint32_t>(r.best_map_id));
        checksum += static_cast<uint64_t>(r.best_mismatch);
    }
    return checksum;
}

} // namespace

int main()
{
    static constexpr uint32_t kNProbe = 8u;
    static constexpr uint32_t kMaxGroupsPerList = 50u;

    const std::string map_dir = msim::kMapDir;
    const std::string query_dir = msim::kQueryDir;

    try {
        const msim::IvfMapBins map = load_ivf_map(map_dir);

        msim::QueryLoader query_loader;
        query_loader.load_from_folder(query_dir);

        CimModule cim;
        cim.setGeometry(msim::kBankBits,
                        msim::kMatBits,
                        msim::kArrayBits,
                        msim::kRowBits,
                        msim::kColBits);

        const std::vector<uint32_t> order = build_order_centroid_nn(map, 0u);
        const msim::IvfPlacement place = msim::compute_ivf_placement(map, order);

        msim::IvfMapWriter writer(cim, place);
        writer.write_all(map);

        msim::IvfMatcher matcher(cim, map, place);

        uint64_t total_queries = 0u;
        uint64_t cim_checksum = 0u;
        uint64_t cpu_checksum = 0u;

        for (size_t step_idx = 0; step_idx < query_loader.steps(); ++step_idx) {
            const msim::QueryStepView step = query_loader.step(step_idx);

            begin_roi(static_cast<uint64_t>(step_idx));
            const auto cim_results =
                matcher.match_one_step(step.gx, step.gy, step.desc_ptr, step.m, kNProbe, kMaxGroupsPerList);
            end_roi(static_cast<uint64_t>(step_idx));
            const auto cpu_results =
                msim::match_one_step_cpu_reference(map, step.gx, step.gy, step.desc_ptr, step.m,
                                                   kNProbe, kMaxGroupsPerList);

            if (cim_results.size() != cpu_results.size()) {
                throw std::runtime_error("ivf_verify: result size mismatch");
            }

            for (size_t qi = 0; qi < cim_results.size(); ++qi) {
                const auto& cim_r = cim_results[qi];
                const auto& cpu_r = cpu_results[qi];
                if (cim_r.best_map_id != cpu_r.best_map_id ||
                    cim_r.best_mismatch != cpu_r.best_mismatch) {
                    std::cerr << "[mismatch] step=" << step_idx
                              << " query_in_step=" << qi
                              << " gx=" << static_cast<int>(step.gx)
                              << " gy=" << static_cast<int>(step.gy)
                              << " cim=(" << cim_r.best_map_id << "," << static_cast<int>(cim_r.best_mismatch) << ")"
                              << " cpu=(" << cpu_r.best_map_id << "," << static_cast<int>(cpu_r.best_mismatch) << ")"
                              << "\n";
                    dump_mismatch_debug(cim, map, place, step, step_idx, qi,
                                        step.desc_ptr + qi * 64u,
                                        kNProbe, kMaxGroupsPerList,
                                        cim_r, cpu_r);
                    return 1;
                }
            }

            total_queries += cim_results.size();
            cim_checksum += checksum_results(cim_results);
            cpu_checksum += checksum_results(cpu_results);

            std::cout << "[step] s=" << step_idx
                      << " m=" << step.m
                      << " checksum=" << checksum_results(cim_results)
                      << "\n";
        }

        std::cout << "[pass] total_queries=" << total_queries
                  << " cim_checksum=" << cim_checksum
                  << " cpu_checksum=" << cpu_checksum
                  << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}
