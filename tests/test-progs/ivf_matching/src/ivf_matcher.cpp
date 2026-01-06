#include "ivf_matcher.hpp"
#include "layout.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace msim {

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

static inline uint8_t clamp_u8(int v, int lo, int hi)
{
    if (v < lo) return static_cast<uint8_t>(lo);
    if (v > hi) return static_cast<uint8_t>(hi);
    return static_cast<uint8_t>(v);
}

void IvfMatcher::mask_and_inplace(MaskRow& a, const MaskRow& b)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(a[i] & b[i]);
}

void IvfMatcher::mask_or_inplace(MaskRow& a, const MaskRow& b)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(a[i] | b[i]);
}

void IvfMatcher::mask_not_inplace(MaskRow& a)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(~a[i]);
}

bool IvfMatcher::mask_any(const MaskRow& a)
{
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != 0u) return true;
    }
    return false;
}

void IvfMatcher::Planes::clear()
{
    for (auto& p : b) p.fill(0);
}

void IvfMatcher::Planes::add_mask(const MaskRow& x)
{
    MaskRow carry = x;
    for (size_t k = 0; k < b.size(); ++k) {
        MaskRow sum{};
        for (size_t i = 0; i < sum.size(); ++i) sum[i] = static_cast<uint8_t>(b[k][i] ^ carry[i]);

        MaskRow new_carry{};
        for (size_t i = 0; i < new_carry.size(); ++i) new_carry[i] = static_cast<uint8_t>(b[k][i] & carry[i]);

        b[k] = sum;
        carry = new_carry;
    }
}

uint8_t IvfMatcher::Planes::lane_value(int lane) const
{
    uint8_t v = 0;
    for (size_t k = 0; k < b.size(); ++k) {
        const uint8_t bit = lane_get_bit(b[k].data(), lane);
        v = static_cast<uint8_t>(v | static_cast<uint8_t>(bit << k));
    }
    return v;
}

IvfMatcher::IvfMatcher(CimModule& cim, const IvfMapBins& map, const IvfPlacement& place)
    : cim_(cim), map_(map), place_(place)
{
    if (map_.postings.nlist == 0) throw std::runtime_error("IvfMatcher: map postings empty");
    if (place_.bucket.size() != map_.postings.nlist) throw std::runtime_error("IvfMatcher: placement K mismatch");
}

std::vector<uint32_t> IvfMatcher::select_lists_cpu(const uint8_t* qdesc64, uint32_t nprobe) const
{
    const uint32_t K = map_.model.nlist;
    if (K == 0) return {};
    if (nprobe == 0 || nprobe > K) nprobe = K;

    const uint32_t dim = map_.model.dim;
    if (dim == 0 || map_.model.centroids.size() != static_cast<size_t>(K * dim)) {
        throw std::runtime_error("IvfMatcher: invalid centroids");
    }

    // ---- build query augmented vector (best-effort, consistent & deterministic) ----
    // NOTE: This is a pragmatic projection for gem5 experiments.
    // If you have exact ZeroAwareIVF projection, replace here.
    std::vector<float> q(dim, 0.0f);

    const float alpha = map_.model.alpha_value;
    const float beta  = map_.model.beta_presence;
    const bool use_idf = (map_.model.use_idf != 0);

    auto vs = [&](int d) -> float {
        return map_.model.val_scale.empty() ? 1.0f : map_.model.val_scale.at(static_cast<size_t>(d));
    };
    auto iw = [&](int d) -> float {
        if (!use_idf) return 1.0f;
        return map_.model.idf_w.empty() ? 1.0f : map_.model.idf_w.at(static_cast<size_t>(d));
    };

    // common case: aug128 = [value64, presence64]
    if (dim >= 128) {
        for (int d = 0; d < 64; ++d) {
            const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
            const float idf  = iw(d);
            const float val  = alpha * idf * vs(d) * static_cast<float>(qv);
            const float pres = beta  * idf * (qv != 0 ? 1.0f : 0.0f);
            q[static_cast<size_t>(d)]      = val;
            q[static_cast<size_t>(64 + d)] = pres;
        }
    } else {
        // fallback: use first 64 dims as value
        const uint32_t D = std::min<uint32_t>(dim, 64);
        for (uint32_t d = 0; d < D; ++d) q[d] = static_cast<float>(qdesc64[d] & 0x0Fu);
    }

    // ---- compute top-nprobe by L2 ----
    struct Pair { float dist; uint32_t id; };
    std::vector<Pair> best;
    best.reserve(nprobe);

    for (uint32_t k = 0; k < K; ++k) {
        const float* c = &map_.model.centroids[static_cast<size_t>(k) * dim];
        float s = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            const float diff = q[i] - c[i];
            s += diff * diff;
        }

        if (best.size() < nprobe) {
            best.push_back({s, k});
            if (best.size() == nprobe) {
                std::nth_element(best.begin(), best.end() - 1, best.end(),
                                 [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
            }
        } else if (s < best.back().dist) {
            best.back() = {s, k};
            std::nth_element(best.begin(), best.end() - 1, best.end(),
                             [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
        }
    }

    std::sort(best.begin(), best.end(), [](const Pair& a, const Pair& b){ return a.dist < b.dist; });

    std::vector<uint32_t> out;
    out.reserve(best.size());
    for (const auto& p : best) out.push_back(p.id);
    return out;
}

IvfMatcher::MaskRow IvfMatcher::geo_eq_mask_axis(uint32_t bucket_id, uint32_t group_in_bucket, bool is_x, uint8_t v)
{
    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    map_geo_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array, row_base);

    const uint8_t lo = static_cast<uint8_t>(v & 0x0Fu);
    const uint8_t hi = static_cast<uint8_t>((v >> 4) & 0x0Fu);

    auto or_nibble_mismatch = [&](bool high, uint8_t nib) -> MaskRow {
        std::vector<uint16_t> rows;
        rows.reserve(4);

        for (int b = 0; b < 4; ++b) {
            const int qbit = (nib >> b) & 1u;
            const bool inv = (qbit != 0);

            uint16_t r = 0;
            if (is_x) {
                r = static_cast<uint16_t>(row_base + (!high ? geo_x_row_low(b, inv) : geo_x_row_high(b, inv)));
            } else {
                r = static_cast<uint16_t>(row_base + (!high ? geo_y_row_low(b, inv) : geo_y_row_high(b, inv)));
            }
            rows.push_back(r);
        }

        cim_.OR(rows,
                0xffu,
                CimModule::Mask::bank(bank),
                CimModule::Mask::colsAll(),
                0,
                CimModule::Mask::mat(mat),
                CimModule::Mask::array(array));

        MaskRow tmp{};
        cim_.copy_temp_to_cpu(tmp.data(), bank, mat, array, 0, kRowBytes);
        return tmp; // mismatch mask
    };

    MaskRow mis_lo = or_nibble_mismatch(false, lo);
    MaskRow mis_hi = or_nibble_mismatch(true,  hi);

    mask_not_inplace(mis_lo);       // eq_lo
    mask_not_inplace(mis_hi);       // eq_hi
    mask_and_inplace(mis_lo, mis_hi); // eq_byte
    return mis_lo;
}

void IvfMatcher::geo_eq_masks_xy(uint32_t bucket_id, uint32_t group_in_bucket, uint8_t qgx, uint8_t qgy, MaskRow& out_mask_xy)
{
    out_mask_xy.fill(0);

    auto one_xy = [&](uint8_t gx, uint8_t gy) {
        MaskRow mx = geo_eq_mask_axis(bucket_id, group_in_bucket, true,  gx);
        MaskRow my = geo_eq_mask_axis(bucket_id, group_in_bucket, false, gy);
        mask_and_inplace(mx, my);
        mask_or_inplace(out_mask_xy, mx);
    };

    for (int dx = -kGeoProbeRadius; dx <= kGeoProbeRadius; ++dx) {
        const uint8_t gx = clamp_u8(static_cast<int>(qgx) + dx, 0, map_.geo_max);
        one_xy(gx, qgy);
    }
    for (int dy = -kGeoProbeRadius; dy <= kGeoProbeRadius; ++dy) {
        const uint8_t gy = clamp_u8(static_cast<int>(qgy) + dy, 0, map_.geo_max);
        one_xy(qgx, gy);
    }
}

void IvfMatcher::desc_mismatch_planes(uint32_t bucket_id, uint32_t group_in_bucket, const uint8_t* qdesc64, Planes& out_planes)
{
    out_planes.clear();

    uint16_t bank = 0, mat = 0, array = 0;
    map_desc_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array);

    std::vector<uint16_t> rows;
    rows.reserve(kDescBits);

    for (int d = 0; d < kDescDims; ++d) {
        rows.clear();
        const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);

        for (int b = 0; b < kDescBits; ++b) {
            const int qbit = (qv >> b) & 1u;
            const uint16_t r = (qbit != 0) ? desc_inv_row(d, b) : desc_true_row(d, b);
            rows.push_back(r);
        }

        cim_.OR(rows,
                0xffu,
                CimModule::Mask::bank(bank),
                CimModule::Mask::colsAll(),
                0,
                CimModule::Mask::mat(mat),
                CimModule::Mask::array(array));

        MaskRow tmp{};
        cim_.copy_temp_to_cpu(tmp.data(), bank, mat, array, 0, kRowBytes);

        out_planes.add_mask(tmp);
    }
}

MatchResult IvfMatcher::match_one_query_desc(uint8_t qgx, uint8_t qgy, const uint8_t* qdesc64, uint32_t nprobe, uint32_t max_groups_per_list)
{
    MatchResult best;

    const uint32_t K = map_.postings.nlist;
    if (K == 0) return best;

    const std::vector<uint32_t> lists = select_lists_cpu(qdesc64, nprobe);

    for (uint32_t lid : lists) {
        const auto& bp = place_.bucket.at(lid);
        const uint32_t groups_total = bp.desc_groups;
        const uint32_t groups = (max_groups_per_list == 0u) ? groups_total : std::min<uint32_t>(groups_total, max_groups_per_list);

        for (uint32_t g = 0; g < groups; ++g) {
            MaskRow geo_xy{};
            geo_eq_masks_xy(lid, g, qgx, qgy, geo_xy);
            if (!mask_any(geo_xy)) continue;

            Planes planes;
            desc_mismatch_planes(lid, g, qdesc64, planes);

            const uint32_t base_off = bp.posting_off + g * static_cast<uint32_t>(kLanesPerGroup);
            const uint32_t remain   = bp.map_count > g * static_cast<uint32_t>(kLanesPerGroup)
                                    ? (bp.map_count - g * static_cast<uint32_t>(kLanesPerGroup))
                                    : 0u;
            const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

            for (uint32_t lane = 0; lane < valid; ++lane) {
                if (lane_get_bit(geo_xy.data(), static_cast<int>(lane)) == 0u) continue;

                const int32_t mid = map_.postings.map_ids.at(static_cast<size_t>(base_off + lane));
                const uint32_t mis = planes.lane_value(static_cast<int>(lane));

                if (mis < best.best_mismatch || (mis == best.best_mismatch && mid < best.best_map_id)) {
                    best.best_mismatch = static_cast<uint8_t>(mis);
                    best.best_map_id = mid;
                }
            }
        }
    }

    return best;
}

} // namespace msim