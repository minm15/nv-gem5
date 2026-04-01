#include "ivf_matcher.hpp"
#include "layout.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>

#if defined(inCIM) && defined(IVF_PHASE_PROFILE)
#include <gem5/m5ops.h>
#endif

namespace msim {

namespace {

#if defined(inCIM) && defined(IVF_PHASE_PROFILE)
static inline uint64_t
profileNowNs()
{
    return m5_rpns();
}

struct ScopedPhaseTimer
{
    uint64_t &accum_ns;
    const uint64_t start_ns;

    explicit ScopedPhaseTimer(uint64_t &accum)
        : accum_ns(accum), start_ns(profileNowNs())
    {}

    ~ScopedPhaseTimer()
    {
        accum_ns += (profileNowNs() - start_ns);
    }
};

struct MatchPhaseTotals
{
    uint64_t total_ns = 0;
    uint64_t list_select_ns = 0;
    uint64_t geo_job_build_ns = 0;
    uint64_t geo_mask_compute_ns = 0;
    uint64_t geo_desc_map_ns = 0;
    uint64_t geo_job_enqueue_ns = 0;
    uint64_t group_work_ns = 0;
    uint64_t desc_issue_ns = 0;
    uint64_t readback_accum_ns = 0;
    uint64_t candidate_scan_ns = 0;

    void print(size_t query_count, size_t work_items) const
    {
        std::printf(
            "[phase-profile] queries=%zu work_items=%zu total_ns=%llu "
            "list_select_ns=%llu geo_job_build_ns=%llu geo_mask_compute_ns=%llu "
            "geo_desc_map_ns=%llu geo_job_enqueue_ns=%llu group_work_ns=%llu "
            "desc_issue_ns=%llu readback_accum_ns=%llu candidate_scan_ns=%llu\n",
            query_count,
            work_items,
            static_cast<unsigned long long>(total_ns),
            static_cast<unsigned long long>(list_select_ns),
            static_cast<unsigned long long>(geo_job_build_ns),
            static_cast<unsigned long long>(geo_mask_compute_ns),
            static_cast<unsigned long long>(geo_desc_map_ns),
            static_cast<unsigned long long>(geo_job_enqueue_ns),
            static_cast<unsigned long long>(group_work_ns),
            static_cast<unsigned long long>(desc_issue_ns),
            static_cast<unsigned long long>(readback_accum_ns),
            static_cast<unsigned long long>(candidate_scan_ns));
    }
};
#else
static inline uint64_t
profileNowNs()
{
    return 0;
}

struct ScopedPhaseTimer
{
    explicit ScopedPhaseTimer(uint64_t &) {}
};

struct MatchPhaseTotals
{
    uint64_t total_ns = 0;
    uint64_t list_select_ns = 0;
    uint64_t geo_job_build_ns = 0;
    uint64_t geo_mask_compute_ns = 0;
    uint64_t geo_desc_map_ns = 0;
    uint64_t geo_job_enqueue_ns = 0;
    uint64_t group_work_ns = 0;
    uint64_t desc_issue_ns = 0;
    uint64_t readback_accum_ns = 0;
    uint64_t candidate_scan_ns = 0;

    void print(size_t, size_t) const {}
};
#endif

} // namespace

struct DescJob {
    uint32_t qid;
    uint32_t lid;
    uint32_t g;
    uint16_t bank, mat, array;
    uint32_t geo_mask_idx;
};

struct MatWork {
    uint32_t qid;
    uint16_t bank, mat;
    uint32_t array_mask;
    size_t job_begin;   // index into jobs_per_q[qid]
    size_t job_end;     // [begin, end)
};

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

using RowQuad = std::array<uint16_t, 4>;
using GeoValueArray = std::array<uint8_t, IvfMatcher::kGeoSweepWidth>;
using GeoMaskArray = std::array<IvfMatcher::MaskRow, IvfMatcher::kGeoSweepWidth>;
using GeoTempBlock = std::array<uint8_t, 2 * IvfMatcher::kGeoSweepWidth * kRowBytes>;
using WordRow = std::array<uint64_t, kRowBytes / sizeof(uint64_t)>;

static_assert((kRowBytes % sizeof(uint64_t)) == 0, "kRowBytes must be word-aligned");

static inline uint64_t
load_u64(const uint8_t *src)
{
    uint64_t value = 0;
    std::memcpy(&value, src, sizeof(value));
    return value;
}

static inline void
store_u64(uint8_t *dst, uint64_t value)
{
    std::memcpy(dst, &value, sizeof(value));
}

static inline uint8_t
count_mismatch_rows_for_lane(const uint8_t* row_block,
                             size_t num_rows,
                             size_t byte_index,
                             uint8_t lane_mask)
{
    uint8_t mismatch = 0u;
    const uint8_t* row_ptr = row_block + byte_index;
    for (size_t row = 0; row < num_rows; ++row, row_ptr += kRowBytes) {
        mismatch = static_cast<uint8_t>(mismatch + ((*row_ptr & lane_mask) != 0u ? 1u : 0u));
    }
    return mismatch;
}

static inline bool
mask_or_and_accumulate(msim::IvfMatcher::MaskRow& out,
                       const msim::IvfMatcher::MaskRow& a,
                       const msim::IvfMatcher::MaskRow& b)
{
    bool any_hit = false;
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= out.size(); i += sizeof(uint64_t)) {
        const uint64_t combined = load_u64(a.data() + i) & load_u64(b.data() + i);
        store_u64(out.data() + i, load_u64(out.data() + i) | combined);
        any_hit = any_hit || (combined != 0u);
    }
    for (; i < out.size(); ++i) {
        const uint8_t combined = static_cast<uint8_t>(a[i] & b[i]);
        out[i] = static_cast<uint8_t>(out[i] | combined);
        any_hit = any_hit || (combined != 0u);
    }
    return any_hit;
}

static inline GeoValueArray
build_geo_probe_values(uint8_t center, uint8_t geo_max)
{
    GeoValueArray vals{};
    size_t idx = 0;
    for (int delta = -kGeoProbeRadius; delta <= kGeoProbeRadius; ++delta) {
        vals[idx++] = clamp_u8(static_cast<int>(center) + delta, 0, geo_max);
    }
    return vals;
}

static inline RowQuad
geo_rows_for_nibble(uint16_t row_base, bool is_x, bool high, uint8_t nibble)
{
    RowQuad rows{};
    for (int bit = 0; bit < 4; ++bit) {
        const bool inv = ((nibble >> bit) & 1u) != 0u;
        if (is_x) {
            rows[static_cast<size_t>(bit)] = static_cast<uint16_t>(
                row_base + (high ? geo_x_row_high(bit, inv) : geo_x_row_low(bit, inv)));
        } else {
            rows[static_cast<size_t>(bit)] = static_cast<uint16_t>(
                row_base + (high ? geo_y_row_high(bit, inv) : geo_y_row_low(bit, inv)));
        }
    }
    return rows;
}

// Batch-compute equality masks for multiple candidate values on one axis.
// Each candidate uses two temp rows (low/high nibble mismatch).
// The helper writes the rows contiguously, then pulls them back in one block copy.
static inline void geo_batch_eq_masks_axis_block(
    CimModule& cim,
    uint16_t bank,
    uint16_t mat,
    uint16_t array,
    uint16_t row_base,
    bool is_x,
    const uint8_t *vals,
    size_t num_vals,
    msim::IvfMatcher::MaskRow *out_eq_masks,
    uint16_t temp_base_row,
    GeoTempBlock &block
) {
    using MaskRow = msim::IvfMatcher::MaskRow;
    if (num_vals == 0) return;

    // Keep geo temp rows separate from descriptor temp rows.
    const uint16_t need_rows = static_cast<uint16_t>(2u * num_vals);
    if (temp_base_row + need_rows > 64) {
        throw std::runtime_error("geo_batch_eq_masks_axis_block: temp rows exceed [0,63] region");
    }

    // A. For each candidate value, issue two ORs and write them to temp rows
    //    temp_base + 2*i and temp_base + 2*i + 1.
    for (size_t i = 0; i < num_vals; ++i) {
        const uint8_t v  = vals[i];
        const uint8_t lo = static_cast<uint8_t>(v & 0x0Fu);
        const uint8_t hi = static_cast<uint8_t>((v >> 4) & 0x0Fu);

        auto issue_nibble = [&](bool high, uint8_t nib, uint16_t temp_row) {
            const RowQuad rows = geo_rows_for_nibble(row_base, is_x, high, nib);
            cim.OR(rows,
                   0xffu,
                   CimModule::Mask::bank(bank),
                   CimModule::Mask::colsAll(),
                   temp_row,
                   CimModule::Mask::mat(mat),
                   CimModule::Mask::array(array));
        };

        issue_nibble(false, lo, static_cast<uint16_t>(temp_base_row + 2u * i + 0u));
        issue_nibble(true,  hi, static_cast<uint16_t>(temp_base_row + 2u * i + 1u));
    }

    // B. Read back the full temp block in one transfer.
    cim.copy_temp_block_to_cpu(block.data(),
                               bank, mat, array,
                               temp_base_row,
                               static_cast<size_t>(need_rows));

    // C. Combine the low/high mismatch rows into a byte-level equality mask.
    for (size_t i = 0; i < num_vals; ++i) {
        const uint8_t* mis_lo = block.data() + (static_cast<size_t>(2u * i + 0u) * kRowBytes);
        const uint8_t* mis_hi = block.data() + (static_cast<size_t>(2u * i + 1u) * kRowBytes);

        MaskRow &eq = out_eq_masks[i];
        size_t j = 0;
        for (; j + sizeof(uint64_t) <= eq.size(); j += sizeof(uint64_t)) {
            store_u64(eq.data() + j, (~load_u64(mis_lo + j)) & (~load_u64(mis_hi + j)));
        }
        for (; j < eq.size(); ++j) {
            const uint8_t a = static_cast<uint8_t>(~mis_lo[j]);
            const uint8_t b = static_cast<uint8_t>(~mis_hi[j]);
            eq[j] = static_cast<uint8_t>(a & b);
        }
    }
}

void IvfMatcher::mask_and_inplace(MaskRow& a, const MaskRow& b)
{
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= a.size(); i += sizeof(uint64_t)) {
        store_u64(a.data() + i, load_u64(a.data() + i) & load_u64(b.data() + i));
    }
    for (; i < a.size(); ++i) {
        a[i] = static_cast<uint8_t>(a[i] & b[i]);
    }
}

void IvfMatcher::mask_or_inplace(MaskRow& a, const MaskRow& b)
{
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= a.size(); i += sizeof(uint64_t)) {
        store_u64(a.data() + i, load_u64(a.data() + i) | load_u64(b.data() + i));
    }
    for (; i < a.size(); ++i) {
        a[i] = static_cast<uint8_t>(a[i] | b[i]);
    }
}

void IvfMatcher::mask_not_inplace(MaskRow& a)
{
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= a.size(); i += sizeof(uint64_t)) {
        store_u64(a.data() + i, ~load_u64(a.data() + i));
    }
    for (; i < a.size(); ++i) {
        a[i] = static_cast<uint8_t>(~a[i]);
    }
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


uint8_t IvfMatcher::Planes::lane_value(int lane) const
{
    uint8_t v = 0;
    for (size_t k = 0; k < b.size(); ++k) {
        const uint8_t bit = lane_get_bit(b[k].data(), lane);
        v = static_cast<uint8_t>(v | static_cast<uint8_t>(bit << k));
    }
    return v;
}

void IvfMatcher::Planes::add_mask(const MaskRow& x)
{
    add_mask_row_bytes(x.data());
}

void IvfMatcher::Planes::add_mask_row_bytes(const uint8_t* row_bytes)
{
    WordRow carry_storage{};
    WordRow next_storage{};
    std::memcpy(carry_storage.data(), row_bytes, kRowBytes);

    WordRow *carry = &carry_storage;
    WordRow *next = &next_storage;

    for (size_t k = 0; k < b.size(); ++k) {
        for (size_t w = 0; w < carry->size(); ++w) {
            const size_t byte_off = w * sizeof(uint64_t);
            const uint64_t bk = load_u64(b[k].data() + byte_off);
            const uint64_t c = (*carry)[w];
            store_u64(b[k].data() + byte_off, bk ^ c);
            (*next)[w] = bk & c;
        }
        std::swap(carry, next);
    }
}

void IvfMatcher::Planes::add_mask_block_bytes(const uint8_t* base,
                                              size_t stride_bytes,
                                              size_t num_rows)
{
    const uint8_t *row_ptr = base;
    for (size_t r = 0; r < num_rows; ++r, row_ptr += stride_bytes) {
        add_mask_row_bytes(row_ptr);
    }
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
        const RowQuad rows = geo_rows_for_nibble(row_base, is_x, high, nib);

        cim_.OR(rows,
                0xffu,
                CimModule::Mask::bank(bank),
                CimModule::Mask::colsAll(),
                0,
                CimModule::Mask::mat(mat),
                CimModule::Mask::array(array));

        MaskRow tmp{};
        cim_.copy_temp_to_cpu(tmp.data(), bank, mat, array, 0, kRowBytes);
        return tmp;
    };

    MaskRow mis_lo = or_nibble_mismatch(false, lo);
    MaskRow mis_hi = or_nibble_mismatch(true,  hi);

    mask_not_inplace(mis_lo);       // eq_lo
    mask_not_inplace(mis_hi);       // eq_hi
    mask_and_inplace(mis_lo, mis_hi); // eq_byte
    return mis_lo;
}

bool IvfMatcher::geo_eq_masks_xy(uint32_t bucket_id, uint32_t group_in_bucket,
                                 uint8_t qgx, uint8_t qgy,
                                 const std::array<uint8_t, kGeoSweepWidth>& gx_vals,
                                 const std::array<uint8_t, kGeoSweepWidth>& gy_vals,
                                 MaskRow& out_mask_xy)
{
    static_cast<void>(qgx);
    static_cast<void>(qgy);
    static constexpr size_t kGeoProbeCenter = kGeoSweepWidth / 2u;
    out_mask_xy.fill(0);

    // Use temp rows [0,63] for geo, and keep descriptor temp rows above that.
    static constexpr uint16_t kGeoTempBase = 0;
    GeoMaskArray eq_x_masks{};
    GeoMaskArray eq_y_masks{};
    GeoTempBlock block{};
    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    map_geo_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array, row_base);

    geo_batch_eq_masks_axis_block(cim_, bank, mat, array, row_base,
                                  /*is_x=*/true, gx_vals.data(), gx_vals.size(),
                                  eq_x_masks.data(), kGeoTempBase, block);
    geo_batch_eq_masks_axis_block(cim_, bank, mat, array, row_base,
                                  /*is_x=*/false, gy_vals.data(), gy_vals.size(),
                                  eq_y_masks.data(), kGeoTempBase, block);

    const MaskRow& mx_fixed = eq_x_masks[kGeoProbeCenter];
    const MaskRow& my_fixed = eq_y_masks[kGeoProbeCenter];

    bool any_hit = false;
    for (size_t i = 0; i < gx_vals.size(); ++i) {
        any_hit = mask_or_and_accumulate(out_mask_xy, eq_x_masks[i], my_fixed) || any_hit;
    }

    for (size_t i = 0; i < gy_vals.size(); ++i) {
        any_hit = mask_or_and_accumulate(out_mask_xy, eq_y_masks[i], mx_fixed) || any_hit;
    }

    return any_hit;
}

void IvfMatcher::desc_mismatch_planes(uint32_t bucket_id, uint32_t group_in_bucket,
                                      const uint8_t* qdesc64, Planes& out_planes)
{
    out_planes.clear();

    // Keep descriptor temp rows away from geo temp rows.
    static constexpr uint16_t kDescTempBase = 64;

    uint16_t bank = 0, mat = 0, array = 0;
    map_desc_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array);

    std::vector<uint8_t> active_dims;
    active_dims.reserve(kDescDims);
    for (int d = 0; d < kDescDims; ++d) {
        if (qdesc64[d] != 0u) {
            active_dims.push_back(static_cast<uint8_t>(d));
        }
    }
    if (active_dims.empty()) {
        return;
    }

    // A. Issue one OR per active descriptor dimension and pack the results
    //    into consecutive temp rows.
    for (size_t packed_idx = 0; packed_idx < active_dims.size(); ++packed_idx) {
        const int d = static_cast<int>(active_dims[packed_idx]);
        const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
        RowQuad rows{};

        for (int b = 0; b < kDescBits; ++b) {
            const int qbit = (qv >> b) & 1u;
            rows[static_cast<size_t>(b)] =
                (qbit != 0) ? desc_inv_row(d, b) : desc_true_row(d, b);
        }

        cim_.OR(rows,
                0xffu,
                CimModule::Mask::bank(bank),
                CimModule::Mask::colsAll(),
                static_cast<uint16_t>(kDescTempBase + packed_idx),
                CimModule::Mask::mat(mat),
                CimModule::Mask::array(array));
    }

    // B. Read back only the temp rows that were written.
    std::vector<uint8_t> block;
    block.resize(active_dims.size() * static_cast<size_t>(kRowBytes));

    cim_.copy_temp_block_to_cpu(block.data(),
                                bank, mat, array,
                                kDescTempBase,
                                active_dims.size());

    // C. Accumulate only the active mismatch rows.
    for (size_t packed_idx = 0; packed_idx < active_dims.size(); ++packed_idx) {
        const uint8_t* row_ptr = block.data() + packed_idx * static_cast<size_t>(kRowBytes);
        out_planes.add_mask_row_bytes(row_ptr);
    }
}

MatchResult IvfMatcher::match_one_query_desc(uint8_t qgx, uint8_t qgy,
                                             const uint8_t* qdesc64,
                                             uint32_t nprobe,
                                             uint32_t max_groups_per_list)
{
    const auto results = match_one_step(qgx, qgy, qdesc64, 1, nprobe, max_groups_per_list);
    return results.empty() ? MatchResult{} : results.front();
}

std::vector<MatchResult>
IvfMatcher::match_one_step(uint8_t qgx, uint8_t qgy,
                           const uint8_t* desc_ptr, size_t m,
                           uint32_t nprobe, uint32_t max_groups_per_list)
{
    // The current placement uses 16 banks (BANK_BITS=4).
    static constexpr uint16_t kNumBanks = 16;
    static constexpr uint16_t kDescTempBase = 64;

    std::vector<MatchResult> results(m);
    if (m == 0) return results;

    MatchPhaseTotals phase_totals;
    const uint64_t total_start_ns = profileNowNs();
    const GeoValueArray gx_probe_vals = build_geo_probe_values(qgx, map_.geo_max);
    const GeoValueArray gy_probe_vals = build_geo_probe_values(qgy, map_.geo_max);

    const uint32_t K = map_.postings.nlist;
    std::vector<uint32_t> geo_group_base(static_cast<size_t>(K) + 1u, 0u);
    for (uint32_t lid = 0; lid < K; ++lid) {
        geo_group_base[static_cast<size_t>(lid) + 1u] =
            geo_group_base[static_cast<size_t>(lid)] + place_.bucket.at(lid).desc_groups;
    }

    const uint32_t total_geo_groups = geo_group_base.back();
    std::vector<MaskRow> geo_masks(total_geo_groups);
    std::vector<uint8_t> geo_ready(total_geo_groups, 0u);
    std::vector<uint8_t> geo_has_hits(total_geo_groups, 0u);

    auto geo_index = [&](uint32_t lid, uint32_t group_in_bucket) -> uint32_t {
        return geo_group_base[static_cast<size_t>(lid)] + group_in_bucket;
    };

    auto ensure_geo_mask = [&](uint32_t lid, uint32_t group_in_bucket) -> bool {
        const uint32_t idx = geo_index(lid, group_in_bucket);
        if (geo_ready[idx] == 0u) {
            geo_has_hits[idx] = 0u;
            {
                ScopedPhaseTimer timer(phase_totals.geo_mask_compute_ns);
                geo_has_hits[idx] = geo_eq_masks_xy(
                    lid,
                    group_in_bucket,
                    qgx,
                    qgy,
                    gx_probe_vals,
                    gy_probe_vals,
                    geo_masks[idx]) ? 1u : 0u;
            }
            geo_ready[idx] = 1u;
        }
        return geo_has_hits[idx] != 0u;
    };

    // Per-query work queues. Descriptor mismatches are counted directly from the
    // read-back temp rows, which is significantly cheaper in no-opt builds than
    // maintaining bit-sliced accumulators and reconstructing lane values later.
    std::vector<std::vector<DescJob>> jobs_per_q(m);
    std::vector<std::vector<uint8_t>> active_dims_per_q(m);

    // Phase 1. Build jobs after list selection and geo filtering.
    for (size_t qi = 0; qi < m; ++qi) {
        const uint8_t* qdesc64 = desc_ptr + qi * 64u;
        auto &active_dims = active_dims_per_q[qi];
        active_dims.reserve(kDescDims);
        for (int d = 0; d < kDescDims; ++d) {
            if (qdesc64[d] != 0u) {
                active_dims.push_back(static_cast<uint8_t>(d));
            }
        }

        std::vector<uint32_t> lists;
        {
            ScopedPhaseTimer timer(phase_totals.list_select_ns);
            lists = select_lists_cpu(qdesc64, nprobe);
        }

        auto& jobs = jobs_per_q[qi];
        jobs.reserve(lists.size() * 4);

        {
            ScopedPhaseTimer timer(phase_totals.geo_job_build_ns);
            for (uint32_t lid : lists) {
                const auto& bp = place_.bucket.at(lid);
                const uint32_t groups_total = bp.desc_groups;
                const uint32_t groups = (max_groups_per_list == 0u)
                    ? groups_total
                    : std::min<uint32_t>(groups_total, max_groups_per_list);

                for (uint32_t g = 0; g < groups; ++g) {
                    const uint32_t geo_idx = geo_index(lid, g);
                    if (!ensure_geo_mask(lid, g)) continue;

                    uint16_t bank=0, mat=0, array=0;
                    {
                        ScopedPhaseTimer timer(phase_totals.geo_desc_map_ns);
                        map_desc_bucket_group_to_region(place_, lid, g, bank, mat, array);
                    }

                    {
                        ScopedPhaseTimer timer(phase_totals.geo_job_enqueue_ns);
                        jobs.push_back(DescJob{
                            static_cast<uint32_t>(qi), lid, g, bank, mat, array, geo_idx
                        });
                    }
                }
            }
        }
    }

    // Phase 2. Group each query into bank/mat work items.
    std::array<std::deque<MatWork>, kNumBanks> bank_q{};
    size_t total_work_items = 0;

    {
        ScopedPhaseTimer timer(phase_totals.group_work_ns);
        for (size_t qi = 0; qi < m; ++qi) {
            auto& jobs = jobs_per_q[qi];
            if (jobs.empty()) continue;

            std::sort(jobs.begin(), jobs.end(), [](const DescJob& a, const DescJob& b){
                if (a.bank != b.bank) return a.bank < b.bank;
                if (a.mat  != b.mat)  return a.mat  < b.mat;
                return a.array < b.array;
            });

            // Create one work item for each contiguous (bank, mat) segment.
            for (size_t base = 0; base < jobs.size(); ) {
                const uint16_t bank = jobs[base].bank;
                const uint16_t mat  = jobs[base].mat;

                size_t end = base;
                uint32_t array_mask = 0;
                while (end < jobs.size() && jobs[end].bank == bank && jobs[end].mat == mat) {
                    array_mask |= CimModule::Mask::array(jobs[end].array);
                    ++end;
                }

                if (bank >= kNumBanks) {
                    throw std::runtime_error("match_one_step: bank out of range (expect 0..15)");
                }

                bank_q[bank].push_back(MatWork{
                    static_cast<uint32_t>(qi),
                    bank, mat,
                    array_mask,
                    base, end
                });
                total_work_items++;

                base = end;
            }
        }
    }

    if (total_work_items == 0) {
        phase_totals.total_ns = profileNowNs() - total_start_ns;
        phase_totals.print(m, 0u);
        return results;
    }
    const size_t profiled_work_items = total_work_items;

    // Phase 3. Round-robin issue across banks:
    //   1. Issue OR commands for all active descriptor dimensions in one work item.
    //   2. Read back the temp rows once per array.
    //   3. Count mismatches directly from those rows and update the best result.
    std::vector<uint8_t> block64(static_cast<size_t>(kDescDims) * kRowBytes);

    const int32_t* const map_ids = map_.postings.map_ids.data();

    while (total_work_items > 0) {
        for (uint16_t bank = 0; bank < kNumBanks; ++bank) {
            if (bank_q[bank].empty()) continue;

            MatWork w = bank_q[bank].front();
            bank_q[bank].pop_front();

            const uint32_t qid = w.qid;
            const uint8_t* qdesc64 = desc_ptr + static_cast<size_t>(qid) * 64u;
            const auto &active_dims = active_dims_per_q[qid];
            const size_t active_dim_count = active_dims.size();

            auto& jobs = jobs_per_q[qid];

            // A. Issue one OR per non-zero descriptor dimension.
            {
                ScopedPhaseTimer timer(phase_totals.desc_issue_ns);
                for (size_t packed_idx = 0; packed_idx < active_dim_count; ++packed_idx) {
                    const int d = static_cast<int>(active_dims[packed_idx]);
                    const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
                    RowQuad rows{};

                    for (int b = 0; b < kDescBits; ++b) {
                        const int qbit = (qv >> b) & 1u;
                        rows[static_cast<size_t>(b)] =
                            (qbit != 0) ? desc_inv_row(d, b) : desc_true_row(d, b);
                    }

                    cim_.OR(rows,
                            0xffu,
                            CimModule::Mask::bank(w.bank),
                            CimModule::Mask::colsAll(),
                            /*temp_row=*/static_cast<uint16_t>(kDescTempBase + packed_idx),
                            CimModule::Mask::mat(w.mat),
                            w.array_mask);
                }
            }

            // B/C. Arrays are already contiguous inside a (bank, mat) work item
            // because jobs were sorted by bank, mat, array. Read back one array at
            // a time, then count mismatches directly from the temp rows.
            for (size_t arr_begin = w.job_begin; arr_begin < w.job_end; ) {
                const uint16_t array = jobs[arr_begin].array;
                size_t arr_end = arr_begin + 1;
                while (arr_end < w.job_end && jobs[arr_end].array == array) {
                    ++arr_end;
                }

                if (active_dim_count != 0u) {
                    ScopedPhaseTimer timer(phase_totals.readback_accum_ns);
                    cim_.copy_temp_block_to_cpu(block64.data(),
                                                w.bank, w.mat, array,
                                                kDescTempBase,
                                                active_dim_count);
                }

                {
                    ScopedPhaseTimer timer(phase_totals.candidate_scan_ns);
                    for (size_t ji = arr_begin; ji < arr_end; ++ji) {
                        const auto& jb = jobs[ji];
                        const auto& bp = place_.bucket.at(jb.lid);
                        const MaskRow& geo_xy = geo_masks[jb.geo_mask_idx];

                        const uint32_t base_off = bp.posting_off + jb.g * static_cast<uint32_t>(kLanesPerGroup);
                        const uint32_t remain   = (bp.map_count > jb.g * static_cast<uint32_t>(kLanesPerGroup))
                                                ? (bp.map_count - jb.g * static_cast<uint32_t>(kLanesPerGroup))
                                                : 0u;
                        const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);
                        const uint32_t valid_bytes = ceil_div_u32(valid, 8u);

                        for (uint32_t byte_index = 0; byte_index < valid_bytes; ++byte_index) {
                            uint8_t candidate_mask = geo_xy[byte_index];
                            if (candidate_mask == 0u) continue;

                            const uint32_t lane_base = byte_index * 8u;
                            if (lane_base + 8u > valid) {
                                const uint32_t tail_bits = valid - lane_base;
                                candidate_mask = static_cast<uint8_t>(
                                    candidate_mask & static_cast<uint8_t>((1u << tail_bits) - 1u));
                                if (candidate_mask == 0u) continue;
                            }

                            for (uint32_t bit = 0; bit < 8u; ++bit) {
                                const uint8_t lane_mask = static_cast<uint8_t>(1u << bit);
                                if ((candidate_mask & lane_mask) == 0u) continue;

                                const uint32_t lane = lane_base + bit;
                                const int32_t mid = map_ids[static_cast<size_t>(base_off + lane)];
                                const uint8_t mis = (active_dim_count == 0u)
                                    ? 0u
                                    : count_mismatch_rows_for_lane(
                                        block64.data(),
                                        active_dim_count,
                                        static_cast<size_t>(byte_index),
                                        lane_mask);

                                auto& best = results[qid];
                                if (mis < best.best_mismatch || (mis == best.best_mismatch && mid < best.best_map_id)) {
                                    best.best_mismatch = mis;
                                    best.best_map_id = mid;
                                }
                            }
                        }
                    }
                }

                arr_begin = arr_end;
            }

            total_work_items--;
        }
    }

    phase_totals.total_ns = profileNowNs() - total_start_ns;
    phase_totals.print(m, profiled_work_items);

    return results;
}

} // namespace msim
