#include "ivf_matcher.hpp"
#include "layout.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <deque>

namespace msim {

struct DescJob {
    uint32_t qid;
    uint32_t lid;
    uint32_t g;
    uint16_t bank, mat, array;
    msim::IvfMatcher::MaskRow geo_xy; 
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

// Batch-compute equality masks for multiple candidate values on one axis.
// Each candidate uses two temp rows (low/high nibble mismatch).
// The helper writes the rows contiguously, then pulls them back in one block copy.
static inline void geo_batch_eq_masks_axis_block(
    CimModule& cim,
    const msim::IvfPlacement& place,
    uint32_t bucket_id,
    uint32_t group_in_bucket,
    bool is_x,
    const std::vector<uint8_t>& vals,
    std::vector<msim::IvfMatcher::MaskRow>& out_eq_masks,
    uint16_t temp_base_row
) {
    using MaskRow = msim::IvfMatcher::MaskRow;

    out_eq_masks.clear();
    out_eq_masks.resize(vals.size());

    if (vals.empty()) return;

    // Keep geo temp rows separate from descriptor temp rows.
    const uint16_t need_rows = static_cast<uint16_t>(2u * vals.size());
    if (temp_base_row + need_rows > 64) {
        throw std::runtime_error("geo_batch_eq_masks_axis_block: temp rows exceed [0,63] region");
    }

    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    map_geo_bucket_group_to_region(place, bucket_id, group_in_bucket, bank, mat, array, row_base);

    std::vector<uint16_t> rows;
    rows.reserve(4);

    // A. For each candidate value, issue two ORs and write them to temp rows
    //    temp_base + 2*i and temp_base + 2*i + 1.
    for (size_t i = 0; i < vals.size(); ++i) {
        const uint8_t v  = vals[i];
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
                        row_base + (!high ? geo_x_row_low(b, inv) : geo_x_row_high(b, inv)));
                } else {
                    r = static_cast<uint16_t>(
                        row_base + (!high ? geo_y_row_low(b, inv) : geo_y_row_high(b, inv)));
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
        issue_nibble(true,  hi, static_cast<uint16_t>(temp_base_row + 2u * i + 1u));
    }

    // B. Read back the full temp block in one transfer.
    std::vector<uint8_t> block;
    block.resize(static_cast<size_t>(need_rows) * static_cast<size_t>(kRowBytes));

    cim.copy_temp_block_to_cpu(block.data(),
                               bank, mat, array,
                               temp_base_row,
                               static_cast<size_t>(need_rows));

    // C. Combine the low/high mismatch rows into a byte-level equality mask.
    for (size_t i = 0; i < vals.size(); ++i) {
        const uint8_t* mis_lo = block.data() + (static_cast<size_t>(2u * i + 0u) * kRowBytes);
        const uint8_t* mis_hi = block.data() + (static_cast<size_t>(2u * i + 1u) * kRowBytes);

        MaskRow eq{};
        for (size_t j = 0; j < eq.size(); ++j) {
            const uint8_t a = static_cast<uint8_t>(~mis_lo[j]);
            const uint8_t b = static_cast<uint8_t>(~mis_hi[j]);
            eq[j] = static_cast<uint8_t>(a & b);
        }
        out_eq_masks[i] = eq;
    }
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
    MaskRow carry{};
    for (size_t i = 0; i < carry.size(); ++i) carry[i] = row_bytes[i];

    for (size_t k = 0; k < b.size(); ++k) {
        MaskRow sum{};
        MaskRow new_carry{};
        for (size_t i = 0; i < sum.size(); ++i) {
            const uint8_t bk = b[k][i];
            const uint8_t c  = carry[i];
            sum[i]       = static_cast<uint8_t>(bk ^ c);
            new_carry[i] = static_cast<uint8_t>(bk & c);
        }
        b[k] = sum;
        carry = new_carry;
    }
}

void IvfMatcher::Planes::add_mask_block_bytes(const uint8_t* base,
                                              size_t stride_bytes,
                                              size_t num_rows)
{
    for (size_t r = 0; r < num_rows; ++r) {
        add_mask_row_bytes(base + r * stride_bytes);
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
        return tmp;
    };

    MaskRow mis_lo = or_nibble_mismatch(false, lo);
    MaskRow mis_hi = or_nibble_mismatch(true,  hi);

    mask_not_inplace(mis_lo);       // eq_lo
    mask_not_inplace(mis_hi);       // eq_hi
    mask_and_inplace(mis_lo, mis_hi); // eq_byte
    return mis_lo;
}

void IvfMatcher::geo_eq_masks_xy(uint32_t bucket_id, uint32_t group_in_bucket,
                                 uint8_t qgx, uint8_t qgy, MaskRow& out_mask_xy)
{
    out_mask_xy.fill(0);

    // Use temp rows [0,63] for geo, and keep descriptor temp rows above that.
    static constexpr uint16_t kGeoTempBase = 0;

    // 1. Fixed-axis masks for the query center.
    std::vector<uint8_t> one_val;

    MaskRow my_fixed{};
    {
        one_val = { qgy };
        std::vector<MaskRow> eqs;
        geo_batch_eq_masks_axis_block(cim_, place_, bucket_id, group_in_bucket,
                                      /*is_x=*/false, one_val, eqs, kGeoTempBase);
        my_fixed = eqs[0];
    }

    MaskRow mx_fixed{};
    {
        one_val = { qgx };
        std::vector<MaskRow> eqs;
        geo_batch_eq_masks_axis_block(cim_, place_, bucket_id, group_in_bucket,
                                      /*is_x=*/true, one_val, eqs, kGeoTempBase);
        mx_fixed = eqs[0];
    }

    // 2. Sweep x around the query location while y stays fixed.
    std::vector<uint8_t> gx_vals;
    gx_vals.reserve(static_cast<size_t>(2 * kGeoProbeRadius + 1));
    for (int dx = -kGeoProbeRadius; dx <= kGeoProbeRadius; ++dx) {
        gx_vals.push_back(clamp_u8(static_cast<int>(qgx) + dx, 0, map_.geo_max));
    }

    std::vector<MaskRow> mx_list;
    geo_batch_eq_masks_axis_block(cim_, place_, bucket_id, group_in_bucket,
                                  /*is_x=*/true, gx_vals, mx_list, kGeoTempBase);

    for (size_t i = 0; i < mx_list.size(); ++i) {
        MaskRow tmp = mx_list[i];
        mask_and_inplace(tmp, my_fixed);
        mask_or_inplace(out_mask_xy, tmp);
    }

    // 3. Sweep y around the query location while x stays fixed.
    std::vector<uint8_t> gy_vals;
    gy_vals.reserve(static_cast<size_t>(2 * kGeoProbeRadius + 1));
    for (int dy = -kGeoProbeRadius; dy <= kGeoProbeRadius; ++dy) {
        gy_vals.push_back(clamp_u8(static_cast<int>(qgy) + dy, 0, map_.geo_max));
    }

    std::vector<MaskRow> my_list;
    geo_batch_eq_masks_axis_block(cim_, place_, bucket_id, group_in_bucket,
                                  /*is_x=*/false, gy_vals, my_list, kGeoTempBase);

    for (size_t i = 0; i < my_list.size(); ++i) {
        MaskRow tmp = my_list[i];
        mask_and_inplace(tmp, mx_fixed);
        mask_or_inplace(out_mask_xy, tmp);
    }
}

void IvfMatcher::desc_mismatch_planes(uint32_t bucket_id, uint32_t group_in_bucket,
                                      const uint8_t* qdesc64, Planes& out_planes)
{
    out_planes.clear();

    // Keep descriptor temp rows away from geo temp rows.
    static constexpr uint16_t kDescTempBase = 64;

    uint16_t bank = 0, mat = 0, array = 0;
    map_desc_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array);

    std::vector<uint16_t> rows;
    rows.reserve(kDescBits);

    // A. Issue one OR per descriptor dimension and store the results in
    //    consecutive temp rows.
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
                static_cast<uint16_t>(kDescTempBase + d),
                CimModule::Mask::mat(mat),
                CimModule::Mask::array(array));
    }

    // B. Read back the 64 temp rows in one transfer.
    std::vector<uint8_t> block;
    block.resize(static_cast<size_t>(kDescDims) * static_cast<size_t>(kRowBytes));

    cim_.copy_temp_block_to_cpu(block.data(),
                                bank, mat, array,
                                kDescTempBase,
                                static_cast<size_t>(kDescDims));

    // C. Accumulate the 64 mismatch rows with the bit-sliced adder.
    for (int d = 0; d < kDescDims; ++d) {
        const uint8_t* row_ptr = block.data() + static_cast<size_t>(d) * kRowBytes;
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

    const uint32_t K = map_.postings.nlist;
    std::vector<uint32_t> geo_group_base(static_cast<size_t>(K) + 1u, 0u);
    for (uint32_t lid = 0; lid < K; ++lid) {
        geo_group_base[static_cast<size_t>(lid) + 1u] =
            geo_group_base[static_cast<size_t>(lid)] + place_.bucket.at(lid).desc_groups;
    }

    const uint32_t total_geo_groups = geo_group_base.back();
    std::vector<MaskRow> geo_masks(total_geo_groups);
    std::vector<uint8_t> geo_ready(total_geo_groups, 0u);

    auto geo_index = [&](uint32_t lid, uint32_t group_in_bucket) -> uint32_t {
        return geo_group_base[static_cast<size_t>(lid)] + group_in_bucket;
    };

    auto get_geo_mask = [&](uint32_t lid, uint32_t group_in_bucket) -> const MaskRow& {
        const uint32_t idx = geo_index(lid, group_in_bucket);
        if (geo_ready[idx] == 0u) {
            geo_eq_masks_xy(lid, group_in_bucket, qgx, qgy, geo_masks[idx]);
            geo_ready[idx] = 1u;
        }
        return geo_masks[idx];
    };

    // Per-query work queues and bit-sliced accumulators.
    std::vector<std::vector<DescJob>> jobs_per_q(m);
    std::vector<std::vector<Planes>>  planes_per_q(m);

    // Phase 1. Build jobs after list selection and geo filtering.
    for (size_t qi = 0; qi < m; ++qi) {
        const uint8_t* qdesc64 = desc_ptr + qi * 64u;

        const std::vector<uint32_t> lists = select_lists_cpu(qdesc64, nprobe);

        auto& jobs = jobs_per_q[qi];
        jobs.reserve(lists.size() * 4);

        for (uint32_t lid : lists) {
            const auto& bp = place_.bucket.at(lid);
            const uint32_t groups_total = bp.desc_groups;
            const uint32_t groups = (max_groups_per_list == 0u)
                ? groups_total
                : std::min<uint32_t>(groups_total, max_groups_per_list);

            for (uint32_t g = 0; g < groups; ++g) {
                const MaskRow& geo_xy = get_geo_mask(lid, g);
                if (!mask_any(geo_xy)) continue;

                uint16_t bank=0, mat=0, array=0;
                map_desc_bucket_group_to_region(place_, lid, g, bank, mat, array);

                jobs.push_back(DescJob{
                    static_cast<uint32_t>(qi), lid, g, bank, mat, array, geo_xy
                });
            }
        }

        // Keep one accumulator per job.
        planes_per_q[qi].resize(jobs.size());
        for (auto& p : planes_per_q[qi]) p.clear();
    }

    // Phase 2. Group each query into bank/mat work items.
    std::array<std::deque<MatWork>, kNumBanks> bank_q{};
    size_t total_work_items = 0;

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

    if (total_work_items == 0) {
        return results;
    }

    // Reuse a shared row buffer while issuing commands.
    std::vector<uint16_t> rows;
    rows.reserve(kDescBits);

    // Phase 3. Round-robin issue across banks:
    //   1. Issue OR commands for all active descriptor dimensions in one work item.
    //   2. Read back the 64 temp rows once per array.
    //   3. Feed the rows into the per-job bit-sliced accumulators.
    //   4. Scan the valid lanes and update the best result.
    std::vector<uint8_t> block64(static_cast<size_t>(kDescDims) * kRowBytes);

    const int32_t* const map_ids = map_.postings.map_ids.data();

    while (total_work_items > 0) {
        for (uint16_t bank = 0; bank < kNumBanks; ++bank) {
            if (bank_q[bank].empty()) continue;

            MatWork w = bank_q[bank].front();
            bank_q[bank].pop_front();

            const uint32_t qid = w.qid;
            const uint8_t* qdesc64 = desc_ptr + static_cast<size_t>(qid) * 64u;

            auto& jobs   = jobs_per_q[qid];
            auto& planes = planes_per_q[qid];

            // A. Issue one OR per non-zero descriptor dimension.
            for (int d = 0; d < kDescDims; ++d) {
                if (qdesc64[d] == 0) continue;
                rows.clear();
                const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);

                for (int b = 0; b < kDescBits; ++b) {
                    const int qbit = (qv >> b) & 1u;
                    const uint16_t r = (qbit != 0) ? desc_inv_row(d, b) : desc_true_row(d, b);
                    rows.push_back(r);
                }

                cim_.OR(rows,
                        0xffu,
                        CimModule::Mask::bank(w.bank),
                        CimModule::Mask::colsAll(),
                        /*temp_row=*/static_cast<uint16_t>(kDescTempBase + d),
                        CimModule::Mask::mat(w.mat),
                        w.array_mask);
            }

            // B. Arrays are already contiguous inside a (bank, mat) work item because
            //    jobs were sorted by bank, mat, array.
            for (size_t arr_begin = w.job_begin; arr_begin < w.job_end; ) {
                const uint16_t array = jobs[arr_begin].array;
                size_t arr_end = arr_begin + 1;
                while (arr_end < w.job_end && jobs[arr_end].array == array) {
                    ++arr_end;
                }

                cim_.copy_temp_block_to_cpu(block64.data(),
                                            w.bank, w.mat, array,
                                            kDescTempBase,
                                            kDescDims);

                // Rows for zero-valued query dimensions were never issued, so make sure
                // they do not contribute to the mismatch accumulator.
                for (int d = 0; d < kDescDims; ++d) {
                    if (qdesc64[d] == 0) {
                        std::memset(block64.data() + static_cast<size_t>(d) * kRowBytes,
                                    0,
                                    kRowBytes);
                    }
                }

                for (size_t ji = arr_begin; ji < arr_end; ++ji) {
                    planes[ji].add_mask_block_bytes(block64.data(), kRowBytes, kDescDims);
                }

                arr_begin = arr_end;
            }

            // C. Scan candidate lanes for each finished job.
            for (size_t ji = w.job_begin; ji < w.job_end; ++ji) {
                const auto& jb = jobs[ji];
                const auto& bp = place_.bucket.at(jb.lid);

                const uint32_t base_off = bp.posting_off + jb.g * static_cast<uint32_t>(kLanesPerGroup);
                const uint32_t remain   = (bp.map_count > jb.g * static_cast<uint32_t>(kLanesPerGroup))
                                        ? (bp.map_count - jb.g * static_cast<uint32_t>(kLanesPerGroup))
                                        : 0u;
                const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

                for (uint32_t lane = 0; lane < valid; ++lane) {
                    if (lane_get_bit(jb.geo_xy.data(), static_cast<int>(lane)) == 0u) continue;

                    const int32_t mid = map_ids[static_cast<size_t>(base_off + lane)];
                    const uint32_t mis = planes[ji].lane_value(static_cast<int>(lane));

                    auto& best = results[qid];
                    if (mis < best.best_mismatch || (mis == best.best_mismatch && mid < best.best_map_id)) {
                        best.best_mismatch = static_cast<uint8_t>(mis);
                        best.best_map_id = mid;
                    }
                }
            }

            total_work_items--;
        }
    }

    return results;
}

} // namespace msim
