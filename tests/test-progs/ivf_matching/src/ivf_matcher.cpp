#include "ivf_matcher.hpp"
#include "layout.hpp"
#include <algorithm>
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
    int dim;            // 0..kDescDims
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

// ---- GEO block-copy helper: batch compute eq masks for multiple candidate values on one axis ----
// 每個 candidate value 需要 2 個 temp rows (lo/hi nibble mismatch)
// 會一次 OR 寫到 temp 連續 rows，再用 copy_temp_block_to_cpu 一次讀回。
static inline void geo_batch_eq_masks_axis_block(
    CimModule& cim,
    const msim::IvfPlacement& place,
    uint32_t bucket_id,
    uint32_t group_in_bucket,
    bool is_x,
    const std::vector<uint8_t>& vals,
    std::vector<msim::IvfMatcher::MaskRow>& out_eq_masks,
    uint16_t temp_base_row // 這段 batch 使用的 temp 起始 row
) {
    using MaskRow = msim::IvfMatcher::MaskRow;

    out_eq_masks.clear();
    out_eq_masks.resize(vals.size());

    if (vals.empty()) return;

    // 避免踩到你 desc 用的 temp (你目前 desc base=64)
    // 這裡要求：temp_base_row + 2*vals.size() <= 64
    const uint16_t need_rows = static_cast<uint16_t>(2u * vals.size());
    if (temp_base_row + need_rows > 64) {
        throw std::runtime_error("geo_batch_eq_masks_axis_block: temp rows exceed [0,63] region");
    }

    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    map_geo_bucket_group_to_region(place, bucket_id, group_in_bucket, bank, mat, array, row_base);

    std::vector<uint16_t> rows;
    rows.reserve(4);

    // (A) 對每個 val：做兩次 OR，分別寫到 temp_base + 2*i, 2*i+1
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

        issue_nibble(false, lo, static_cast<uint16_t>(temp_base_row + 2u * i + 0u)); // mis_lo
        issue_nibble(true,  hi, static_cast<uint16_t>(temp_base_row + 2u * i + 1u)); // mis_hi
    }

    // (B) 一次 block copy：把 2*vals.size() 個 rows 全讀回 CPU
    std::vector<uint8_t> block;
    block.resize(static_cast<size_t>(need_rows) * static_cast<size_t>(kRowBytes));

    cim.copy_temp_block_to_cpu(block.data(),
                               bank, mat, array,
                               temp_base_row,
                               static_cast<size_t>(need_rows));

    // (C) 組合成 eq_byte = ~mis_lo & ~mis_hi
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
        return tmp; // mismatch mask
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

    // geo temp 用 [0,63]，desc 你已經用 base=64 以上
    static constexpr uint16_t kGeoTempBase = 0;

    // ---- 1) 先算固定軸：my(qgy)、mx(qgx) 各一次 ----
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

    // ---- 2) dx 探索：gx 變動、gy 固定 ----
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

    // ---- 3) dy 探索：gy 變動、gx 固定 ----
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

    // 避開 geo 使用的 temp_row=0
    static constexpr uint16_t kDescTempBase = 64;

    uint16_t bank = 0, mat = 0, array = 0;
    map_desc_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array);

    std::vector<uint16_t> rows;
    rows.reserve(kDescBits);

    // (A) 64 dims：OR 寫到 temp 的連續 64 rows
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
                static_cast<uint16_t>(kDescTempBase + d),  // <<<<<<<<<<<< 這裡變動
                CimModule::Mask::mat(mat),
                CimModule::Mask::array(array));
    }

    // (B) 一次 block copy 回 CPU：64 rows
    std::vector<uint8_t> block;
    block.resize(static_cast<size_t>(kDescDims) * static_cast<size_t>(kRowBytes));

    cim_.copy_temp_block_to_cpu(block.data(),
                                bank, mat, array,
                                kDescTempBase,
                                static_cast<size_t>(kDescDims));

    // (C) ripple-carry 累積 64 rows
    for (int d = 0; d < kDescDims; ++d) {
        const uint8_t* row_ptr = block.data() + static_cast<size_t>(d) * kRowBytes;
        out_planes.add_mask_row_bytes(row_ptr);
    }
}

// MatchResult IvfMatcher::match_one_query_desc(uint8_t qgx, uint8_t qgy, const uint8_t* qdesc64, uint32_t nprobe, uint32_t max_groups_per_list)
// {
//     MatchResult best;
//     const uint32_t K = map_.postings.nlist;
//     if (K == 0) return best;

//     const std::vector<uint32_t> lists = select_lists_cpu(qdesc64, nprobe);

//     // -------- Phase 1: 收集需要做 desc 的 groups（先做 geo filter）--------
//     std::vector<DescJob> jobs;
//     jobs.reserve(lists.size() * 4); // 粗略估

//     for (uint32_t lid : lists) {
//         const auto& bp = place_.bucket.at(lid);
//         const uint32_t groups_total = bp.desc_groups;
//         const uint32_t groups = (max_groups_per_list == 0u)
//             ? groups_total
//             : std::min<uint32_t>(groups_total, max_groups_per_list);

//         for (uint32_t g = 0; g < groups; ++g) {
//             MaskRow geo_xy{};
//             geo_eq_masks_xy(lid, g, qgx, qgy, geo_xy);
//             if (!mask_any(geo_xy)) continue;

//             uint16_t bank=0, mat=0, array=0;
//             map_desc_bucket_group_to_region(place_, lid, g, bank, mat, array);

//             jobs.push_back(DescJob{lid, g, bank, mat, array, geo_xy});
//         }
//     }

//     if (jobs.empty()) return best;

//     // -------- Phase 2: 依 (bank,mat) 分組，對同一組用 array_mask 做 lockstep OR --------
//     std::sort(jobs.begin(), jobs.end(), [](const DescJob& a, const DescJob& b){
//         if (a.bank != b.bank) return a.bank < b.bank;
//         if (a.mat  != b.mat)  return a.mat  < b.mat;
//         return a.array < b.array;
//     });

//     // 對每個 job 建一個 Planes（最後用來 lane_value）
//     std::vector<Planes> planes_of_job(jobs.size());
//     for (auto& p : planes_of_job) p.clear();

//     std::vector<uint16_t> rows;
//     rows.reserve(kDescBits);

//     // 走訪每一個 (bank,mat) group
//     for (size_t base = 0; base < jobs.size(); ) {
//         const uint16_t bank = jobs[base].bank;
//         const uint16_t mat  = jobs[base].mat;

//         size_t end = base;
//         uint32_t array_mask = 0;
//         while (end < jobs.size() && jobs[end].bank == bank && jobs[end].mat == mat) {
//             array_mask |= CimModule::Mask::array(jobs[end].array);
//             ++end;
//         }

//         // 對這個 (bank,mat) 做 64 個 dim 的 OR（一次 OR 覆蓋多個 arrays）
//         for (int d = 0; d < kDescDims; ++d) {
//             rows.clear();
//             const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);

//             for (int b = 0; b < kDescBits; ++b) {
//                 const int qbit = (qv >> b) & 1u;
//                 const uint16_t r = (qbit != 0) ? desc_inv_row(d, b) : desc_true_row(d, b);
//                 rows.push_back(r);
//             }

//             cim_.OR(rows,
//                     0xffu,
//                     CimModule::Mask::bank(bank),
//                     CimModule::Mask::colsAll(),
//                     /*temp_row=*/0,
//                     CimModule::Mask::mat(mat),
//                     array_mask);

//             // OR 後：每個 array 各自把 temp 讀回來，更新自己的 planes
//             for (size_t i = base; i < end; ++i) {
//                 MaskRow tmp{};
//                 cim_.copy_temp_to_cpu(tmp.data(), bank, mat, jobs[i].array, 0, kRowBytes);
//                 planes_of_job[i].add_mask(tmp);
//             }
//         }

//         // 有了 planes 後，逐 job 做 lane 掃描（沿用你原本的邏輯）
//         for (size_t i = base; i < end; ++i) {
//             const auto& jb = jobs[i];
//             const auto& bp = place_.bucket.at(jb.lid);

//             const uint32_t base_off = bp.posting_off + jb.g * static_cast<uint32_t>(kLanesPerGroup);
//             const uint32_t remain   = bp.map_count > jb.g * static_cast<uint32_t>(kLanesPerGroup)
//                                     ? (bp.map_count - jb.g * static_cast<uint32_t>(kLanesPerGroup))
//                                     : 0u;
//             const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

//             for (uint32_t lane = 0; lane < valid; ++lane) {
//                 if (lane_get_bit(jb.geo_xy.data(), static_cast<int>(lane)) == 0u) continue;

//                 const int32_t mid = map_.postings.map_ids.at(static_cast<size_t>(base_off + lane));
//                 const uint32_t mis = planes_of_job[i].lane_value(static_cast<int>(lane));

//                 if (mis < best.best_mismatch || (mis == best.best_mismatch && mid < best.best_map_id)) {
//                     best.best_mismatch = static_cast<uint8_t>(mis);
//                     best.best_map_id = mid;
//                 }
//             }
//         }

//         base = end;
//     }

//     return best;
// }

std::vector<MatchResult>
IvfMatcher::match_one_step(uint8_t qgx, uint8_t qgy,
                           const uint8_t* desc_ptr, size_t m,
                           uint32_t nprobe, uint32_t max_groups_per_list)
{
    // 固定 16 banks (BANK_BITS=4)
    static constexpr uint16_t kNumBanks = 16;

    std::vector<MatchResult> results(m);

    // ---- per-query: jobs + planes ----
    std::vector<std::vector<DescJob>> jobs_per_q(m);
    std::vector<std::vector<Planes>>  planes_per_q(m);

    // ============ Phase 1: 建 jobs（含 geo filter）============
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
                MaskRow geo_xy{};
                // no geo
                geo_xy.fill(0xFF);
                // with geo
                // geo_eq_masks_xy(lid, g, qgx, qgy, geo_xy);
                // if (!mask_any(geo_xy)) continue;

                uint16_t bank=0, mat=0, array=0;
                map_desc_bucket_group_to_region(place_, lid, g, bank, mat, array);

                jobs.push_back(DescJob{
                    static_cast<uint32_t>(qi), lid, g, bank, mat, array, geo_xy
                });
            }
        }

        // planes size 跟 jobs 一樣
        planes_per_q[qi].resize(jobs.size());
        for (auto& p : planes_per_q[qi]) p.clear();
    }

    // ============ Phase 2: 對每個 query 形成 MatWork，丟進 bank queue ============
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

        // 建立每段 (bank,mat) 的 work
        for (size_t base = 0; base < jobs.size(); ) {
            const uint16_t bank = jobs[base].bank;
            const uint16_t mat  = jobs[base].mat;

            size_t end = base;
            uint32_t array_mask = 0;
            while (end < jobs.size() && jobs[end].bank == bank && jobs[end].mat == mat) {
                array_mask |= CimModule::Mask::array(jobs[end].array);
                ++end;
            }

            // bank 必須在 0..15
            if (bank >= kNumBanks) {
                throw std::runtime_error("match_one_step: bank out of range (expect 0..15)");
            }

            bank_q[bank].push_back(MatWork{
                static_cast<uint32_t>(qi),
                bank, mat,
                array_mask,
                base, end,
                0
            });
            total_work_items++;

            base = end;
        }
    }

    if (total_work_items == 0) {
        return results; // 全都沒有通過 geo filter
    }

    // 共用 rows buffer（每次 issue 會重建）
    std::vector<uint16_t> rows;
    rows.reserve(kDescBits);

    // ============ Phase 3: Round-robin issue（bank-level interleaving, Route-1 block temp read）============
    //
    // Route-1: 對每個 MatWork (=同一個 qid + bank + mat + array_mask + job range)
    //   1) 連續 issue 64 dims，把每個 dim 的 OR 結果寫到 temp 的連續 64 rows
    //      temp_row = kDescTempBase + d
    //   2) 對這個 work 範圍內出現的每個 array：一次 copy_temp_block_to_cpu 讀回 64 rows
    //   3) 把這 64 rows 餵給該 array 的所有 jobs 對應的 planes（Planes::add_mask_bytes）
    //   4) 做 lane scan 更新 results[qid]
    //
    // 注意：你要先在檔案某處定義 kDescTempBase（避免和 geo temp_row=0 撞）
    // 例如：static constexpr uint16_t kDescTempBase = 64;

    static constexpr uint16_t kDescTempBase = 64;

    // 共用：把 jobs[i].array 分組用的暫存（避免每次 new 太多）
    std::vector<uint16_t> uniq_arrays;
    uniq_arrays.reserve(32);

    // 每個 array 對應哪些 job indices（用平行陣列存，避免 unordered_map）
    std::vector<uint16_t> arr_ids;
    std::vector<std::vector<size_t>> arr_jobs;

    // 64 rows block buffer（每 row kRowBytes）
    std::vector<uint8_t> block64(static_cast<size_t>(kDescDims) * kRowBytes);

    // Round-robin：一次處理一個 MatWork（完成 64 dim + 1 次 block copy）
    while (total_work_items > 0) {
        for (uint16_t bank = 0; bank < kNumBanks; ++bank) {
            if (bank_q[bank].empty()) continue;

            MatWork w = bank_q[bank].front();
            bank_q[bank].pop_front();

            const uint32_t qid = w.qid;
            const uint8_t* qdesc64 = desc_ptr + static_cast<size_t>(qid) * 64u;

            auto& jobs   = jobs_per_q[qid];
            auto& planes = planes_per_q[qid];

            // ---- (A) issue 64 dims：每個 dim 寫到 temp_row = kDescTempBase + d ----
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

            // ---- (B) 建立「array -> jobs indices」分組（只看這個 work 的 job range）----
            uniq_arrays.clear();
            arr_ids.clear();
            arr_jobs.clear();

            // 先收 uniq arrays（保持小規模，O(n^2) ok；你每個 work 通常不會太多 jobs）
            for (size_t ji = w.job_begin; ji < w.job_end; ++ji) {
                const uint16_t a = jobs[ji].array;
                bool seen = false;
                for (uint16_t x : uniq_arrays) {
                    if (x == a) { seen = true; break; }
                }
                if (!seen) uniq_arrays.push_back(a);
            }

            arr_ids = uniq_arrays;
            arr_jobs.resize(arr_ids.size());

            for (size_t ji = w.job_begin; ji < w.job_end; ++ji) {
                const uint16_t a = jobs[ji].array;
                for (size_t k = 0; k < arr_ids.size(); ++k) {
                    if (arr_ids[k] == a) {
                        arr_jobs[k].push_back(ji);
                        break;
                    }
                }
            }

            // ---- (C) per array：一次 block copy 64 rows，餵給該 array 的所有 planes ----
            for (size_t k = 0; k < arr_ids.size(); ++k) {
                const uint16_t array = arr_ids[k];

                // 把 temp 中連續 64 rows 搬回 CPU：block64 佈局 = [row0][row1]...[row63]
                cim_.copy_temp_block_to_cpu(block64.data(),
                                            w.bank, w.mat, array,
                                            /*start_row=*/kDescTempBase,
                                            /*num_rows=*/kDescDims);
                
                for (int d = 0; d < kDescDims; ++d) {
                    if (qdesc64[d] == 0) {
                        // block64 的佈局假設是 Row-Major: [Row0][Row1]...
                        // 將該 Row 的所有 bytes (kRowBytes) 設為 0
                        std::memset(block64.data() + static_cast<size_t>(d) * kRowBytes, 
                                    0, 
                                    kRowBytes);
                    }
                }
                // 對這個 array 的每個 job，把 64 rows 加到 planes[job]
                // 假設你已經實作：Planes::add_mask_bytes(const uint8_t* rows64, size_t strideBytes)
                // - rows64 指向 row0 的起始位址
                // - strideBytes = kRowBytes
                for (size_t ji : arr_jobs[k]) {
                    planes[ji].add_mask_block_bytes(block64.data(), kRowBytes, kDescDims);
                }
            }

            // ---- (D) lane scan（這個 MatWork 完成後即可掃描）----
            for (size_t ji = w.job_begin; ji < w.job_end; ++ji) {
                const auto& jb = jobs[ji];
                const auto& bp = place_.bucket.at(jb.lid);

                const uint32_t base_off = bp.posting_off + jb.g * static_cast<uint32_t>(kLanesPerGroup);
                const uint32_t remain   = (bp.map_count > jb.g * static_cast<uint32_t>(kLanesPerGroup))
                                        ? (bp.map_count - jb.g * static_cast<uint32_t>(kLanesPerGroup))
                                        : 0u;
                const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

                for (uint32_t lane = 0; lane < valid; ++lane) {
                    // if (lane_get_bit(jb.geo_xy.data(), static_cast<int>(lane)) == 0u) continue;

                    const int32_t mid = map_.postings.map_ids.at(static_cast<size_t>(base_off + lane));
                    const uint32_t mis = planes[ji].lane_value(static_cast<int>(lane));

                    auto& best = results[qid];
                    if (mis < best.best_mismatch || (mis == best.best_mismatch && mid < best.best_map_id)) {
                        best.best_mismatch = static_cast<uint8_t>(mis);
                        best.best_map_id = mid;
                    }
                }
            }

            // 這個 MatWork 完成（不再 push 回 queue）
            total_work_items--;
        }
    }

    return results;
}

// MatchResult IvfMatcher::match_one_query_desc(uint8_t qgx, uint8_t qgy, const uint8_t* qdesc64, uint32_t nprobe, uint32_t max_groups_per_list)
// {
//     MatchResult best;

//     const uint32_t K = map_.postings.nlist;
//     if (K == 0) return best;

//     const std::vector<uint32_t> lists = select_lists_cpu(qdesc64, nprobe);

//     for (uint32_t lid : lists) {
//         const auto& bp = place_.bucket.at(lid);
//         const uint32_t groups_total = bp.desc_groups;
//         const uint32_t groups = (max_groups_per_list == 0u) ? groups_total : std::min<uint32_t>(groups_total, max_groups_per_list);

//         for (uint32_t g = 0; g < groups; ++g) {
//             MaskRow geo_xy{};
//             geo_eq_masks_xy(lid, g, qgx, qgy, geo_xy);
//             if (!mask_any(geo_xy)) continue;

//             Planes planes;
//             desc_mismatch_planes(lid, g, qdesc64, planes);

//             const uint32_t base_off = bp.posting_off + g * static_cast<uint32_t>(kLanesPerGroup);
//             const uint32_t remain   = bp.map_count > g * static_cast<uint32_t>(kLanesPerGroup)
//                                     ? (bp.map_count - g * static_cast<uint32_t>(kLanesPerGroup))
//                                     : 0u;
//             const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

//             for (uint32_t lane = 0; lane < valid; ++lane) {
//                 if (lane_get_bit(geo_xy.data(), static_cast<int>(lane)) == 0u) continue;

//                 const int32_t mid = map_.postings.map_ids.at(static_cast<size_t>(base_off + lane));
//                 const uint32_t mis = planes.lane_value(static_cast<int>(lane));

//                 if (mis < best.best_mismatch || (mis == best.best_mismatch && mid < best.best_map_id)) {
//                     best.best_mismatch = static_cast<uint8_t>(mis);
//                     best.best_map_id = mid;
//                 }
//             }
//         }
//     }

//     return best;
// }

} // namespace msim