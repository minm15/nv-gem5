#include "roi_match.hpp"
#include "cim_api.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace lab {

static inline uint8_t clamp_u8_int(int v)
{
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

static inline int ilog2_floor_u32(uint32_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return 31 - __builtin_clz(x);
#else
    int r = 0;
    while (x >>= 1) ++r;
    return r;
#endif
}

static inline int ilog2_ceil_u32(uint32_t x)
{
    const int f = ilog2_floor_u32(x);
    const uint32_t p = 1u << f;
    return (p == x) ? f : (f + 1);
}

static inline void build_floor_ceil_p2m1(std::array<uint8_t, 256>& floor_p2m1,
                                         std::array<uint8_t, 256>& ceil_p2m1)
{
    for (int v = 0; v <= 255; ++v) {
        const uint32_t vp1 = static_cast<uint32_t>(v) + 1u;

        const int tf = ilog2_floor_u32(vp1);
        uint32_t flo = (1u << tf) - 1u;
        if (flo > 255u) flo = 255u;
        floor_p2m1[static_cast<size_t>(v)] = static_cast<uint8_t>(flo);

        const int tc = ilog2_ceil_u32(vp1);
        uint32_t cei = (1u << tc) - 1u;
        if (cei > 255u) cei = 255u;
        ceil_p2m1[static_cast<size_t>(v)] = static_cast<uint8_t>(cei);
    }
}

static inline int t_from_thr_p2m1(uint8_t thr)
{
    if (thr == 255u) return 8;
    const uint32_t x = static_cast<uint32_t>(thr) + 1u;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#else
    int t = 0;
    uint32_t y = x;
    while ((y & 1u) == 0u) { y >>= 1; ++t; }
    return t;
#endif
}

static constexpr int kLanesWords = 8; // 512 lanes / 64 = 8 words
static constexpr int kLanesPerGroup = NvmWriter::kLanesPerGroup;
static constexpr int kRowBytes = NvmWriter::kRowBytes;

static inline void load_mask_u64(const uint8_t* mask64B, uint64_t out[kLanesWords])
{
    std::memcpy(out, mask64B, 64);
}

static inline bool words_any_nonzero(const uint64_t w[kLanesWords])
{
    uint64_t acc = 0;
    for (int i = 0; i < kLanesWords; ++i) acc |= w[i];
    return acc != 0;
}

static inline int ctz64(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while ((x & 1ull) == 0ull) { x >>= 1; ++n; }
    return n;
#endif
}

static constexpr int kCounterBits =
    (kDims + 1 <= 2) ? 1 :
    (kDims + 1 <= 4) ? 2 :
    (kDims + 1 <= 8) ? 3 :
    (kDims + 1 <= 16) ? 4 :
    (kDims + 1 <= 32) ? 5 :
    (kDims + 1 <= 64) ? 6 :
    (kDims + 1 <= 128) ? 7 : 8;

using Planes = std::array<std::array<uint64_t, kLanesWords>, kCounterBits>;

static inline void planes_zero(Planes& p)
{
    for (int k = 0; k < kCounterBits; ++k) {
        for (int w = 0; w < kLanesWords; ++w) p[static_cast<size_t>(k)][static_cast<size_t>(w)] = 0;
    }
}

static inline void planes_add_mask(Planes& p, const uint64_t mask_words[kLanesWords])
{
    uint64_t carry[kLanesWords];
    for (int w = 0; w < kLanesWords; ++w) carry[w] = mask_words[w];

    for (int k = 0; k < kCounterBits; ++k) {
        for (int w = 0; w < kLanesWords; ++w) {
            const uint64_t a = p[static_cast<size_t>(k)][static_cast<size_t>(w)];
            const uint64_t c = carry[w];
            p[static_cast<size_t>(k)][static_cast<size_t>(w)] = a ^ c;
            carry[w] = a & c;
        }
        uint64_t acc = 0;
        for (int w = 0; w < kLanesWords; ++w) acc |= carry[w];
        if (acc == 0) break;
    }
}

static inline int planes_argmin_lane_masked(const Planes& p, const uint64_t cand_init[kLanesWords])
{
    uint64_t cand[kLanesWords];
    for (int w = 0; w < kLanesWords; ++w) cand[w] = cand_init[w];

    if (!words_any_nonzero(cand)) return -1;

    for (int k = kCounterBits - 1; k >= 0; --k) {
        uint64_t zeros[kLanesWords];
        for (int w = 0; w < kLanesWords; ++w) {
            zeros[w] = cand[w] & ~p[static_cast<size_t>(k)][static_cast<size_t>(w)];
        }
        if (words_any_nonzero(zeros)) {
            for (int w = 0; w < kLanesWords; ++w) cand[w] = zeros[w];
        } else {
            for (int w = 0; w < kLanesWords; ++w) cand[w] = cand[w] & p[static_cast<size_t>(k)][static_cast<size_t>(w)];
        }
        if (!words_any_nonzero(cand)) return -1;
    }

    for (int w = 0; w < kLanesWords; ++w) {
        const uint64_t x = cand[w];
        if (x) return w * 64 + ctz64(x);
    }
    return -1;
}

static inline uint32_t planes_get_lane_value(const Planes& p, int lane)
{
    const int w = lane >> 6;
    const int b = lane & 63;
    uint32_t v = 0;
    for (int k = 0; k < kCounterBits; ++k) {
        const uint64_t word = p[static_cast<size_t>(k)][static_cast<size_t>(w)];
        v |= static_cast<uint32_t>(((word >> b) & 1ull) << k);
    }
    return v;
}

static inline uint16_t desc_orig_row(int d, int b)
{
    return static_cast<uint16_t>(d * (2 * kValueBits) + b);
}

static inline uint16_t desc_inv_row(int d, int b)
{
    return static_cast<uint16_t>(d * (2 * kValueBits) + kValueBits + b);
}

static inline void group_to_desc_bank_array(const Placement& place,
                                            uint32_t group,
                                            uint16_t& bank,
                                            uint16_t& array)
{
    const uint32_t bank_sel = group >> 8; // /256
    array = static_cast<uint16_t>(group & 0xFFu);
    switch (bank_sel) {
        case 0: bank = place.bank_desc0; break;
        case 1: bank = place.bank_desc1; break;
        case 2: bank = place.bank_desc2; break;
        case 3: bank = place.bank_desc3; break;
        default: bank = place.bank_desc0; break;
    }
}

static inline void group_to_pos_array_rowbase(uint32_t group,
                                              uint16_t& array,
                                              uint16_t& row_base)
{
    array = static_cast<uint16_t>(group / NvmWriter::kGroupsPerPosArray);
    const uint16_t block = static_cast<uint16_t>(group % NvmWriter::kGroupsPerPosArray);
    row_base = static_cast<uint16_t>(block * NvmWriter::kPosRowsPerGroup);
}

static inline void or_many_rows_batched(CimModule& cim,
                                        uint16_t bank,
                                        uint16_t mat,
                                        uint16_t array,
                                        const std::vector<uint16_t>& rows_all,
                                        std::array<uint8_t, kRowBytes>& out_or)
{
    out_or.fill(0);

    alignas(64) std::array<uint8_t, kRowBytes> tmp{};
    std::vector<uint16_t> chunk;
    chunk.reserve(4);

    const size_t n = rows_all.size();
    size_t i = 0;
    while (i < n) {
        chunk.clear();
        const size_t take = std::min<size_t>(4, n - i);
        for (size_t j = 0; j < take; ++j) chunk.push_back(rows_all[i + j]);

        cim.OR(chunk,
               0xffu,
               CimModule::Mask::bank(bank),
               CimModule::Mask::colsAll(),
               0,
               CimModule::Mask::mat(mat),
               CimModule::Mask::array(array));

        cim.copy_temp_to_cpu(tmp.data(), bank, mat, array, 0, kRowBytes);

        for (size_t k = 0; k < kRowBytes; ++k) out_or[k] = static_cast<uint8_t>(out_or[k] | tmp[k]);
        i += take;
    }
}

static inline void cim_geo_leq_mask_for_group(CimModule& cim,
                                             const Placement& place,
                                             uint32_t group,
                                             bool is_x,
                                             int t,
                                             std::array<uint8_t, kRowBytes>& out_leq)
{
    if (t >= 8) {
        out_leq.fill(0xFF);
        return;
    }

    uint16_t pos_array = 0, row_base = 0;
    group_to_pos_array_rowbase(group, pos_array, row_base);

    std::vector<uint16_t> rows;
    rows.reserve(8);

    const int last_row = 7 - t;          // rows 0..last_row are MSB..t bits
    const uint16_t base = static_cast<uint16_t>(row_base + (is_x ? 0 : 8));

    for (int r = 0; r <= last_row; ++r) rows.push_back(static_cast<uint16_t>(base + static_cast<uint16_t>(r)));

    std::array<uint8_t, kRowBytes> violate{};
    or_many_rows_batched(cim, place.bank_pos, place.mat, pos_array, rows, violate);

    for (size_t i = 0; i < kRowBytes; ++i) out_leq[i] = static_cast<uint8_t>(~violate[i]);
}

RoiResults run_roi_match(CimModule& cim, const Workload& wl, const Placement& place)
{
    std::array<uint8_t, 256> floor_p2m1{};
    std::array<uint8_t, 256> ceil_p2m1{};
    build_floor_ceil_p2m1(floor_p2m1, ceil_p2m1);

    alignas(64) std::array<uint8_t, kRowBytes> tempRow{};
    std::vector<uint16_t> rows;
    rows.reserve(kValueBits);

    alignas(64) std::array<uint8_t, kRowBytes> mask_x_hi{};
    alignas(64) std::array<uint8_t, kRowBytes> mask_x_lo{};
    alignas(64) std::array<uint8_t, kRowBytes> mask_y_hi{};
    alignas(64) std::array<uint8_t, kRowBytes> mask_y_lo{};
    alignas(64) std::array<uint8_t, kRowBytes> geo_mask_bytes{};

    RoiResults out;
    out.per_query.resize(static_cast<size_t>(wl.Q));

    const uint32_t groups = static_cast<uint32_t>(wl.M / kLanesPerGroup);

    for (size_t qi = 0; qi < wl.Q; ++qi) {
        const uint8_t rx = wl.query_rx[qi];
        const uint8_t ry = wl.query_ry[qi];
        const int range = static_cast<int>(wl.query_range[qi]);

        const uint8_t x_lo = clamp_u8_int(static_cast<int>(rx) - range);
        const uint8_t x_hi = clamp_u8_int(static_cast<int>(rx) + range);
        const uint8_t y_lo = clamp_u8_int(static_cast<int>(ry) - range);
        const uint8_t y_hi = clamp_u8_int(static_cast<int>(ry) + range);

        const uint8_t thr_x_lo = floor_p2m1[static_cast<size_t>(x_lo)];
        const uint8_t thr_x_hi = ceil_p2m1[static_cast<size_t>(x_hi)];
        const uint8_t thr_y_lo = floor_p2m1[static_cast<size_t>(y_lo)];
        const uint8_t thr_y_hi = ceil_p2m1[static_cast<size_t>(y_hi)];

        const int t_x_lo = t_from_thr_p2m1(thr_x_lo);
        const int t_x_hi = t_from_thr_p2m1(thr_x_hi);
        const int t_y_lo = t_from_thr_p2m1(thr_y_lo);
        const int t_y_hi = t_from_thr_p2m1(thr_y_hi);

        int64_t best_global = -1;
        uint32_t best_mis = 0xFFFFFFFFu;

        for (uint32_t g = 0; g < groups; ++g) {
            // Geo gating for this group (bank_pos, pos_array depends on g).
            cim_geo_leq_mask_for_group(cim, place, g, true, t_x_hi, mask_x_hi);
            cim_geo_leq_mask_for_group(cim, place, g, true, t_x_lo, mask_x_lo);
            cim_geo_leq_mask_for_group(cim, place, g, false, t_y_hi, mask_y_hi);
            cim_geo_leq_mask_for_group(cim, place, g, false, t_y_lo, mask_y_lo);

            for (size_t i = 0; i < kRowBytes; ++i) {
                const uint8_t x_gt_lo = static_cast<uint8_t>(~mask_x_lo[i]);
                const uint8_t y_gt_lo = static_cast<uint8_t>(~mask_y_lo[i]);
                const uint8_t x_ok = static_cast<uint8_t>(mask_x_hi[i] & x_gt_lo);
                const uint8_t y_ok = static_cast<uint8_t>(mask_y_hi[i] & y_gt_lo);
                geo_mask_bytes[i] = static_cast<uint8_t>(x_ok & y_ok);
            }

            uint64_t geo_words[kLanesWords];
            load_mask_u64(geo_mask_bytes.data(), geo_words);
            if (!words_any_nonzero(geo_words)) continue;
            //for (int w = 0; w < kLanesWords; ++w) geo_words[w] = ~0ull; // no-gate: all lanes valid

            // Desc bank/array for this group.
            uint16_t bank = 0, array = 0;
            group_to_desc_bank_array(place, g, bank, array);

            Planes planes{};
            planes_zero(planes);

            for (int d = 0; d < kDims; ++d) {
                rows.clear();
                const uint8_t qv = wl.query_desc[qi][static_cast<size_t>(d)] & 0x0Fu;

                for (int b = 0; b < kValueBits; ++b) {
                    const int qbit = (qv >> b) & 1u;
                    const uint16_t r = static_cast<uint16_t>(qbit ? desc_inv_row(d, b) : desc_orig_row(d, b));
                    rows.push_back(r);
                }

                cim.OR(rows,
                       0xffu,
                       CimModule::Mask::bank(bank),
                       CimModule::Mask::colsAll(),
                       0,
                       CimModule::Mask::mat(place.mat),
                       CimModule::Mask::array(array));

                cim.copy_temp_to_cpu(tempRow.data(), bank, place.mat, array, 0, kRowBytes);

                uint64_t mask_words[kLanesWords];
                load_mask_u64(tempRow.data(), mask_words);
                planes_add_mask(planes, mask_words);
            }

            const int lane = planes_argmin_lane_masked(planes, geo_words);
            if (lane < 0) continue;

            const uint32_t mis = planes_get_lane_value(planes, lane);
            const int64_t cand_global = static_cast<int64_t>(g) * kLanesPerGroup + lane;

            if (mis < best_mis || (mis == best_mis && (best_global < 0 || cand_global < best_global))) {
                best_mis = mis;
                best_global = cand_global;
                if (best_mis == 0) break;
            }
        }

        out.per_query[qi].best_global = best_global;
        out.per_query[qi].best_mismatch = best_mis;
    }

    return out;
}

} // namespace lab
