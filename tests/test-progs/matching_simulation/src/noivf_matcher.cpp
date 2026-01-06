#include "noivf_matcher.hpp"
#include "msim_config.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace msim {

static inline uint64_t load_u64_le(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | static_cast<uint64_t>(p[i]);
    }
    return v;
}

static inline void store_u64_le(uint8_t* p, uint64_t v)
{
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<uint8_t>(v & 0xFFu);
        v >>= 8;
    }
}

void NoIvfMatcher::mask_and_inplace(MaskRow& a, const MaskRow& b)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(a[i] & b[i]);
}

void NoIvfMatcher::mask_or_inplace(MaskRow& a, const MaskRow& b)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(a[i] | b[i]);
}

void NoIvfMatcher::mask_not_inplace(MaskRow& a)
{
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(~a[i]);
}

bool NoIvfMatcher::mask_any(const MaskRow& a)
{
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != 0u) return true;
    }
    return false;
}

void NoIvfMatcher::Planes::clear()
{
    for (auto& p : b) p.fill(0);
}

// Bit-sliced increment: b[0] is LSB plane.
void NoIvfMatcher::Planes::add_mask(const MaskRow& x)
{
    MaskRow carry = x;
    for (size_t k = 0; k < b.size(); ++k) {
        MaskRow sum{};
        for (size_t i = 0; i < sum.size(); ++i) {
            sum[i] = static_cast<uint8_t>(b[k][i] ^ carry[i]);
        }
        MaskRow new_carry{};
        for (size_t i = 0; i < new_carry.size(); ++i) {
            new_carry[i] = static_cast<uint8_t>(b[k][i] & carry[i]);
        }
        b[k] = sum;
        carry = new_carry;
    }
}

uint8_t NoIvfMatcher::Planes::lane_value(int lane) const
{
    uint8_t v = 0;
    for (size_t k = 0; k < b.size(); ++k) {
        const uint8_t bit = lane_get_bit(b[k].data(), lane);
        v = static_cast<uint8_t>(v | static_cast<uint8_t>(bit << k));
    }
    return v;
}

NoIvfMatcher::NoIvfMatcher(CimModule& cim, const Placement& place, uint8_t geo_max)
    : cim_(cim), place_(place), geo_max_(geo_max)
{
}

void NoIvfMatcher::map_desc_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array) const
{
    const uint32_t arrays_per_bank = (1u << msim::kMatBits) * (1u << msim::kArrayBits); // 256
    const uint32_t bank_sel = group / arrays_per_bank;
    const uint32_t within   = group % arrays_per_bank;
    if (bank_sel >= place_.desc_bank_count) throw std::runtime_error("desc group bank_sel out of range");

    const uint32_t arrays_per_mat = (1u << msim::kArrayBits); // 16
    bank = static_cast<uint16_t>(place_.desc_bank0 + bank_sel);
    mat  = static_cast<uint16_t>(within / arrays_per_mat);
    array= static_cast<uint16_t>(within % arrays_per_mat);
}

void NoIvfMatcher::map_geo_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array, uint16_t& row_base) const
{
    const uint32_t geo_array_linear = group / static_cast<uint32_t>(msim::kGeoGroupsPerArray);
    const uint32_t group_in_array   = group % static_cast<uint32_t>(msim::kGeoGroupsPerArray);

    const uint32_t arrays_per_bank = (1u << msim::kMatBits) * (1u << msim::kArrayBits); // 256
    const uint32_t bank_sel = geo_array_linear / arrays_per_bank;
    const uint32_t within   = geo_array_linear % arrays_per_bank;
    if (bank_sel >= place_.geo_bank_count) throw std::runtime_error("geo group bank_sel out of range");

    const uint32_t arrays_per_mat = (1u << msim::kArrayBits); // 16
    bank = static_cast<uint16_t>(place_.geo_bank0 + bank_sel);
    mat  = static_cast<uint16_t>(within / arrays_per_mat);
    array= static_cast<uint16_t>(within % arrays_per_mat);

    row_base = geo_row_base_for_group_in_array(group_in_array);
}

NoIvfMatcher::MaskRow NoIvfMatcher::geo_eq_mask_axis(uint32_t group, bool is_x, uint8_t v)
{
    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    map_geo_group_to_region(group, bank, mat, array, row_base);

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
                if (!high) r = static_cast<uint16_t>(row_base + geo_x_row_low(b, inv));
                else       r = static_cast<uint16_t>(row_base + geo_x_row_high(b, inv));
            } else {
                if (!high) r = static_cast<uint16_t>(row_base + geo_y_row_low(b, inv));
                else       r = static_cast<uint16_t>(row_base + geo_y_row_high(b, inv));
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
        cim_.copy_temp_to_cpu(tmp.data(), bank, mat, array, 0, msim::kRowBytes);
        return tmp; // mismatch mask for the nibble
    };

    MaskRow mis_lo = or_nibble_mismatch(false, lo);
    MaskRow mis_hi = or_nibble_mismatch(true,  hi);

    mask_not_inplace(mis_lo); // eq_lo
    mask_not_inplace(mis_hi); // eq_hi
    mask_and_inplace(mis_lo, mis_hi); // eq_byte

    return mis_lo;
}

void NoIvfMatcher::geo_eq_masks_xy(uint32_t group, uint8_t qgx, uint8_t qgy, MaskRow& out_mask_xy)
{
    out_mask_xy.fill(0);

    auto one_xy = [&](uint8_t gx, uint8_t gy) {
        MaskRow mx = geo_eq_mask_axis(group, true, gx);
        MaskRow my = geo_eq_mask_axis(group, false, gy);
        mask_and_inplace(mx, my);
        mask_or_inplace(out_mask_xy, mx);
    };

    // 10 probes: x sweep (5) + y sweep (5). Center duplicated (still counts).
    for (int dx = -msim::kGeoProbeRadius; dx <= msim::kGeoProbeRadius; ++dx) {
        const uint8_t gx = clamp_u8(static_cast<int>(qgx) + dx, 0, geo_max_);
        one_xy(gx, qgy);
    }
    for (int dy = -msim::kGeoProbeRadius; dy <= msim::kGeoProbeRadius; ++dy) {
        const uint8_t gy = clamp_u8(static_cast<int>(qgy) + dy, 0, geo_max_);
        one_xy(qgx, gy);
    }
}

void NoIvfMatcher::desc_mismatch_planes(uint32_t group, const uint8_t* qdesc64, Planes& out_planes)
{
    out_planes.clear();

    uint16_t bank = 0, mat = 0, array = 0;
    map_desc_group_to_region(group, bank, mat, array);

    std::vector<uint16_t> rows;
    rows.reserve(msim::kDescBits);

    for (int d = 0; d < msim::kDescDims; ++d) {
        rows.clear();
        const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);

        for (int b = 0; b < msim::kDescBits; ++b) {
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
        cim_.copy_temp_to_cpu(tmp.data(), bank, mat, array, 0, msim::kRowBytes);
        // dump_hex_prefix(tmp, 64);

        out_planes.add_mask(tmp); // add mismatch bit for this dim
    }
}

MatchResult NoIvfMatcher::match_one_query_desc(uint8_t qgx, uint8_t qgy, const uint8_t* qdesc64, uint32_t map_N, uint32_t max_groups)
{
    MatchResult best;

    const uint32_t total_groups = (map_N + static_cast<uint32_t>(msim::kLanesPerGroup) - 1u) / static_cast<uint32_t>(msim::kLanesPerGroup);
    const uint32_t groups = (max_groups == 0u) ? total_groups : std::min<uint32_t>(total_groups, max_groups);

    for (uint32_t g = 0; g < groups; ++g) {
        MaskRow geo_xy{};
        geo_eq_masks_xy(g, qgx, qgy, geo_xy);
        if (!mask_any(geo_xy)) continue;

        Planes planes;
        desc_mismatch_planes(g, qdesc64, planes);

        const uint32_t base = g * static_cast<uint32_t>(msim::kLanesPerGroup);
        const uint32_t valid = std::min<uint32_t>(static_cast<uint32_t>(msim::kLanesPerGroup), map_N - base);

        for (uint32_t lane = 0; lane < valid; ++lane) {
            if (lane_get_bit(geo_xy.data(), static_cast<int>(lane)) == 0u) continue;

            const uint32_t mid = base + lane;
            const uint32_t mis = planes.lane_value(static_cast<int>(lane));

            if (mis < best.best_mismatch || (mis == best.best_mismatch && static_cast<int32_t>(mid) < best.best_map_id)) {
                best.best_mismatch = mis;
                best.best_map_id = static_cast<int32_t>(mid);
            }
        }
    }

    return best;
}

} // namespace msim