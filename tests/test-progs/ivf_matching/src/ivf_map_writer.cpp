#include "ivf_map_writer.hpp"
#include "layout.hpp"
#include "msim_config.hpp"
#include <array>
#include <algorithm>
#include <stdexcept>

namespace msim {

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

static inline uint8_t q4_bit(uint8_t q4, int b)
{
    return static_cast<uint8_t>((q4 >> b) & 1u);
}

IvfMapWriter::IvfMapWriter(CimModule& cim, const IvfPlacement& place)
    : cim_(cim), place_(place)
{
}

void IvfMapWriter::write_desc_bucket_group(const IvfMapBins& map, uint32_t bucket_id, uint32_t group_in_bucket)
{
    uint16_t bank = 0, mat = 0, array = 0;
    map_desc_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array);

    const auto& bp = place_.bucket.at(bucket_id);
    const uint32_t base_off = bp.posting_off + group_in_bucket * static_cast<uint32_t>(kLanesPerGroup);
    const uint32_t remain   = bp.map_count > group_in_bucket * static_cast<uint32_t>(kLanesPerGroup)
                            ? (bp.map_count - group_in_bucket * static_cast<uint32_t>(kLanesPerGroup))
                            : 0u;
    const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

    std::array<uint8_t, kRowBytes> row{};

    for (int d = 0; d < kDescDims; ++d) {
        for (int b = 0; b < kDescBits; ++b) {
            const uint16_t r_true = desc_true_row(d, b);
            const uint16_t r_inv  = desc_inv_row(d, b);

            // true row
            row.fill(0);
            for (uint32_t lane = 0; lane < static_cast<uint32_t>(kLanesPerGroup); ++lane) {
                uint8_t bitv = 1u;
                if (lane < valid) {
                    const int32_t mid = map.postings.map_ids.at(static_cast<size_t>(base_off + lane));
                    const uint8_t q4  = map.desc_q4.at(static_cast<size_t>(mid) * 64u + static_cast<size_t>(d));
                    bitv = q4_bit(q4, b);
                }
                lane_set_bit(row.data(), static_cast<int>(lane), bitv);
            }
            cim_.copy_to_cim(bank, mat, array, r_true, row.data(), kRowBytes);

            // inv row
            row.fill(0);
            for (uint32_t lane = 0; lane < static_cast<uint32_t>(kLanesPerGroup); ++lane) {
                uint8_t bitv = 1u;
                if (lane < valid) {
                    const int32_t mid = map.postings.map_ids.at(static_cast<size_t>(base_off + lane));
                    const uint8_t q4  = map.desc_q4.at(static_cast<size_t>(mid) * 64u + static_cast<size_t>(d));
                    bitv = static_cast<uint8_t>(1u - q4_bit(q4, b));
                }
                lane_set_bit(row.data(), static_cast<int>(lane), bitv);
            }
            cim_.copy_to_cim(bank, mat, array, r_inv, row.data(), kRowBytes);
        }
    }
}

void IvfMapWriter::write_geo_bucket_group(const IvfMapBins& map, uint32_t bucket_id, uint32_t group_in_bucket)
{
    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    map_geo_bucket_group_to_region(place_, bucket_id, group_in_bucket, bank, mat, array, row_base);

    const auto& bp = place_.bucket.at(bucket_id);
    const uint32_t base_off = bp.posting_off + group_in_bucket * static_cast<uint32_t>(kLanesPerGroup);
    const uint32_t remain   = bp.map_count > group_in_bucket * static_cast<uint32_t>(kLanesPerGroup)
                            ? (bp.map_count - group_in_bucket * static_cast<uint32_t>(kLanesPerGroup))
                            : 0u;
    const uint32_t valid    = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), remain);

    std::array<uint8_t, kRowBytes> row{};

    auto write_one_row = [&](uint16_t rid_row, bool is_x, bool high, int bit, bool inv) {
        row.fill(0);
        for (uint32_t lane = 0; lane < static_cast<uint32_t>(kLanesPerGroup); ++lane) {
            uint8_t bitv = 1u;
            if (lane < valid) {
                const int32_t mid = map.postings.map_ids.at(static_cast<size_t>(base_off + lane));
                const uint8_t gx = map.geo_grid.at(static_cast<size_t>(mid) * 2u + 0u);
                const uint8_t gy = map.geo_grid.at(static_cast<size_t>(mid) * 2u + 1u);
                const uint8_t v  = is_x ? gx : gy;

                const uint8_t nib = high ? static_cast<uint8_t>((v >> 4) & 0x0Fu) : static_cast<uint8_t>(v & 0x0Fu);
                const uint8_t bv  = static_cast<uint8_t>((nib >> bit) & 1u);
                bitv = inv ? static_cast<uint8_t>(1u - bv) : bv;
            }
            lane_set_bit(row.data(), static_cast<int>(lane), bitv);
        }
        cim_.copy_to_cim(bank, mat, array, static_cast<uint16_t>(row_base + rid_row), row.data(), kRowBytes);
    };

    for (int b = 0; b < 4; ++b) {
        write_one_row(geo_x_row_low(b, false),  true,  false, b, false);
        write_one_row(geo_x_row_low(b, true),   true,  false, b, true);
        write_one_row(geo_x_row_high(b, false), true,  true,  b, false);
        write_one_row(geo_x_row_high(b, true),  true,  true,  b, true);

        write_one_row(geo_y_row_low(b, false),  false, false, b, false);
        write_one_row(geo_y_row_low(b, true),   false, false, b, true);
        write_one_row(geo_y_row_high(b, false), false, true,  b, false);
        write_one_row(geo_y_row_high(b, true),  false, true,  b, true);
    }
}

void IvfMapWriter::write_all(const IvfMapBins& map)
{
    const uint32_t K = map.postings.nlist;
    if (K == 0) return;
    if (place_.bucket.size() != K) throw std::runtime_error("IvfMapWriter: placement K mismatch");

    for (uint32_t b = 0; b < K; ++b) {
        const uint32_t groups = place_.bucket[b].desc_groups;
        for (uint32_t g = 0; g < groups; ++g) {
            write_desc_bucket_group(map, b, g);
            write_geo_bucket_group(map, b, g);
        }
    }
}

} // namespace msim