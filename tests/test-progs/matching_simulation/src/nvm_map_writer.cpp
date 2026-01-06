#include "nvm_map_writer.hpp"
#include "msim_config.hpp"
#include <algorithm>
#include <stdexcept>

namespace msim {

NvmMapWriter::NvmMapWriter(CimModule& cim, const Placement& place)
    : cim_(cim), place_(place)
{
}

void NvmMapWriter::map_desc_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array) const
{
    const uint32_t arrays_per_bank = (1u << kMatBits) * (1u << kArrayBits); // 256
    const uint32_t bank_sel = group / arrays_per_bank;
    const uint32_t within   = group % arrays_per_bank;

    if (bank_sel >= place_.desc_bank_count) {
        throw std::runtime_error("Desc group exceeds allocated desc banks");
    }

    const uint32_t mats_per_bank = (1u << kMatBits);   // 16
    const uint32_t arrays_per_mat = (1u << kArrayBits); // 16

    bank = static_cast<uint16_t>(place_.desc_bank0 + bank_sel);
    mat  = static_cast<uint16_t>(within / arrays_per_mat);
    array= static_cast<uint16_t>(within % arrays_per_mat);

    (void)mats_per_bank;
}

void NvmMapWriter::map_geo_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array, uint16_t& row_base) const
{
    const uint32_t geo_array_linear = group / static_cast<uint32_t>(kGeoGroupsPerArray); // 16 groups per geo array
    const uint32_t group_in_array   = group % static_cast<uint32_t>(kGeoGroupsPerArray);

    const uint32_t arrays_per_bank = (1u << kMatBits) * (1u << kArrayBits); // 256
    const uint32_t bank_sel = geo_array_linear / arrays_per_bank;
    const uint32_t within   = geo_array_linear % arrays_per_bank;

    if (bank_sel >= place_.geo_bank_count) {
        throw std::runtime_error("Geo group exceeds allocated geo banks");
    }

    const uint32_t arrays_per_mat = (1u << kArrayBits); // 16
    bank = static_cast<uint16_t>(place_.geo_bank0 + bank_sel);
    mat  = static_cast<uint16_t>(within / arrays_per_mat);
    array= static_cast<uint16_t>(within % arrays_per_mat);

    row_base = geo_row_base_for_group_in_array(group_in_array);
}

void NvmMapWriter::write_desc_group(const MapBins& map, uint32_t group)
{
    uint16_t bank = 0, mat = 0, array = 0;
    map_desc_group_to_region(group, bank, mat, array);

    const uint32_t base = group * static_cast<uint32_t>(kLanesPerGroup);
    const uint32_t valid = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), map.N - base);

    std::array<uint8_t, kRowBytes> row{};

    for (int d = 0; d < kDescDims; ++d) {
        for (int b = 0; b < kDescBits; ++b) {
            const uint16_t r_true = desc_true_row(d, b);
            const uint16_t r_inv  = desc_inv_row(d, b);

            row.fill(0);
            for (uint32_t lane = 0; lane < static_cast<uint32_t>(kLanesPerGroup); ++lane) {
                uint8_t bitv = 1u;
                if (lane < valid) {
                    const uint32_t mid = base + lane;
                    const uint8_t q4 = map.desc_q4[static_cast<size_t>(mid) * 64 + static_cast<size_t>(d)];
                    bitv = q4_bit(q4, b);
                }
                lane_set_bit(row.data(), static_cast<int>(lane), bitv);
            }
            cim_.copy_to_cim(bank, mat, array, r_true, row.data(), kRowBytes);

            row.fill(0);
            for (uint32_t lane = 0; lane < static_cast<uint32_t>(kLanesPerGroup); ++lane) {
                uint8_t bitv = 1u;
                if (lane < valid) {
                    const uint32_t mid = base + lane;
                    const uint8_t q4 = map.desc_q4[static_cast<size_t>(mid) * 64 + static_cast<size_t>(d)];
                    bitv = static_cast<uint8_t>(1u - q4_bit(q4, b));
                }
                lane_set_bit(row.data(), static_cast<int>(lane), bitv);
            }
            cim_.copy_to_cim(bank, mat, array, r_inv, row.data(), kRowBytes);
        }
    }
}

void NvmMapWriter::write_geo_group(const MapBins& map, uint32_t group)
{
    uint16_t bank = 0, mat = 0, array = 0, row_base = 0;
    map_geo_group_to_region(group, bank, mat, array, row_base);

    const uint32_t base = group * static_cast<uint32_t>(kLanesPerGroup);
    const uint32_t valid = std::min<uint32_t>(static_cast<uint32_t>(kLanesPerGroup), map.N - base);

    std::array<uint8_t, kRowBytes> row{};

    auto write_one_row = [&](uint16_t rid_row, bool is_x, bool high, int bit, bool inv) {
        row.fill(0);
        for (uint32_t lane = 0; lane < static_cast<uint32_t>(kLanesPerGroup); ++lane) {
            uint8_t bitv = 1u;
            if (lane < valid) {
                const uint32_t mid = base + lane;
                const uint8_t gx = map.geo_grid[static_cast<size_t>(mid) * 2 + 0];
                const uint8_t gy = map.geo_grid[static_cast<size_t>(mid) * 2 + 1];
                const uint8_t v  = is_x ? gx : gy;
                const uint8_t nib = high ? static_cast<uint8_t>((v >> 4) & 0x0Fu) : static_cast<uint8_t>(v & 0x0Fu);
                const uint8_t bv = static_cast<uint8_t>((nib >> bit) & 1u);
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

void NvmMapWriter::write_all(const MapBins& map)
{
    if (map.N == 0) return;

    const uint32_t groups = (map.N + static_cast<uint32_t>(kLanesPerGroup) - 1u) / static_cast<uint32_t>(kLanesPerGroup);

    for (uint32_t g = 0; g < groups; ++g) {
        write_desc_group(map, g);
        write_geo_group(map, g);
    }
}

} // namespace msim