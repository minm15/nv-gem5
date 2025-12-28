#include "nvm_writer.hpp"
#include "cim_api.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lab {

NvmWriter::NvmWriter(Placement placement)
    : placement_(placement)
{}

void NvmWriter::group_to_desc_bank_array(uint32_t group,
                                        const Placement& p,
                                        uint16_t& out_bank,
                                        uint16_t& out_array)
{
    // 1024 groups total for 2^19 maps with 512 lanes/group.
    // Each desc bank holds 256 arrays => 256 groups/bank.
    const uint32_t bank_sel = group >> 8;    // /256
    out_array = static_cast<uint16_t>(group & 0xFFu);

    switch (bank_sel) {
        case 0: out_bank = p.bank_desc0; break;
        case 1: out_bank = p.bank_desc1; break;
        case 2: out_bank = p.bank_desc2; break;
        case 3: out_bank = p.bank_desc3; break;
        default: throw std::runtime_error("group_to_desc_bank_array: group out of range");
    }
}

void NvmWriter::group_to_pos_array_rowbase(uint32_t group,
                                          uint16_t& out_array,
                                          uint16_t& out_row_base)
{
    // 32 groups per pos array, stacked along rows in blocks of 16.
    out_array = static_cast<uint16_t>(group / kGroupsPerPosArray); // 0..31 for 1024 groups
    const uint16_t block = static_cast<uint16_t>(group % kGroupsPerPosArray);
    out_row_base = static_cast<uint16_t>(block * kPosRowsPerGroup);
}

void NvmWriter::write_all(CimModule& cim, const Workload& wl) const
{
    if (wl.M == 0) return;
    if ((wl.M % kLanesPerGroup) != 0) {
        throw std::runtime_error("wl.M must be multiple of 512 for this layout");
    }

    const uint32_t groups = static_cast<uint32_t>(wl.M / kLanesPerGroup);

    // Desc: groups across 4 banks (256 groups/bank). Pos: all groups in bank_pos (32 arrays).
    for (uint32_t g = 0; g < groups; ++g) {
        write_desc_group(cim, wl, g);
        write_pos_group(cim, wl, g);
    }
}

static inline uint16_t desc_orig_row(int d, int b)
{
    return static_cast<uint16_t>(d * (2 * kValueBits) + b);
}

static inline uint16_t desc_inv_row(int d, int b)
{
    return static_cast<uint16_t>(d * (2 * kValueBits) + kValueBits + b);
}

void NvmWriter::write_desc_group(CimModule& cim,
                                const Workload& wl,
                                uint32_t group) const
{
    uint16_t bank = 0, array = 0;
    group_to_desc_bank_array(group, placement_, bank, array);

    const uint32_t base = group * kLanesPerGroup;

    // Build and write 512 rows:
    // - rows 0..511 are used for 64 dims * (orig(4) + inv(4))
    for (int d = 0; d < kDims; ++d) {
        // printf("%d\n", d);
        for (int b = 0; b < kValueBits; ++b) {
            std::array<uint8_t, kRowBytes> row1{};
            row1.fill(0);

            for (int lane = 0; lane < kLanesPerGroup; ++lane) {
                const uint8_t v = wl.map_desc[static_cast<size_t>(base + lane)][static_cast<size_t>(d)] & 0x0Fu;
                const bool bit = ((v >> b) & 1u) != 0u;
                set_lane_bit(row1, lane, bit);
            }

            const uint16_t r1 = desc_orig_row(d, b);
            const uint16_t r0 = desc_inv_row(d, b);

            // orig: bit==1 mask
            cim.copy_to_cim(bank, placement_.mat, array, r1, row1.data(), kRowBytes);

            // inv: bit==0 mask = ~orig
            for (auto& x : row1) x = static_cast<uint8_t>(~x);
            cim.copy_to_cim(bank, placement_.mat, array, r0, row1.data(), kRowBytes);
        }
    }
}

void NvmWriter::write_pos_group(CimModule& cim,
                               const Workload& wl,
                               uint32_t group) const
{
    // Position bitplanes are stored in bank_pos. Within a pos-array, rows are partitioned:
    // for block in 0..31:
    //   rows block*16 + [0..7] => x bits [7..0] (MSB..LSB)
    //   rows block*16 + [8..15] => y bits [7..0] (MSB..LSB)
    uint16_t pos_array = 0, row_base = 0;
    group_to_pos_array_rowbase(group, pos_array, row_base);

    const uint32_t base = group * kLanesPerGroup;

    for (int r = 0; r < 8; ++r) {
        const int bit = 7 - r; // row0 is MSB
        std::array<uint8_t, kRowBytes> row{};
        row.fill(0);
        for (int lane = 0; lane < kLanesPerGroup; ++lane) {
            const uint8_t x = wl.map_x[static_cast<size_t>(base + lane)];
            set_lane_bit(row, lane, ((x >> bit) & 1u) != 0u);
        }
        cim.copy_to_cim(placement_.bank_pos, placement_.mat, pos_array,
                        static_cast<uint16_t>(row_base + r), row.data(), kRowBytes);
    }

    for (int r = 0; r < 8; ++r) {
        const int bit = 7 - r;
        std::array<uint8_t, kRowBytes> row{};
        row.fill(0);
        for (int lane = 0; lane < kLanesPerGroup; ++lane) {
            const uint8_t y = wl.map_y[static_cast<size_t>(base + lane)];
            set_lane_bit(row, lane, ((y >> bit) & 1u) != 0u);
        }
        cim.copy_to_cim(placement_.bank_pos, placement_.mat, pos_array,
                        static_cast<uint16_t>(row_base + 8 + r), row.data(), kRowBytes);
    }
}

} // namespace lab
