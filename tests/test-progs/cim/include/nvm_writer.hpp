#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "workload.hpp"

class CimModule; // from cim_api.hpp

namespace lab {

// Memory placement / layout policy.
struct Placement {
    // Descriptor banks (4 banks)
    uint16_t bank_desc0 = 0;
    uint16_t bank_desc1 = 1;
    uint16_t bank_desc2 = 2;
    uint16_t bank_desc3 = 3;

    // Position bank (x/y bitplanes, packed by groups within rows)
    uint16_t bank_pos = 4;

    uint16_t mat = 0;
};

// Writer that encodes workload into the CIM memory layout.
class NvmWriter {
public:
    // Geometry (match your CimHandler bit slicing)
    static constexpr int kBankBits  = 3;  // 8 banks (we use 0..4)
    static constexpr int kMatBits   = 4;  // mat fixed to 0 in this benchmark
    static constexpr int kArrayBits = 4;  // 256 arrays per bank
    static constexpr int kRowBits   = 9;  // 512 rows per array
    static constexpr int kColBits   = 6;  // 512 columns (1-bit lanes) => 64 bytes per row

    static constexpr int kLanesPerGroup = 512;
    static constexpr int kRowBytes = kLanesPerGroup / 8;

    // Position packing: 16 rows per group (x:8, y:8), 32 groups per array
    static constexpr int kPosRowsPerGroup = 16;
    static constexpr int kGroupsPerPosArray = 32;

    explicit NvmWriter(Placement placement);

    const Placement& placement() const { return placement_; }

    // Encode and write all map descriptors + positions.
    void write_all(CimModule& cim, const Workload& wl) const;

private:
    Placement placement_;

    static inline void set_lane_bit(std::array<uint8_t, kRowBytes>& row,
                                    int lane, bool one)
    {
        const int byte_i = lane >> 3;
        const int bit_i = lane & 7;
        if (one) row[static_cast<size_t>(byte_i)] |= static_cast<uint8_t>(1u << bit_i);
    }

    static void group_to_desc_bank_array(uint32_t group,
                                         const Placement& p,
                                         uint16_t& out_bank,
                                         uint16_t& out_array);

    static void group_to_pos_array_rowbase(uint32_t group,
                                           uint16_t& out_array,
                                           uint16_t& out_row_base);

    void write_desc_group(CimModule& cim,
                          const Workload& wl,
                          uint32_t group) const;

    void write_pos_group(CimModule& cim,
                         const Workload& wl,
                         uint32_t group) const;
};

} // namespace lab
