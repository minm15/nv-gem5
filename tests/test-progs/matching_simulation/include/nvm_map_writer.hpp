#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include "msim_bins.hpp"
#include "layout.hpp"

// Keep CimModule API unchanged (your cim_api.hpp/cpp).
#include "cim_api.hpp"

namespace msim {

class NvmMapWriter {
public:
    explicit NvmMapWriter(CimModule& cim, const Placement& place);

    void write_all(const MapBins& map);

private:
    CimModule& cim_;
    Placement place_;

    void write_desc_group(const MapBins& map, uint32_t group);
    void write_geo_group(const MapBins& map, uint32_t group);

    void map_desc_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array) const;
    void map_geo_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array, uint16_t& row_base) const;

    static inline uint8_t q4_bit(uint8_t v, int b) { return static_cast<uint8_t>((v >> b) & 1u); }
};

} // namespace msim