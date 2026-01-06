#include "layout.hpp"
#include <cmath>
#include <stdexcept>

namespace msim {

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

Placement compute_placement(uint32_t map_N)
{
    Placement p;
    p.total_banks = 16;

    const uint32_t groups = ceil_div_u32(map_N, static_cast<uint32_t>(kLanesPerGroup));
    const uint32_t arrays_per_bank = (1u << kMatBits) * (1u << kArrayBits); // 256

    const uint32_t desc_arrays = groups; // 1 group per desc array
    const uint32_t geo_arrays  = ceil_div_u32(groups, static_cast<uint32_t>(kGeoGroupsPerArray)); // 16 groups per geo array

    const uint32_t desc_banks = ceil_div_u32(desc_arrays, arrays_per_bank);
    const uint32_t geo_banks  = ceil_div_u32(geo_arrays,  arrays_per_bank);

    if (desc_banks + geo_banks > p.total_banks) {
        throw std::runtime_error("Not enough banks for desc+geo with current geometry");
    }

    p.desc_bank0 = 0;
    p.desc_bank_count = static_cast<uint16_t>(desc_banks);
    p.geo_bank0 = static_cast<uint16_t>(p.desc_bank0 + p.desc_bank_count);
    p.geo_bank_count = static_cast<uint16_t>(geo_banks);

    return p;
}

} // namespace msim