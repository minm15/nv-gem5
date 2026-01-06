#pragma once
#include <cstdint>
#include <cstddef>
#include "msim_config.hpp"

namespace msim {

struct Placement {
    uint16_t total_banks = 16;
    uint16_t desc_bank0 = 0;
    uint16_t desc_bank_count = 0;
    uint16_t geo_bank0 = 0;
    uint16_t geo_bank_count = 0;
};

Placement compute_placement(uint32_t map_N);

static inline void lane_set_bit(uint8_t* row, int lane, uint8_t v)
{
    const int byte_i = lane >> 3;
    const int bit_i  = lane & 7;
    const uint8_t mask = static_cast<uint8_t>(1u << bit_i);
    if (v) row[byte_i] = static_cast<uint8_t>(row[byte_i] | mask);
    else   row[byte_i] = static_cast<uint8_t>(row[byte_i] & static_cast<uint8_t>(~mask));
}

static inline uint8_t lane_get_bit(const uint8_t* row, int lane)
{
    const int byte_i = lane >> 3;
    const int bit_i  = lane & 7;
    return static_cast<uint8_t>((row[byte_i] >> bit_i) & 1u);
}

// Descriptor row mapping (q4): for each dim d,
// rows: [d*8 + 0..3] -> true bit0..3, [d*8 + 4..7] -> inv bit0..3.
static inline uint16_t desc_true_row(int d, int b) { return static_cast<uint16_t>(d * (2 * kDescBits) + b); }
static inline uint16_t desc_inv_row (int d, int b) { return static_cast<uint16_t>(d * (2 * kDescBits) + kDescBits + b); }

// Geometry group rows (32):
// x low nibble: 0..7  (bit0 true, bit0 inv, bit1 true, bit1 inv, ...)
// x high nibble: 8..15
// y low nibble: 16..23
// y high nibble: 24..31
static inline uint16_t geo_row_base_for_group_in_array(uint32_t group_in_array)
{
    return static_cast<uint16_t>(group_in_array * kGeoRowsPerGroup);
}

static inline uint16_t geo_x_row_low (int b, bool inv) { return static_cast<uint16_t>(0  + 2 * b + (inv ? 1 : 0)); }
static inline uint16_t geo_x_row_high(int b, bool inv) { return static_cast<uint16_t>(8  + 2 * b + (inv ? 1 : 0)); }
static inline uint16_t geo_y_row_low (int b, bool inv) { return static_cast<uint16_t>(16 + 2 * b + (inv ? 1 : 0)); }
static inline uint16_t geo_y_row_high(int b, bool inv) { return static_cast<uint16_t>(24 + 2 * b + (inv ? 1 : 0)); }

} // namespace msim