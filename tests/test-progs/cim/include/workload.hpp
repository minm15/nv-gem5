#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lab {

// Descriptor: 64 dims, each dim is a 4-bit value stored in uint8_t (low nibble).
static constexpr int kDims = 64;
static constexpr int kValueBits = 4;

struct Workload {
    // Map (database)
    size_t M = 0;
    std::vector<std::array<uint8_t, kDims>> map_desc;  // [M][64], each in [0,15]
    std::vector<uint8_t> map_x;                        // [M]
    std::vector<uint8_t> map_y;                        // [M]

    // Query
    size_t Q = 0;
    std::vector<std::array<uint8_t, kDims>> query_desc; // [Q][64], each in [0,15]
    std::vector<uint8_t> query_rx;                      // [Q]
    std::vector<uint8_t> query_ry;                      // [Q]
    std::vector<uint8_t> query_range;                   // [Q] (radius in same 8-bit coordinate space)
};

} // namespace lab
