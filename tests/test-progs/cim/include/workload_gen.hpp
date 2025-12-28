#pragma once
#include <cstdint>
#include <string>
#include "workload.hpp"

namespace lab {

// Load (or generate if missing / incompatible) fixed-length binary datasets.
// - map_path: sequence of MapRecord { uint8 desc[64]; uint8 x; uint8 y; }
// - query_path: sequence of QueryRecord { uint8 desc[64]; uint8 rx; uint8 ry; uint8 range; }
Workload load_or_make_bin_workload(
    size_t M,
    size_t Q,
    uint32_t seed,
    uint8_t default_range,
    const std::string& map_path = "map.bin",
    const std::string& query_path = "query.bin");

} // namespace lab
