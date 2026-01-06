#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace msim {

struct MapBins {
    uint32_t N = 0;                   // number of map descriptors
    uint8_t geo_max = 0;
    std::vector<uint8_t> desc_q4;     // (N,64) uint8
    std::vector<uint8_t> geo_grid;    // (N,2)  uint8
};

struct QueryBins {
    uint32_t total_rows = 0;          // Q rows (descriptors)
    uint32_t steps = 0;               // S steps
    std::vector<uint8_t> desc_q4;     // (total_rows,64)
    std::vector<uint8_t> step_pose;   // (steps,2) [qgx,qgy]
    std::vector<uint32_t> offsets;    // (steps+1)
};

MapBins load_map_bins(const std::string& map_dir);
QueryBins load_query_bins(const std::string& query_dir);

} // namespace msim