#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "msim_json.hpp"
#include "bin_reader.hpp"

namespace msim {

struct GeoQParams {
    double min_x = 0.0;
    double min_y = 0.0;
    double R = 0.0;
    uint32_t max = 0;
};

class MapLoader {
public:
    bool load_from_folder(const std::string& map_dir);

    size_t map_size() const { return n_; }

    const std::vector<uint8_t>& desc_q4() const { return desc_q4_; }
    const std::vector<uint8_t>& geo_grid() const { return geo_grid_; }

    const GeoQParams& geo_params() const { return geo_; }

    const std::string& map_dir() const { return map_dir_; }
    const std::string& map_json_path() const { return map_json_path_; }

    const JsonValue& json_root() const { return root_; }

private:
    std::string map_dir_;
    std::string map_json_path_;
    JsonValue root_;

    size_t n_ = 0;
    GeoQParams geo_;

    std::vector<uint8_t> desc_q4_;   // (N,64)
    std::vector<uint8_t> geo_grid_;  // (N,2)

    static std::string join_path(const std::string& a, const std::string& b);
};

} // namespace msim