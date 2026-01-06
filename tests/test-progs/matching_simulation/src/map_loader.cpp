#include "map_loader.hpp"

#include <stdexcept>

namespace msim {

std::string MapLoader::join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

bool MapLoader::load_from_folder(const std::string& map_dir) {
    map_dir_ = map_dir;
    map_json_path_ = join_path(map_dir_, "map.json");

    root_ = load_json_file(map_json_path_);

    // Validate minimal schema
    if (!root_.has("bins")) throw std::runtime_error("map.json missing 'bins'");
    if (!root_.has("geo_grid")) throw std::runtime_error("map.json missing 'geo_grid'");

    // Read geo params (optional for this stage, but useful for debug and checks)
    const JsonValue& gg = root_.at("geo_grid");
    if (const JsonValue* mx = gg.try_get("max")) {
        if (!mx->is_number()) throw std::runtime_error("geo_grid.max must be a number");
        geo_.max = static_cast<uint32_t>(mx->get_number());
    }
    if (const JsonValue* qp = gg.try_get("qparams")) {
        if (qp->is_array()) {
            const auto& a = qp->as_array();
            if (a.size() == 3) {
                geo_.min_x = a[0].get_number();
                geo_.min_y = a[1].get_number();
                geo_.R     = a[2].get_number();
            }
        }
    }

    // Read desc_q4
    const BinMeta descm = read_bin_meta(root_, "map_desc_q4");
    if (descm.dtype != "uint8") throw std::runtime_error("map_desc_q4 dtype must be uint8");
    if (descm.shape.size() != 2 || descm.shape[1] != 64) throw std::runtime_error("map_desc_q4 shape must be (N,64)");

    n_ = descm.shape[0];
    const size_t desc_count = numel_from_shape(descm.shape);

    const std::string desc_path = join_path(map_dir_, descm.file);
    const size_t desc_fs = file_size_bytes(desc_path);
    if (desc_fs != descm.bytes) {
        throw std::runtime_error("map_desc_q4 file size mismatch: expected " + std::to_string(descm.bytes) +
                                 ", got " + std::to_string(desc_fs));
    }
    desc_q4_ = read_bin_as<uint8_t>(desc_path, desc_count);

    // Read geo_grid
    const BinMeta geom = read_bin_meta(root_, "map_geo_grid");
    if (geom.dtype != "uint8") throw std::runtime_error("map_geo_grid dtype must be uint8");
    if (geom.shape.size() != 2 || geom.shape[0] != n_ || geom.shape[1] != 2) {
        throw std::runtime_error("map_geo_grid shape must be (N,2) and N must match map_desc_q4");
    }
    const size_t geo_count = numel_from_shape(geom.shape);

    const std::string geo_path = join_path(map_dir_, geom.file);
    const size_t geo_fs = file_size_bytes(geo_path);
    if (geo_fs != geom.bytes) {
        throw std::runtime_error("map_geo_grid file size mismatch: expected " + std::to_string(geom.bytes) +
                                 ", got " + std::to_string(geo_fs));
    }
    geo_grid_ = read_bin_as<uint8_t>(geo_path, geo_count);

    return true;
}

} // namespace msim