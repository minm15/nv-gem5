#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "msim_json.hpp"
#include "bin_reader.hpp"

namespace msim {

struct GeoGridParams {
    float min_x = 0.0f;
    float min_y = 0.0f;
    float R     = 1.0f;
    uint8_t max = 0;
};

class MapLoader {
public:
    struct IvfData {
        bool enabled = false;

        uint32_t nlist = 0;
        uint32_t dim   = 0;

        std::vector<float>    centroids_f32;        // (K*D)
        std::vector<uint32_t> postings_offsets_u32; // (K+1)
        std::vector<int32_t>  postings_map_ids_i32; // (total)

        // optional extras (ZeroAwareIVF)
        std::vector<float> val_scale_f32; // (64)
        std::vector<float> idf_w_f32;     // (64)
        float alpha_value   = 1.0f;
        float beta_presence = 5.0f;
        bool  use_idf       = true;

        const float* centroid_ptr(uint32_t lid) const;
        uint32_t postings_begin(uint32_t lid) const;
        uint32_t postings_end(uint32_t lid) const;
        uint32_t postings_size(uint32_t lid) const;
    };

    bool load_from_folder(const std::string& map_dir);

    size_t n() const { return n_; }
    uint8_t desc_bits() const { return desc_bits_; }
    const GeoGridParams& geo() const { return geo_; }

    const std::vector<uint8_t>& desc_q4_u8() const { return desc_q4_u8_; }   // (N*64)
    const std::vector<uint8_t>& geo_grid_u8() const { return geo_grid_u8_; } // (N*2)

    const IvfData& ivf() const { return ivf_; }

private:
    size_t n_ = 0;
    uint8_t desc_bits_ = 0;
    GeoGridParams geo_{};

    std::vector<uint8_t> desc_q4_u8_;
    std::vector<uint8_t> geo_grid_u8_;

    IvfData ivf_{};

    static void validate_or_throw(bool ok, const std::string& msg);
    static uint64_t get_u64(const std::string& ctx, const JsonValue& j);
    static double   get_f64(const std::string& ctx, const JsonValue& j);
    static std::string get_str(const std::string& ctx, const JsonValue& j);

    static std::string join_path(const std::string& a, const std::string& b);

    bool load_dense_map_bins_(const std::string& dir);
    bool load_ivf_bins_if_present_(const std::string& dir);
};

} // namespace msim