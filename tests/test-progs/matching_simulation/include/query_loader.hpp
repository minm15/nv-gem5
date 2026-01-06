#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "msim_json.hpp"
#include "bin_reader.hpp"

namespace msim {

struct QueryStepView {
    uint8_t gx = 0;
    uint8_t gy = 0;

    const uint8_t* desc_ptr = nullptr; // Points into contiguous query_desc_q4 buffer
    size_t m = 0;                      // Number of descriptors for this step

    int32_t relodo_i = -1;
    double t_now = 0.0;

    bool has_optional_meta = false;
};

class QueryLoader {
public:
    bool load_from_folder(const std::string& query_dir);

    size_t steps() const { return steps_; }
    size_t total_query_desc_rows() const { return total_rows_; }

    QueryStepView step(size_t s) const;

    const std::string& query_dir() const { return query_dir_; }
    const std::string& query_json_path() const { return query_json_path_; }

    const JsonValue& json_root() const { return root_; }

private:
    std::string query_dir_;
    std::string query_json_path_;
    JsonValue root_;

    size_t steps_ = 0;
    size_t total_rows_ = 0;

    // Bins
    std::vector<uint8_t>  query_desc_q4_;        // (Q,64) uint8
    std::vector<uint8_t>  step_pose_grid_;       // (S,2)  uint8
    std::vector<uint32_t> step_offsets_;         // (S+1)  uint32
    std::vector<uint16_t> step_counts_;          // (S)    uint16
    std::vector<int32_t>  step_relodo_i_;        // (S)    int32 (optional)
    std::vector<double>   step_t_now_;           // (S)    float64 (optional)

    bool has_relodo_i_ = false;
    bool has_t_now_ = false;

    static std::string join_path(const std::string& a, const std::string& b);

    template <typename T>
    static void validate_file_size_or_throw(const std::string& path, size_t expect_bytes);
};

} // namespace msim