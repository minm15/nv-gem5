#include "msim_bins.hpp"
#include "file_utils.hpp"
#include "msim_config.hpp"
#include <stdexcept>

namespace msim {

static std::string join_path(const std::string& a, const std::string& b)
{
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

MapBins load_map_bins(const std::string& map_dir)
{
    MapBins m;
    m.geo_max = msim::kGeoMaxDefault;

    const std::string f_desc = join_path(map_dir, "map_desc_q4.bin");
    const std::string f_geo  = join_path(map_dir, "map_geo_grid.bin");

    const size_t geo_bytes = file_size_bytes(f_geo);
    if (geo_bytes % 2 != 0) throw std::runtime_error("map_geo_grid.bin size not multiple of 2");
    m.N = static_cast<uint32_t>(geo_bytes / 2);

    const size_t desc_bytes = file_size_bytes(f_desc);
    const size_t expect_desc = static_cast<size_t>(m.N) * 64;
    if (desc_bytes != expect_desc) {
        throw std::runtime_error("map_desc_q4.bin size mismatch: got=" + std::to_string(desc_bytes) +
                                 " expect=" + std::to_string(expect_desc));
    }

    m.desc_q4 = read_file_u8(f_desc);
    m.geo_grid = read_file_u8(f_geo);

    return m;
}

QueryBins load_query_bins(const std::string& query_dir)
{
    QueryBins q;

    const std::string f_desc   = join_path(query_dir, "query_desc_q4.bin");
    const std::string f_pose   = join_path(query_dir, "query_step_pose_grid.bin");
    const std::string f_offset = join_path(query_dir, "query_step_offsets.bin");

    const size_t desc_bytes = file_size_bytes(f_desc);
    if (desc_bytes % 64 != 0) throw std::runtime_error("query_desc_q4.bin size not multiple of 64");
    q.total_rows = static_cast<uint32_t>(desc_bytes / 64);

    const size_t pose_bytes = file_size_bytes(f_pose);
    if (pose_bytes % 2 != 0) throw std::runtime_error("query_step_pose_grid.bin size not multiple of 2");
    q.steps = static_cast<uint32_t>(pose_bytes / 2);

    q.offsets = read_file_u32(f_offset);
    if (q.offsets.size() != static_cast<size_t>(q.steps) + 1) {
        throw std::runtime_error("query_step_offsets length mismatch");
    }
    if (q.offsets.front() != 0u) {
        throw std::runtime_error("query_step_offsets[0] must be 0");
    }
    if (q.offsets.back() != q.total_rows) {
        throw std::runtime_error("query_step_offsets[last] must equal total_rows");
    }

    q.desc_q4 = read_file_u8(f_desc);
    q.step_pose = read_file_u8(f_pose);

    return q;
}

} // namespace msim