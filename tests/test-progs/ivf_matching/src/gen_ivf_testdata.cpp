#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

static void write_file_bytes(const fs::path& path, const void* data, size_t bytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open for write: " + path.string());
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("failed to write bytes: " + path.string());
}

template <class T>
static void write_file_vec(const fs::path& path, const std::vector<T>& values)
{
    if (values.empty()) {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("failed to create empty file: " + path.string());
        return;
    }

    write_file_bytes(path, values.data(), values.size() * sizeof(T));
}

static std::string json_escape(const std::string& value)
{
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"':  out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    return out.str();
}

static size_t bytes_of(const std::string& dtype, const std::vector<size_t>& shape)
{
    size_t numel = 1u;
    for (size_t dim : shape) numel *= dim;

    if (dtype == "uint8" || dtype == "int8") return numel;
    if (dtype == "uint16" || dtype == "int16") return numel * 2u;
    if (dtype == "uint32" || dtype == "int32" || dtype == "float32") return numel * 4u;
    if (dtype == "float64") return numel * 8u;
    throw std::runtime_error("unknown dtype: " + dtype);
}

static std::string shape_json(const std::vector<size_t>& shape)
{
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0u) out << ", ";
        out << shape[i];
    }
    out << "]";
    return out.str();
}

static void write_map_json(const fs::path& map_dir,
                           uint32_t N,
                           uint32_t K,
                           uint32_t D,
                           uint32_t postings_total,
                           uint8_t geo_max)
{
    struct Bin {
        std::string key;
        std::string file;
        std::string dtype;
        std::vector<size_t> shape;
    };

    const std::vector<Bin> bins = {
        {"map_desc_q4", "map_desc_q4.bin", "uint8", {N, 64}},
        {"map_geo_grid", "map_geo_grid.bin", "uint8", {N, 2}},
        {"ivf_centroids", "ivf_centroids.bin", "float32", {K, D}},
        {"ivf_postings_offsets", "ivf_postings_offsets.bin", "uint32", {K + 1u}},
        {"ivf_postings_map_ids", "ivf_postings_map_ids.bin", "int32", {postings_total}},
        {"ivf_val_scale", "ivf_val_scale.bin", "float32", {64}},
        {"ivf_idf_w", "ivf_idf_w.bin", "float32", {64}},
    };

    std::ostringstream out;
    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"endianness\": \"little\",\n";
    out << "  \"desc_bits\": 4,\n";
    out << "  \"geo_grid\": {\n";
    out << "    \"dtype\": \"uint8\",\n";
    out << "    \"max\": " << static_cast<int>(geo_max) << ",\n";
    out << "    \"qparams\": [0.0, 0.0, 1.0],\n";
    out << "    \"meaning\": \"deterministic verifier dataset\"\n";
    out << "  },\n";
    out << "  \"ivf\": {\n";
    out << "    \"nlist\": " << K << ",\n";
    out << "    \"space\": \"aug128_l2\",\n";
    out << "    \"extra\": {\n";
    out << "      \"val_scale\": {\"ref\": \"ivf_val_scale\"},\n";
    out << "      \"idf_w\": {\"ref\": \"ivf_idf_w\"},\n";
    out << "      \"alpha_value\": 1.0,\n";
    out << "      \"beta_presence\": 5.0,\n";
    out << "      \"use_idf\": 1\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"bins\": {\n";
    for (size_t i = 0; i < bins.size(); ++i) {
        const auto& bin = bins[i];
        out << "    \"" << bin.key << "\": {\n";
        out << "      \"file\": \"" << json_escape(bin.file) << "\",\n";
        out << "      \"dtype\": \"" << bin.dtype << "\",\n";
        out << "      \"shape\": " << shape_json(bin.shape) << ",\n";
        out << "      \"c_order\": true,\n";
        out << "      \"bytes\": " << bytes_of(bin.dtype, bin.shape) << "\n";
        out << "    }" << (i + 1u < bins.size() ? "," : "") << "\n";
    }
    out << "  },\n";
    out << "  \"desc_quant\": {\n";
    out << "    \"method\": \"handcrafted_verifier\",\n";
    out << "    \"thresholds\": [],\n";
    out << "    \"note\": \"Each query case has a known expected winner\"\n";
    out << "  }\n";
    out << "}\n";

    const std::string text = out.str();
    write_file_bytes(map_dir / "map.json", text.data(), text.size());
}

static void write_query_json(const fs::path& query_dir,
                             size_t steps,
                             size_t rows,
                             uint8_t geo_max)
{
    struct Bin {
        std::string key;
        std::string file;
        std::string dtype;
        std::vector<size_t> shape;
    };

    const std::vector<Bin> bins = {
        {"query_desc_q4", "query_desc_q4.bin", "uint8", {rows, 64}},
        {"query_step_pose_grid", "query_step_pose_grid.bin", "uint8", {steps, 2}},
        {"query_step_offsets", "query_step_offsets.bin", "uint32", {steps + 1u}},
        {"query_step_counts", "query_step_counts.bin", "uint16", {steps}},
        {"query_step_relodo_i", "query_step_relodo_i.bin", "int32", {steps}},
        {"query_step_t_now", "query_step_t_now.bin", "float64", {steps}},
    };

    std::ostringstream out;
    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"endianness\": \"little\",\n";
    out << "  \"session\": \"ivf_verify\",\n";
    out << "  \"desc_bits\": 4,\n";
    out << "  \"geo_grid\": {\n";
    out << "    \"dtype\": \"uint8\",\n";
    out << "    \"max\": " << static_cast<int>(geo_max) << ",\n";
    out << "    \"qparams\": [0.0, 0.0, 1.0],\n";
    out << "    \"meaning\": \"deterministic verifier dataset\"\n";
    out << "  },\n";
    out << "  \"ivf\": {\n";
    out << "    \"centroids_ref\": \"use map.json/ivf_centroids.bin\",\n";
    out << "    \"postings_ref\": \"use map.json/ivf_postings_*.bin\",\n";
    out << "    \"map_json_ref\": \"../map/map.json\"\n";
    out << "  },\n";
    out << "  \"bins\": {\n";
    for (size_t i = 0; i < bins.size(); ++i) {
        const auto& bin = bins[i];
        out << "    \"" << bin.key << "\": {\n";
        out << "      \"file\": \"" << json_escape(bin.file) << "\",\n";
        out << "      \"dtype\": \"" << bin.dtype << "\",\n";
        out << "      \"shape\": " << shape_json(bin.shape) << ",\n";
        out << "      \"c_order\": true,\n";
        out << "      \"bytes\": " << bytes_of(bin.dtype, bin.shape) << "\n";
        out << "    }" << (i + 1u < bins.size() ? "," : "") << "\n";
    }
    out << "  }\n";
    out << "}\n";

    const std::string text = out.str();
    write_file_bytes(query_dir / "query.json", text.data(), text.size());
}

static void fill_desc_all(std::vector<uint8_t>& desc, uint32_t map_id, uint8_t value)
{
    const size_t base = static_cast<size_t>(map_id) * 64u;
    for (size_t d = 0; d < 64u; ++d) desc[base + d] = static_cast<uint8_t>(value & 0x0Fu);
}

static void fill_desc_halves(std::vector<uint8_t>& desc,
                             uint32_t map_id,
                             uint8_t first_half,
                             uint8_t second_half)
{
    const size_t base = static_cast<size_t>(map_id) * 64u;
    for (size_t d = 0; d < 32u; ++d) desc[base + d] = static_cast<uint8_t>(first_half & 0x0Fu);
    for (size_t d = 32u; d < 64u; ++d) desc[base + d] = static_cast<uint8_t>(second_half & 0x0Fu);
}

static void set_geo(std::vector<uint8_t>& geo, uint32_t map_id, uint8_t gx, uint8_t gy)
{
    const size_t base = static_cast<size_t>(map_id) * 2u;
    geo[base + 0u] = gx;
    geo[base + 1u] = gy;
}

static void fill_query_all(std::vector<uint8_t>& query_desc, size_t query_id, uint8_t value)
{
    const size_t base = query_id * 64u;
    for (size_t d = 0; d < 64u; ++d) query_desc[base + d] = static_cast<uint8_t>(value & 0x0Fu);
}

static void fill_query_halves(std::vector<uint8_t>& query_desc,
                              size_t query_id,
                              uint8_t first_half,
                              uint8_t second_half)
{
    const size_t base = query_id * 64u;
    for (size_t d = 0; d < 32u; ++d) query_desc[base + d] = static_cast<uint8_t>(first_half & 0x0Fu);
    for (size_t d = 32u; d < 64u; ++d) query_desc[base + d] = static_cast<uint8_t>(second_half & 0x0Fu);
}

} // namespace

int main()
{
    try {
        const fs::path root_dir = fs::path("test");
        const fs::path map_dir = root_dir / "map";
        const fs::path query_dir = root_dir / "query";

        fs::create_directories(map_dir);
        fs::create_directories(query_dir);

        const uint32_t K = 8u;
        const uint32_t per_bucket = 4u;
        const uint32_t N = K * per_bucket;
        const uint32_t D = 128u;
        const uint8_t geo_max = 15u;

        std::vector<uint8_t> map_desc(static_cast<size_t>(N) * 64u, 15u);
        std::vector<uint8_t> map_geo(static_cast<size_t>(N) * 2u, 15u);
        std::vector<float> centroids(static_cast<size_t>(K) * D, 0.0f);
        std::vector<uint32_t> offsets(static_cast<size_t>(K) + 1u, 0u);
        std::vector<int32_t> map_ids(static_cast<size_t>(N), 0);
        std::vector<float> val_scale(64u, 1.0f);
        std::vector<float> idf_w(64u, 1.0f);

        for (uint32_t lid = 0; lid < K; ++lid) {
            offsets[static_cast<size_t>(lid)] = lid * per_bucket;
            for (uint32_t j = 0; j < per_bucket; ++j) {
                const uint32_t map_id = lid * per_bucket + j;
                map_ids[static_cast<size_t>(lid * per_bucket + j)] = static_cast<int32_t>(map_id);
            }
            centroids[static_cast<size_t>(lid) * D] = static_cast<float>(lid);
        }
        offsets.back() = N;

        fill_desc_all(map_desc, 0u, 1u);  set_geo(map_geo, 0u, 5u, 5u);
        fill_desc_all(map_desc, 1u, 2u);  set_geo(map_geo, 1u, 4u, 5u);
        fill_desc_all(map_desc, 2u, 3u);  set_geo(map_geo, 2u, 5u, 7u);
        fill_desc_all(map_desc, 3u, 4u);  set_geo(map_geo, 3u, 9u, 9u);

        fill_desc_all(map_desc, 4u, 6u);  set_geo(map_geo, 4u, 2u, 2u);
        fill_desc_all(map_desc, 5u, 6u);  set_geo(map_geo, 5u, 2u, 2u);
        fill_desc_all(map_desc, 6u, 7u);  set_geo(map_geo, 6u, 2u, 2u);
        fill_desc_all(map_desc, 7u, 8u);  set_geo(map_geo, 7u, 0u, 0u);

        fill_desc_halves(map_desc, 8u, 11u, 6u);  set_geo(map_geo, 8u, 8u, 1u);
        fill_desc_halves(map_desc, 9u, 1u, 7u);   set_geo(map_geo, 9u, 8u, 1u);
        fill_desc_all(map_desc, 10u, 12u);        set_geo(map_geo, 10u, 10u, 10u);
        fill_desc_all(map_desc, 11u, 13u);        set_geo(map_geo, 11u, 11u, 11u);

        for (uint32_t map_id = 12u; map_id < N; ++map_id) {
            fill_desc_all(map_desc, map_id, static_cast<uint8_t>((map_id % 14u) + 1u));
            set_geo(map_geo, map_id, 15u, 15u);
        }

        write_file_vec(map_dir / "map_desc_q4.bin", map_desc);
        write_file_vec(map_dir / "map_geo_grid.bin", map_geo);
        write_file_vec(map_dir / "ivf_centroids.bin", centroids);
        write_file_vec(map_dir / "ivf_postings_offsets.bin", offsets);
        write_file_vec(map_dir / "ivf_postings_map_ids.bin", map_ids);
        write_file_vec(map_dir / "ivf_val_scale.bin", val_scale);
        write_file_vec(map_dir / "ivf_idf_w.bin", idf_w);
        write_map_json(map_dir, N, K, D, N, geo_max);

        const size_t steps = 5u;
        const size_t query_rows = 7u;

        std::vector<uint8_t> query_desc(query_rows * 64u, 0u);
        std::vector<uint8_t> step_pose_grid = {
            5u, 5u,
            2u, 2u,
            8u, 1u,
            14u, 14u,
            0u, 0u,
        };
        std::vector<uint32_t> step_offsets = {0u, 3u, 4u, 5u, 6u, 7u};
        std::vector<uint16_t> step_counts = {3u, 1u, 1u, 1u, 1u};
        std::vector<int32_t> relodo_i = {0, 1, 2, 3, 4};
        std::vector<double> t_now = {0.0, 1.0, 2.0, 3.0, 4.0};

        fill_query_all(query_desc, 0u, 1u);
        fill_query_all(query_desc, 1u, 2u);
        fill_query_all(query_desc, 2u, 3u);
        fill_query_all(query_desc, 3u, 6u);
        fill_query_halves(query_desc, 4u, 0u, 7u);
        fill_query_all(query_desc, 5u, 4u);
        fill_query_all(query_desc, 6u, 8u);

        write_file_vec(query_dir / "query_desc_q4.bin", query_desc);
        write_file_vec(query_dir / "query_step_pose_grid.bin", step_pose_grid);
        write_file_vec(query_dir / "query_step_offsets.bin", step_offsets);
        write_file_vec(query_dir / "query_step_counts.bin", step_counts);
        write_file_vec(query_dir / "query_step_relodo_i.bin", relodo_i);
        write_file_vec(query_dir / "query_step_t_now.bin", t_now);
        write_query_json(query_dir, steps, query_rows, geo_max);

        std::cout << "[ok] wrote deterministic verifier dataset\n";
        std::cout << "  map:   " << map_dir << "\n";
        std::cout << "  query: " << query_dir << "\n";
        std::cout << "[cases] exact=mid0 x_sweep=mid1 y_sweep=mid2 tie=mid4 zero_dims=mid9 no_match edge=mid7\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}
