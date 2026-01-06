#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void write_file_bytes(const fs::path& p, const void* data, size_t bytes)
{
    std::ofstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open for write: " + p.string());
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("failed to write bytes: " + p.string());
}

template <class T>
static void write_file_vec(const fs::path& p, const std::vector<T>& v)
{
    if (v.empty()) {
        std::ofstream f(p, std::ios::binary);
        if (!f) throw std::runtime_error("failed to create empty file: " + p.string());
        return;
    }
    write_file_bytes(p, v.data(), v.size() * sizeof(T));
}

static std::string json_escape(const std::string& s)
{
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
        case '\\': o << "\\\\"; break;
        case '"':  o << "\\\""; break;
        case '\n': o << "\\n";  break;
        case '\r': o << "\\r";  break;
        case '\t': o << "\\t";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                o << "\\u"
                  << std::hex << std::uppercase
                  << (int)((unsigned char)c / 16)
                  << (int)((unsigned char)c % 16)
                  << std::nouppercase << std::dec;
            } else {
                o << c;
            }
        }
    }
    return o.str();
}

static size_t bytes_of(const std::string& dtype, const std::vector<size_t>& shape)
{
    size_t elem = 1;
    for (size_t d : shape) elem *= d;

    size_t s = 0;
    if      (dtype == "uint8")   s = 1;
    else if (dtype == "int8")    s = 1;
    else if (dtype == "uint16")  s = 2;
    else if (dtype == "int16")   s = 2;
    else if (dtype == "uint32")  s = 4;
    else if (dtype == "int32")   s = 4;
    else if (dtype == "float32") s = 4;
    else if (dtype == "float64") s = 8;
    else throw std::runtime_error("unknown dtype: " + dtype);

    return elem * s;
}

static std::string shape_json(const std::vector<size_t>& shape)
{
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) o << ", ";
        o << shape[i];
    }
    o << "]";
    return o.str();
}

static void write_map_json(
    const fs::path& map_dir,
    uint32_t N,
    uint32_t K,
    uint32_t D,
    uint8_t geo_max)
{
    // bins metadata must match bin_reader.cpp expectations: bins.<key> = {file,dtype,shape,c_order,bytes}
    struct Bin { std::string key, file, dtype; std::vector<size_t> shape; };

    const std::vector<Bin> bins = {
        {"map_desc_q4",            "map_desc_q4.bin",            "uint8",   {N, 64}},
        {"map_geo_grid",           "map_geo_grid.bin",           "uint8",   {N, 2}},
        {"ivf_centroids",          "ivf_centroids.bin",          "float32", {K, D}},
        {"ivf_postings_offsets",   "ivf_postings_offsets.bin",   "uint32",  {K + 1u}},
        {"ivf_postings_map_ids",   "ivf_postings_map_ids.bin",   "int32",   {static_cast<size_t>(K) * 16u}},
        {"ivf_val_scale",          "ivf_val_scale.bin",          "float32", {64}},
        {"ivf_idf_w",              "ivf_idf_w.bin",              "float32", {64}},
    };

    std::ostringstream o;
    o << "{\n";
    o << "  \"version\": 1,\n";
    o << "  \"endianness\": \"little\",\n";
    o << "  \"desc_bits\": 4,\n";
    o << "  \"geo_grid\": {\n";
    o << "    \"dtype\": \"uint8\",\n";
    o << "    \"max\": " << (int)geo_max << ",\n";
    o << "    \"qparams\": [0.0, 0.0, 100.0],\n";
    o << "    \"meaning\": \"grid_x, grid_y in [0..max]; computed by shared span R\"\n";
    o << "  },\n";

    o << "  \"ivf\": {\n";
    o << "    \"nlist\": " << K << ",\n";
    o << "    \"space\": \"aug128_l2\",\n";
    o << "    \"extra\": {\n";
    o << "      \"val_scale\": {\"ref\": \"ivf_val_scale\"},\n";
    o << "      \"idf_w\": {\"ref\": \"ivf_idf_w\"},\n";
    o << "      \"alpha_value\": 1.0,\n";
    o << "      \"beta_presence\": 5.0,\n";
    o << "      \"use_idf\": 1\n";
    o << "    }\n";
    o << "  },\n";

    o << "  \"bins\": {\n";
    for (size_t i = 0; i < bins.size(); ++i) {
        const auto& b = bins[i];
        const size_t bytes = bytes_of(b.dtype, b.shape);
        o << "    \"" << b.key << "\": {\n";
        o << "      \"file\": \"" << json_escape(b.file) << "\",\n";
        o << "      \"dtype\": \"" << b.dtype << "\",\n";
        o << "      \"shape\": " << shape_json(b.shape) << ",\n";
        o << "      \"c_order\": true,\n";
        o << "      \"bytes\": " << bytes << "\n";
        o << "    }" << (i + 1 < bins.size() ? "," : "") << "\n";
    }
    o << "  },\n";

    // optional but harmless
    o << "  \"desc_quant\": {\n";
    o << "    \"method\": \"synthetic_random\",\n";
    o << "    \"thresholds\": [],\n";
    o << "    \"note\": \"testdata generator\"\n";
    o << "  }\n";

    o << "}\n";

    write_file_bytes(map_dir / "map.json", o.str().data(), o.str().size());
}

static void write_query_json(
    const fs::path& query_dir,
    size_t S,
    size_t Q,
    uint8_t geo_max)
{
    struct Bin { std::string key, file, dtype; std::vector<size_t> shape; };

    const std::vector<Bin> bins = {
        {"query_desc_q4",        "query_desc_q4.bin",        "uint8",   {Q, 64}},
        {"query_step_pose_grid", "query_step_pose_grid.bin", "uint8",   {S, 2}},
        {"query_step_offsets",   "query_step_offsets.bin",   "uint32",  {S + 1}},
        {"query_step_counts",    "query_step_counts.bin",    "uint16",  {S}},
        {"query_step_relodo_i",  "query_step_relodo_i.bin",  "int32",   {S}},
        {"query_step_t_now",     "query_step_t_now.bin",     "float64", {S}},
    };

    std::ostringstream o;
    o << "{\n";
    o << "  \"version\": 1,\n";
    o << "  \"endianness\": \"little\",\n";
    o << "  \"session\": \"synthetic_test\",\n";
    o << "  \"desc_bits\": 4,\n";
    o << "  \"geo_grid\": {\n";
    o << "    \"dtype\": \"uint8\",\n";
    o << "    \"max\": " << (int)geo_max << ",\n";
    o << "    \"qparams\": [0.0, 0.0, 100.0],\n";
    o << "    \"meaning\": \"grid_x, grid_y in [0..max]; computed by shared span R\"\n";
    o << "  },\n";

    o << "  \"ivf\": {\n";
    o << "    \"centroids_ref\": \"use map.json/ivf_centroids.bin\",\n";
    o << "    \"postings_ref\": \"use map.json/ivf_postings_*.bin\",\n";
    o << "    \"map_json_ref\": \"../map/map.json\"\n";
    o << "  },\n";

    o << "  \"bins\": {\n";
    for (size_t i = 0; i < bins.size(); ++i) {
        const auto& b = bins[i];
        const size_t bytes = bytes_of(b.dtype, b.shape);
        o << "    \"" << b.key << "\": {\n";
        o << "      \"file\": \"" << json_escape(b.file) << "\",\n";
        o << "      \"dtype\": \"" << b.dtype << "\",\n";
        o << "      \"shape\": " << shape_json(b.shape) << ",\n";
        o << "      \"c_order\": true,\n";
        o << "      \"bytes\": " << bytes << "\n";
        o << "    }" << (i + 1 < bins.size() ? "," : "") << "\n";
    }
    o << "  }\n";
    o << "}\n";

    write_file_bytes(query_dir / "query.json", o.str().data(), o.str().size());
}

int main()
{
    try {
        const fs::path out_root = fs::path("test");
        const fs::path map_dir  = out_root / "map";
        const fs::path qry_dir  = out_root / "query";

        fs::create_directories(map_dir);
        fs::create_directories(qry_dir);

        // ---- parameters ----
        const uint32_t K = 1024;          // nlist (buckets)
        const uint32_t per_bucket = 16;   // each bucket has 16 map ids
        const uint32_t N = K * per_bucket;
        const uint32_t D = 128;           // centroid dim (aug128)
        const uint8_t  geo_max = 90;

        // deterministic RNG
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> q4dist(0, 15);
        std::uniform_int_distribution<int> geodist(0, geo_max);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

        // ---- map_desc_q4 (N,64) uint8 ----
        std::vector<uint8_t> map_desc_q4(static_cast<size_t>(N) * 64u);
        for (auto& x : map_desc_q4) x = static_cast<uint8_t>(q4dist(rng));

        // ---- map_geo_grid (N,2) uint8 ----
        std::vector<uint8_t> map_geo(static_cast<size_t>(N) * 2u);
        for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
            map_geo[i * 2 + 0] = static_cast<uint8_t>(geodist(rng));
            map_geo[i * 2 + 1] = static_cast<uint8_t>(geodist(rng));
        }

        // ---- ivf_centroids (K,D) float32 ----
        std::vector<float> centroids(static_cast<size_t>(K) * D);
        for (auto& x : centroids) x = fdist(rng);

        // ---- postings_offsets (K+1) uint32, postings_map_ids (K*16) int32 ----
        std::vector<uint32_t> offsets(static_cast<size_t>(K) + 1u, 0u);
        std::vector<int32_t>  map_ids(static_cast<size_t>(K) * per_bucket);

        uint32_t cur = 0;
        for (uint32_t lid = 0; lid < K; ++lid) {
            offsets[lid] = cur;
            for (uint32_t j = 0; j < per_bucket; ++j) {
                const uint32_t mid = lid * per_bucket + j; // global map_id
                map_ids[static_cast<size_t>(cur + j)] = static_cast<int32_t>(mid);
            }
            cur += per_bucket;
        }
        offsets[K] = cur;

        // ---- optional ivf extras (64) float32 ----
        std::vector<float> val_scale(64, 1.0f);
        std::vector<float> idf_w(64, 1.0f);
        for (int i = 0; i < 64; ++i) {
            val_scale[static_cast<size_t>(i)] = 1.0f;
            idf_w[static_cast<size_t>(i)] = 1.0f;
        }

        // write map bins
        write_file_vec(map_dir / "map_desc_q4.bin", map_desc_q4);
        write_file_vec(map_dir / "map_geo_grid.bin", map_geo);
        write_file_vec(map_dir / "ivf_centroids.bin", centroids);
        write_file_vec(map_dir / "ivf_postings_offsets.bin", offsets);
        write_file_vec(map_dir / "ivf_postings_map_ids.bin", map_ids);
        write_file_vec(map_dir / "ivf_val_scale.bin", val_scale);
        write_file_vec(map_dir / "ivf_idf_w.bin", idf_w);

        write_map_json(map_dir, N, K, D, geo_max);

        // ---- query: 5 descriptors in 1 step ----
        const size_t S = 1;
        const size_t Q = 5;

        std::vector<uint8_t> q_desc(static_cast<size_t>(Q) * 64u);
        for (auto& x : q_desc) x = static_cast<uint8_t>(q4dist(rng));

        std::vector<uint8_t> pose_grid(S * 2u, 0);
        pose_grid[0] = static_cast<uint8_t>(geodist(rng)); // gx
        pose_grid[1] = static_cast<uint8_t>(geodist(rng)); // gy

        std::vector<uint32_t> step_offsets = {0u, static_cast<uint32_t>(Q)};
        std::vector<uint16_t> step_counts  = {static_cast<uint16_t>(Q)};
        std::vector<int32_t>  relodo_i     = {0};
        std::vector<double>   t_now        = {0.0};

        write_file_vec(qry_dir / "query_desc_q4.bin", q_desc);
        write_file_vec(qry_dir / "query_step_pose_grid.bin", pose_grid);
        write_file_vec(qry_dir / "query_step_offsets.bin", step_offsets);
        write_file_vec(qry_dir / "query_step_counts.bin", step_counts);
        write_file_vec(qry_dir / "query_step_relodo_i.bin", relodo_i);
        write_file_vec(qry_dir / "query_step_t_now.bin", t_now);

        write_query_json(qry_dir, S, Q, geo_max);

        std::cout << "[ok] wrote testdata to:\n";
        std::cout << "  " << map_dir.string() << "\n";
        std::cout << "  " << qry_dir.string() << "\n";
        std::cout << "[map] K=" << K << " per_bucket=" << per_bucket << " N=" << N << " D=" << D << "\n";
        std::cout << "[qry] steps=" << S << " Q=" << Q << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}