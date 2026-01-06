#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

static void write_file(const fs::path& p, const void* data, size_t bytes) {
    std::ofstream ofs(p, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open for write: " + p.string());
    }
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!ofs) {
        throw std::runtime_error("Write failed: " + p.string());
    }
}

template <class T>
static void write_vec(const fs::path& p, const std::vector<T>& v) {
    write_file(p, v.data(), v.size() * sizeof(T));
}

static uint64_t file_size_bytes(const fs::path& p) {
    return fs::exists(p) ? static_cast<uint64_t>(fs::file_size(p)) : 0ULL;
}

static void write_map_json(
    const fs::path& out_dir,
    uint32_t desc_bits,
    uint32_t geo_max,
    double min_x,
    double min_y,
    double R,
    uint32_t N)
{
    const fs::path json_p = out_dir / "map.json";
    const fs::path desc_p = out_dir / "map_desc_q4.bin";
    const fs::path geo_p  = out_dir / "map_geo_grid.bin";

    const uint64_t desc_bytes = file_size_bytes(desc_p);
    const uint64_t geo_bytes  = file_size_bytes(geo_p);

    std::ofstream ofs(json_p);
    if (!ofs) throw std::runtime_error("Cannot open for write: " + json_p.string());

    ofs
    << "{\n"
    << "  \"version\": 1,\n"
    << "  \"endianness\": \"little\",\n"
    << "  \"desc_bits\": " << desc_bits << ",\n"
    << "  \"geo_grid\": {\n"
    << "    \"dtype\": \"uint8\",\n"
    << "    \"max\": " << geo_max << ",\n"
    << "    \"qparams\": [" << std::setprecision(17) << min_x << ", " << min_y << ", " << R << "],\n"
    << "    \"meaning\": \"grid_x, grid_y in [0..max]\"\n"
    << "  },\n"
    << "  \"bins\": {\n"
    << "    \"map_desc_q4\": {\n"
    << "      \"file\": \"map_desc_q4.bin\",\n"
    << "      \"dtype\": \"uint8\",\n"
    << "      \"shape\": [" << N << ", 64],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << desc_bytes << "\n"
    << "    },\n"
    << "    \"map_geo_grid\": {\n"
    << "      \"file\": \"map_geo_grid.bin\",\n"
    << "      \"dtype\": \"uint8\",\n"
    << "      \"shape\": [" << N << ", 2],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << geo_bytes << "\n"
    << "    }\n"
    << "  }\n"
    << "}\n";
}

static void write_query_json(
    const fs::path& out_dir,
    const std::string& session,
    uint32_t desc_bits,
    uint32_t geo_max,
    double min_x,
    double min_y,
    double R,
    uint32_t steps,
    uint32_t total_rows,
    const std::string& map_json_ref)
{
    const fs::path json_p = out_dir / "query.json";

    auto b = [&](const char* fname) -> uint64_t {
        return file_size_bytes(out_dir / fname);
    };

    std::ofstream ofs(json_p);
    if (!ofs) throw std::runtime_error("Cannot open for write: " + json_p.string());

    ofs
    << "{\n"
    << "  \"version\": 1,\n"
    << "  \"endianness\": \"little\",\n"
    << "  \"session\": " << "\"" << session << "\"" << ",\n"
    << "  \"desc_bits\": " << desc_bits << ",\n"
    << "  \"geo_grid\": {\n"
    << "    \"dtype\": \"uint8\",\n"
    << "    \"max\": " << geo_max << ",\n"
    << "    \"qparams\": [" << std::setprecision(17) << min_x << ", " << min_y << ", " << R << "],\n"
    << "    \"meaning\": \"grid_x, grid_y in [0..max]\"\n"
    << "  },\n"
    << "  \"ivf\": {\n"
    << "    \"map_json_ref\": " << "\"" << map_json_ref << "\"\n"
    << "  },\n"
    << "  \"bins\": {\n"
    << "    \"query_desc_q4\": {\n"
    << "      \"file\": \"query_desc_q4.bin\",\n"
    << "      \"dtype\": \"uint8\",\n"
    << "      \"shape\": [" << total_rows << ", 64],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << b("query_desc_q4.bin") << "\n"
    << "    },\n"
    << "    \"query_step_pose_grid\": {\n"
    << "      \"file\": \"query_step_pose_grid.bin\",\n"
    << "      \"dtype\": \"uint8\",\n"
    << "      \"shape\": [" << steps << ", 2],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << b("query_step_pose_grid.bin") << "\n"
    << "    },\n"
    << "    \"query_step_offsets\": {\n"
    << "      \"file\": \"query_step_offsets.bin\",\n"
    << "      \"dtype\": \"uint32\",\n"
    << "      \"shape\": [" << (steps + 1) << "],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << b("query_step_offsets.bin") << "\n"
    << "    },\n"
    << "    \"query_step_counts\": {\n"
    << "      \"file\": \"query_step_counts.bin\",\n"
    << "      \"dtype\": \"uint16\",\n"
    << "      \"shape\": [" << steps << "],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << b("query_step_counts.bin") << "\n"
    << "    },\n"
    << "    \"query_step_relodo_i\": {\n"
    << "      \"file\": \"query_step_relodo_i.bin\",\n"
    << "      \"dtype\": \"int32\",\n"
    << "      \"shape\": [" << steps << "],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << b("query_step_relodo_i.bin") << "\n"
    << "    },\n"
    << "    \"query_step_t_now\": {\n"
    << "      \"file\": \"query_step_t_now.bin\",\n"
    << "      \"dtype\": \"float64\",\n"
    << "      \"shape\": [" << steps << "],\n"
    << "      \"c_order\": true,\n"
    << "      \"bytes\": " << b("query_step_t_now.bin") << "\n"
    << "    }\n"
    << "  }\n"
    << "}\n";
}

int main(int argc, char** argv) {
    try {
        fs::path root = "testdata";
        if (argc >= 2) root = fs::path(argv[1]);

        const fs::path map_dir   = root / "map";
        const fs::path query_dir = root / "query";
        fs::create_directories(map_dir);
        fs::create_directories(query_dir);

        const uint32_t N = 512;
        const uint32_t Q = 4;
        const uint32_t steps = 1;

        const uint32_t desc_bits = 4;
        const uint32_t geo_max = 90;

        const double min_x = -100.0;
        const double min_y = -50.0;
        const double R     = 200.0;

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> d_q4(0, 15);
        std::uniform_int_distribution<int> d_geo(0, static_cast<int>(geo_max));

        std::vector<uint8_t> map_desc(static_cast<size_t>(N) * 64);
        std::vector<uint8_t> map_geo(static_cast<size_t>(N) * 2);

        for (uint32_t i = 0; i < N; ++i) {
            for (int k = 0; k < 64; ++k) {
                map_desc[static_cast<size_t>(i) * 64 + k] = static_cast<uint8_t>(d_q4(rng));
            }
            map_geo[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(d_geo(rng));
            map_geo[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(d_geo(rng));
        }

        const uint32_t hit_id = 123;
        const uint8_t hit_gx = map_geo[static_cast<size_t>(hit_id) * 2 + 0];
        const uint8_t hit_gy = map_geo[static_cast<size_t>(hit_id) * 2 + 1];

        std::vector<uint8_t> query_desc(static_cast<size_t>(Q) * 64);
        for (uint32_t qi = 0; qi < Q; ++qi) {
            std::memcpy(
                &query_desc[static_cast<size_t>(qi) * 64],
                &map_desc[static_cast<size_t>(hit_id) * 64],
                64);
        }

        std::vector<uint8_t> query_step_pose(static_cast<size_t>(steps) * 2);
        query_step_pose[0] = hit_gx;
        query_step_pose[1] = hit_gy;

        std::vector<uint32_t> offsets = {0, Q};
        std::vector<uint16_t> counts  = {static_cast<uint16_t>(Q)};
        std::vector<int32_t>  relodo_i = {0};
        std::vector<double>   t_now = {0.0};

        write_vec(map_dir / "map_desc_q4.bin", map_desc);
        write_vec(map_dir / "map_geo_grid.bin", map_geo);

        const std::string map_json_ref = (map_dir / "map.json").string();
        write_map_json(map_dir, desc_bits, geo_max, min_x, min_y, R, N);

        write_vec(query_dir / "query_desc_q4.bin", query_desc);
        write_vec(query_dir / "query_step_pose_grid.bin", query_step_pose);
        write_vec(query_dir / "query_step_offsets.bin", offsets);
        write_vec(query_dir / "query_step_counts.bin", counts);
        write_vec(query_dir / "query_step_relodo_i.bin", relodo_i);
        write_vec(query_dir / "query_step_t_now.bin", t_now);

        write_query_json(query_dir, "test_session", desc_bits, geo_max, min_x, min_y, R, steps, Q, map_json_ref);

        std::cout << "[gen] wrote:\n";
        std::cout << "  " << (map_dir / "map.json") << "\n";
        std::cout << "  " << (query_dir / "query.json") << "\n";
        std::cout << "  N=" << N << " Q=" << Q << " steps=" << steps
                  << " hit_id=" << hit_id
                  << " hit_gx=" << static_cast<int>(hit_gx)
                  << " hit_gy=" << static_cast<int>(hit_gy) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gen][ERROR] " << e.what() << "\n";
        return 1;
    }
}