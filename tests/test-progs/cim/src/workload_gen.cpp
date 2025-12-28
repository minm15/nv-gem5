#include "workload_gen.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace lab {

namespace {

#pragma pack(push, 1)
struct MapRecord {
    uint8_t desc[kDims];
    uint8_t x;
    uint8_t y;
};
struct QueryRecord {
    uint8_t desc[kDims];
    uint8_t rx;
    uint8_t ry;
    uint8_t range;
};
#pragma pack(pop)

static bool file_ok_size(const std::string& path, uint64_t expected_bytes)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) return false;
    const auto sz = static_cast<uint64_t>(f.tellg());
    return sz == expected_bytes;
}

static void write_map_bin(const std::string& path, size_t M, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> val4(0, 15);
    std::uniform_int_distribution<int> val8(0, 255);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open " + path + " for write");

    MapRecord rec{};
    for (size_t i = 0; i < M; ++i) {
        for (int d = 0; d < kDims; ++d) rec.desc[d] = static_cast<uint8_t>(val4(rng));
        rec.x = static_cast<uint8_t>(val8(rng));
        rec.y = static_cast<uint8_t>(val8(rng));
        out.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    }
}

static void write_query_bin(const std::string& path, size_t Q, uint32_t seed, uint8_t default_range)
{
    std::mt19937 rng(seed ^ 0x9e3779b9u);
    std::uniform_int_distribution<int> val4(0, 15);
    std::uniform_int_distribution<int> val8(0, 255);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open " + path + " for write");

    QueryRecord rec{};
    for (size_t i = 0; i < Q; ++i) {
        for (int d = 0; d < kDims; ++d) rec.desc[d] = static_cast<uint8_t>(val4(rng));
        rec.rx = static_cast<uint8_t>(val8(rng));
        rec.ry = static_cast<uint8_t>(val8(rng));
        rec.range = default_range;
        out.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    }
}

static void load_map_bin(const std::string& path, size_t M, Workload& wl)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path + " for read");

    wl.map_desc.resize(M);
    wl.map_x.resize(M);
    wl.map_y.resize(M);

    MapRecord rec{};
    for (size_t i = 0; i < M; ++i) {
        in.read(reinterpret_cast<char*>(&rec), sizeof(rec));
        if (!in) throw std::runtime_error("short read in " + path);
        for (int d = 0; d < kDims; ++d) wl.map_desc[i][d] = static_cast<uint8_t>(rec.desc[d] & 0x0Fu);
        wl.map_x[i] = rec.x;
        wl.map_y[i] = rec.y;
    }
}

static void load_query_bin(const std::string& path, size_t Q, Workload& wl)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path + " for read");

    wl.query_desc.resize(Q);
    wl.query_rx.resize(Q);
    wl.query_ry.resize(Q);
    wl.query_range.resize(Q);

    QueryRecord rec{};
    for (size_t i = 0; i < Q; ++i) {
        in.read(reinterpret_cast<char*>(&rec), sizeof(rec));
        if (!in) throw std::runtime_error("short read in " + path);
        for (int d = 0; d < kDims; ++d) wl.query_desc[i][d] = static_cast<uint8_t>(rec.desc[d] & 0x0Fu);
        wl.query_rx[i] = rec.rx;
        wl.query_ry[i] = rec.ry;
        wl.query_range[i] = rec.range;
    }
}

} // namespace

Workload load_or_make_bin_workload(
    size_t M,
    size_t Q,
    uint32_t seed,
    uint8_t default_range,
    const std::string& map_path,
    const std::string& query_path)
{
    // Validate / create binaries.
    const uint64_t map_expected = static_cast<uint64_t>(M) * sizeof(MapRecord);
    const uint64_t qry_expected = static_cast<uint64_t>(Q) * sizeof(QueryRecord);

    if (!file_ok_size(map_path, map_expected)) {
        write_map_bin(map_path, M, seed);
    }
    if (!file_ok_size(query_path, qry_expected)) {
        write_query_bin(query_path, Q, seed, default_range);
    }

    Workload wl;
    wl.M = M;
    wl.Q = Q;

    load_map_bin(map_path, M, wl);
    load_query_bin(query_path, Q, wl);
    return wl;
}

} // namespace lab
