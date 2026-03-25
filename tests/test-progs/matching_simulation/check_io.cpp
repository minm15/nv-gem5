#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

static const std::string MAP_DIR   = "tests/test-progs/export_gem5/kdtree_matching/2013-01-10/map";
static const std::string QUERY_DIR = "tests/test-progs/export_gem5/kdtree_matching/2013-01-10/query";

struct BinMeta {
    std::string key;     // e.g. "map_desc_q4"
    std::string file;    // e.g. "map_desc_q4.bin"
    std::string dtype;   // e.g. "uint8"
    std::vector<int64_t> shape; // e.g. [N,64]
    int64_t bytes = -1;
};

static std::string read_text(const fs::path& p) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("Cannot open: " + p.string());
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

static int64_t file_size_bytes(const fs::path& p) {
    if (!fs::exists(p)) return -1;
    return static_cast<int64_t>(fs::file_size(p));
}

static int64_t dtype_bytes(const std::string& dt) {
    if (dt == "uint8") return 1;
    if (dt == "int8") return 1;
    if (dt == "uint16") return 2;
    if (dt == "int16") return 2;
    if (dt == "uint32") return 4;
    if (dt == "int32") return 4;
    if (dt == "float32") return 4;
    if (dt == "float64") return 8;
    throw std::runtime_error("Unsupported dtype: " + dt);
}

static int64_t prod_shape(const std::vector<int64_t>& sh) {
    int64_t n = 1;
    for (auto v : sh) n *= v;
    return n;
}

static std::vector<int64_t> parse_int_list(const std::string& s) {
    // parse numbers from something like: [ 1105164, 64 ]
    std::vector<int64_t> out;
    std::regex num_re(R"((-?\d+))");
    auto it = std::sregex_iterator(s.begin(), s.end(), num_re);
    auto ed = std::sregex_iterator();
    for (; it != ed; ++it) out.push_back(std::stoll((*it)[1].str()));
    return out;
}

static BinMeta find_bin_meta(const std::string& json_text, const std::string& key) {
    // find block like:  "map_desc_q4": { ... "file": "...", "dtype": "...", "shape": [...], "bytes": ... }
    // non-greedy matching inside braces
    std::regex block_re("\"" + key + R"("\s*:\s*\{([\s\S]*?)\}\s*,?)", std::regex::icase);
    std::smatch m;
    if (!std::regex_search(json_text, m, block_re)) {
        throw std::runtime_error("Cannot find bins entry for key: " + key);
    }
    std::string block = m[1].str();

    auto get_str = [&](const std::string& k) -> std::string {
        std::regex re("\"" + k + R"("\s*:\s*")" + R"(([^"]+))" + R"(")", std::regex::icase);
        std::smatch mm;
        if (!std::regex_search(block, mm, re)) throw std::runtime_error("Missing field '" + k + "' for key " + key);
        return mm[1].str();
    };

    auto get_shape = [&]() -> std::vector<int64_t> {
        std::regex re(R"("shape"\s*:\s*\[([\s\S]*?)\])", std::regex::icase);
        std::smatch mm;
        if (!std::regex_search(block, mm, re)) throw std::runtime_error("Missing field 'shape' for key " + key);
        return parse_int_list(mm[1].str());
    };

    auto get_bytes = [&]() -> int64_t {
        std::regex re(R"("bytes"\s*:\s*(\d+))", std::regex::icase);
        std::smatch mm;
        if (!std::regex_search(block, mm, re)) return -1;
        return std::stoll(mm[1].str());
    };

    BinMeta bm;
    bm.key = key;
    bm.file = get_str("file");
    bm.dtype = get_str("dtype");
    bm.shape = get_shape();
    bm.bytes = get_bytes();
    return bm;
}

template <typename T>
static void read_all(const fs::path& p, std::vector<T>& out, int64_t n_elems) {
    out.resize(static_cast<size_t>(n_elems));
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open bin: " + p.string());
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n_elems * (int64_t)sizeof(T)));
    if (!in) throw std::runtime_error("Short read: " + p.string());
}

static void ok(const std::string& s){ std::cout << "[ OK ] " << s << "\n"; }
static void info(const std::string& s){ std::cout << "[info] " << s << "\n"; }

static void check_bin_basic(const fs::path& dir, const BinMeta& bm) {
    fs::path p = dir / bm.file;
    if (!fs::exists(p)) throw std::runtime_error("Missing file: " + p.string());

    int64_t expected = prod_shape(bm.shape) * dtype_bytes(bm.dtype);
    int64_t actual = file_size_bytes(p);

    if (bm.bytes != -1 && bm.bytes != expected) {
        throw std::runtime_error("JSON bytes mismatch for " + bm.file + ": json=" + std::to_string(bm.bytes) +
                                 " computed=" + std::to_string(expected));
    }
    if (actual != expected) {
        throw std::runtime_error("File size mismatch for " + bm.file + ": file=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
    ok(bm.key + " -> " + bm.file + " size OK (" + std::to_string(expected) + " bytes)");
}

static void check_map(const fs::path& map_dir) {
    info("MAP dir: " + map_dir.string());
    std::string jtxt = read_text(map_dir / "map.json");

    // bins we care
    BinMeta map_desc = find_bin_meta(jtxt, "map_desc_q4");
    BinMeta map_geo  = find_bin_meta(jtxt, "map_geo_grid");
    BinMeta cen      = find_bin_meta(jtxt, "ivf_centroids");
    BinMeta off      = find_bin_meta(jtxt, "ivf_postings_offsets");
    BinMeta pid      = find_bin_meta(jtxt, "ivf_postings_map_ids");
    BinMeta vs       = find_bin_meta(jtxt, "ivf_val_scale");
    BinMeta idf      = find_bin_meta(jtxt, "ivf_idf_w");

    // basic
    check_bin_basic(map_dir, map_desc);
    check_bin_basic(map_dir, map_geo);
    check_bin_basic(map_dir, cen);
    check_bin_basic(map_dir, off);
    check_bin_basic(map_dir, pid);
    check_bin_basic(map_dir, vs);
    check_bin_basic(map_dir, idf);

    // load small arrays / needed checks
    const int64_t N = map_desc.shape.at(0);
    const int64_t Kp1 = off.shape.at(0);
    const int64_t K = Kp1 - 1;

    // offsets uint32
    std::vector<uint32_t> offsets;
    read_all<uint32_t>(map_dir / off.file, offsets, Kp1);

    // check monotonic, offsets[-1] == postings length
    bool mono = true;
    for (int64_t i = 1; i < Kp1; i++) if (offsets[i] < offsets[i-1]) { mono = false; break; }
    if (!mono) throw std::runtime_error("map offsets not monotonic");
    if (offsets[0] != 0) throw std::runtime_error("map offsets[0] != 0");
    ok("IVF offsets monotonic and starts at 0");

    const int64_t postings_len = pid.shape.at(0);
    if ((int64_t)offsets.back() != postings_len) {
        throw std::runtime_error("map offsets[-1] != postings_len: " + std::to_string(offsets.back()) +
                                 " vs " + std::to_string(postings_len));
    }
    ok("IVF offsets[-1] matches postings length");

    // postings map_ids int32: range [0, N-1]
    std::vector<int32_t> map_ids;
    read_all<int32_t>(map_dir / pid.file, map_ids, postings_len);

    int32_t mn = INT32_MAX, mx = INT32_MIN;
    for (auto v : map_ids) { mn = std::min(mn, v); mx = std::max(mx, v); }
    if (mn < 0 || mx >= (int32_t)N) {
        throw std::runtime_error("postings_map_ids out of range: [" + std::to_string(mn) + "," + std::to_string(mx) +
                                 "] vs N=" + std::to_string(N));
    }
    ok("postings_map_ids range OK");

    // descriptor q4 range check (stream read to avoid huge RAM)
    {
        fs::path p = map_dir / map_desc.file;
        std::ifstream in(p, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open: " + p.string());
        const size_t buf_elems = 1 << 20; // 1M bytes for uint8
        std::vector<uint8_t> buf(buf_elems);

        uint8_t vmin = 255, vmax = 0;
        while (in) {
            in.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size());
            std::streamsize got = in.gcount();
            if (got <= 0) break;
            for (std::streamsize i = 0; i < got; i++) {
                uint8_t v = buf[(size_t)i];
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
        }
        if (vmax > 15) throw std::runtime_error("map_desc_q4 vmax > 15: " + std::to_string((int)vmax));
        ok("map_desc_q4 range OK [0..15], observed min=" + std::to_string((int)vmin) +
           " max=" + std::to_string((int)vmax));
    }

    // geo grid max check (also stream)
    {
        fs::path p = map_dir / map_geo.file;
        std::ifstream in(p, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open: " + p.string());
        const size_t buf_elems = 1 << 20;
        std::vector<uint8_t> buf(buf_elems);
        uint8_t vmax = 0;
        while (in) {
            in.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size());
            std::streamsize got = in.gcount();
            if (got <= 0) break;
            for (std::streamsize i = 0; i < got; i++) vmax = std::max(vmax, buf[(size_t)i]);
        }
        ok("map_geo_grid observed max=" + std::to_string((int)vmax) + " (uint8)");
    }

    info("MAP checks done. N=" + std::to_string(N) + ", K=" + std::to_string(K) +
         ", postings=" + std::to_string(postings_len));
}

static void check_query(const fs::path& query_dir) {
    info("QUERY dir: " + query_dir.string());
    std::string jtxt = read_text(query_dir / "query.json");

    BinMeta qdesc = find_bin_meta(jtxt, "query_desc_q4");
    BinMeta qpose = find_bin_meta(jtxt, "query_step_pose_grid");
    BinMeta qoff  = find_bin_meta(jtxt, "query_step_offsets");
    BinMeta qcnt  = find_bin_meta(jtxt, "query_step_counts");
    BinMeta qrid  = find_bin_meta(jtxt, "query_step_relodo_i");

    check_bin_basic(query_dir, qdesc);
    check_bin_basic(query_dir, qpose);
    check_bin_basic(query_dir, qoff);
    check_bin_basic(query_dir, qcnt);
    check_bin_basic(query_dir, qrid);

    const int64_t Q = qdesc.shape.at(0);
    const int64_t S = qpose.shape.at(0);

    // load offsets/cnts/poses
    std::vector<uint32_t> offsets;
    read_all<uint32_t>(query_dir / qoff.file, offsets, qoff.shape.at(0));
    if ((int64_t)offsets.size() != S + 1) {
        throw std::runtime_error("query offsets length != S+1: " + std::to_string(offsets.size()) +
                                 " vs " + std::to_string(S+1));
    }
    if (offsets[0] != 0) throw std::runtime_error("query offsets[0] != 0");
    for (int64_t i = 1; i < (int64_t)offsets.size(); i++) {
        if (offsets[i] < offsets[i-1]) throw std::runtime_error("query offsets not monotonic");
    }
    if ((int64_t)offsets.back() != Q) {
        throw std::runtime_error("query offsets[-1] != Q: " + std::to_string(offsets.back()) +
                                 " vs " + std::to_string(Q));
    }
    ok("query offsets monotonic, offsets[-1]==Q");

    std::vector<uint16_t> counts;
    read_all<uint16_t>(query_dir / qcnt.file, counts, qcnt.shape.at(0));
    if ((int64_t)counts.size() != S) throw std::runtime_error("query counts length != S");
    uint64_t sum = 0;
    for (auto c : counts) sum += c;
    if ((int64_t)sum != Q) {
        throw std::runtime_error("sum(counts) != Q: " + std::to_string(sum) + " vs " + std::to_string(Q));
    }
    ok("query counts sum == Q");

    // offsets diffs == counts
    for (int64_t i = 0; i < S; i++) {
        uint32_t diff = offsets[i+1] - offsets[i];
        if (diff != counts[i]) {
            throw std::runtime_error("offset diff != count at step " + std::to_string(i) +
                                     ": diff=" + std::to_string(diff) + " count=" + std::to_string(counts[i]));
        }
    }
    ok("offset diffs match counts per step");

    // descriptor q4 range check (stream)
    {
        fs::path p = query_dir / qdesc.file;
        std::ifstream in(p, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open: " + p.string());
        const size_t buf_elems = 1 << 20;
        std::vector<uint8_t> buf(buf_elems);
        uint8_t vmin = 255, vmax = 0;
        while (in) {
            in.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size());
            std::streamsize got = in.gcount();
            if (got <= 0) break;
            for (std::streamsize i = 0; i < got; i++) {
                uint8_t v = buf[(size_t)i];
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
        }
        if (vmax > 15) throw std::runtime_error("query_desc_q4 vmax > 15: " + std::to_string((int)vmax));
        ok("query_desc_q4 range OK [0..15], observed min=" + std::to_string((int)vmin) +
           " max=" + std::to_string((int)vmax));
    }

    // pose grid range check (uint8)
    {
        fs::path p = query_dir / qpose.file;
        std::ifstream in(p, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open: " + p.string());
        const size_t buf_elems = 1 << 20;
        std::vector<uint8_t> buf(buf_elems);
        uint8_t vmax = 0;
        while (in) {
            in.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size());
            std::streamsize got = in.gcount();
            if (got <= 0) break;
            for (std::streamsize i = 0; i < got; i++) vmax = std::max(vmax, buf[(size_t)i]);
        }
        ok("query_step_pose_grid observed max=" + std::to_string((int)vmax));
    }

    info("QUERY checks done. Steps S=" + std::to_string(S) + ", total desc rows Q=" + std::to_string(Q));
}

int main() {
    try {
        fs::path map_dir(MAP_DIR);
        fs::path query_dir(QUERY_DIR);

        if (!fs::exists(map_dir / "map.json")) throw std::runtime_error("Missing map.json in " + map_dir.string());
        if (!fs::exists(query_dir / "query.json")) throw std::runtime_error("Missing query.json in " + query_dir.string());

        std::cout << "=== matching_simulation check_io ===\n";
        check_map(map_dir);
        std::cout << "\n";
        check_query(query_dir);
        std::cout << "\n✅ ALL C++ CHECKS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}