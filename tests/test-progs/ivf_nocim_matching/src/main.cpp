// pf_kernel_ivf_dir_roi.cpp
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

#ifdef USE_M5
  #include <gem5/m5ops.h>
#else
  static inline void m5_reset_stats(uint64_t, uint64_t) {}
  static inline void m5_dump_stats (uint64_t, uint64_t) {}
  static inline void m5_work_begin (uint64_t, uint64_t) {}
  static inline void m5_work_end   (uint64_t, uint64_t) {}
#endif

static bool should_print_progress(uint64_t current, uint64_t total) {
    if (total == 0) return false;
    if (total <= 10) return true;

    const uint64_t interval = (total + 9) / 10;
    return current == total || (current % interval) == 0;
}

// ------------------------
// basic file utils
// ------------------------
static void read_exact(std::ifstream& f, void* buf, size_t n) {
    f.read(reinterpret_cast<char*>(buf), n);
    if (!f || (size_t)f.gcount() != n) throw std::runtime_error("read_exact failed");
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return (bool)f;
}

static uint64_t file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return (uint64_t)f.tellg();
}

template <class T>
static std::vector<T> load_bin_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    uint64_t sz = file_size(path);
    if (sz % sizeof(T) != 0) throw std::runtime_error("bad size for type: " + path);
    std::vector<T> v((size_t)(sz / sizeof(T)));
    if (!v.empty()) read_exact(f, v.data(), (size_t)sz);
    return v;
}

template <class T>
static std::vector<T> load_bin_vec_optional(const std::string& path) {
    if (!file_exists(path)) return {};
    return load_bin_vec<T>(path);
}

// ------------------------
// IMMEDIATE FIX:
// export_gem5.py writes offsets as uint32 (both map & query).
// Do NOT guess by file size divisibility.
// ------------------------
static std::vector<uint64_t> load_offsets_from_u32(const std::string& path) {
    auto v32 = load_bin_vec<uint32_t>(path);
    std::vector<uint64_t> v(v32.size());
    for (size_t i = 0; i < v32.size(); i++) v[i] = (uint64_t)v32[i];
    return v;
}

// ------------------------
// super tiny JSON key extractor (no dependency)
// only handles patterns like:  "alpha_value": 1.0
// ------------------------
static bool json_get_number(const std::string& json, const std::string& key, double& out) {
    std::string pat = "\"" + key + "\"";
    size_t p = json.find(pat);
    if (p == std::string::npos) return false;
    p = json.find(':', p);
    if (p == std::string::npos) return false;
    p++;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) p++;

    size_t e = p;
    while (e < json.size()) {
        char c = json[e];
        if (c == ',' || c == '}' || c == '\n' || c == '\r' || c == ' ' || c == '\t') break;
        e++;
    }
    if (e <= p) return false;
    try {
        out = std::stod(json.substr(p, e - p));
        return true;
    } catch (...) {
        return false;
    }
}

static std::string slurp_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

// ------------------------
// data structures
// ------------------------
struct IVFIndex {
    uint32_t K = 0;       // nlist
    uint32_t nprobe = 0;
    uint32_t top_k = 3;
    uint32_t Nmap = 0;

    float alpha = 1.0f;   // alpha_value in export
    float beta  = 5.0f;   // beta_presence in export
    uint32_t flags = 1u;  // bit0: use_idf

    // map arrays (exported)
    std::vector<uint8_t> map_xy;    // map_geo_grid.bin: Nmap*2 (uint8)
    std::vector<uint8_t> map_desc;  // map_desc_q4.bin : Nmap*64 (uint8)

    // IVF arrays (exported)
    std::vector<float> centroids;       // ivf_centroids.bin : K*D (float32)
    uint32_t D = 0;                     // centroid dimension
    std::vector<uint64_t> list_offsets; // ivf_postings_offsets.bin : K+1 (uint32 in export)
    std::vector<uint32_t> postings;     // ivf_postings_map_ids.bin : total (int32 in export)

    // optional weights (exported only for ZeroAware)
    std::vector<float> val_scale;   // ivf_val_scale.bin : (64,) float32
    std::vector<float> idf_w;       // ivf_idf_w.bin     : (64,) float32

    float wv[64]{};
    float wz[64]{};

    void build_weights() {
        float vs[64], iw[64];
        for (int d = 0; d < 64; d++) {
            vs[d] = (val_scale.size() == 64) ? val_scale[d] : 1.0f;
            iw[d] = (idf_w.size() == 64)     ? idf_w[d]     : 1.0f;
        }
        const bool use_idf = (flags & 1u) != 0;
        for (int d = 0; d < 64; d++) {
            const float idf = use_idf ? iw[d] : 1.0f;
            wv[d] = alpha * idf * vs[d];
            wz[d] = beta  * idf;
        }
    }
};

struct FrameQueries {
    uint32_t frame_id = 0;
    uint32_t Q = 0;
    uint8_t qrx = 0, qry = 0;
    const uint8_t* qdesc = nullptr; // Q*64
};

struct QueryPack {
    std::vector<uint8_t>  pose_grid;    // query_step_pose_grid.bin : S*2 uint8
    std::vector<uint32_t> step_counts;  // query_step_counts.bin    : S uint16 -> expand to u32
    std::vector<uint64_t> step_offsets; // query_step_offsets.bin   : S+1 uint32
    std::vector<uint8_t>  qdesc;        // query_desc_q4.bin        : totalQ*64 uint8
    uint32_t S = 0;
};

// ------------------------
// scoring
// ------------------------
static inline uint16_t score_equal_nonzero_64(const uint8_t* q, const uint8_t* c) {
    uint16_t s = 0;
    for (int d = 0; d < 64; d++) {
        const uint8_t cv = c[d];
        s += (cv != 0 && cv == q[d]) ? 1 : 0;
    }
    return s;
}

// ------------------------
// ROI kernel
// supports D=128 (zeroaware augmented) or D=64 (fallback)
// ------------------------
uint64_t matcher_ivf_roi(const IVFIndex& idx, const FrameQueries& fr) {
    const int qx = (int)fr.qrx;
    const int qy = (int)fr.qry;
    const uint32_t topk = std::min<uint32_t>(idx.top_k, 16u);

    uint64_t checksum = 0;
    std::vector<std::pair<float, uint32_t>> d2_lid;
    d2_lid.reserve(idx.K);

    std::vector<float> q_aug(idx.D);

    auto cmp_d2 = [](const auto& a, const auto& b){ return a.first < b.first; };

    for (uint32_t qi = 0; qi < fr.Q; qi++) {
        const uint8_t* q = &fr.qdesc[(size_t)qi * 64];

        if (idx.D == 128) {
            for (int d = 0; d < 64; d++) {
                const float qv = float(q[d]);
                const float z  = (q[d] != 0) ? 1.0f : 0.0f;
                q_aug[d]      = idx.wv[d] * qv;
                q_aug[64 + d] = idx.wz[d] * z;
            }
        } else if (idx.D == 64) {
            for (int d = 0; d < 64; d++) {
                const float qv = float(q[d]);
                q_aug[d] = idx.wv[d] * qv;
            }
        } else {
            throw std::runtime_error("unsupported centroid dimension D=" + std::to_string(idx.D));
        }

        d2_lid.clear();
        for (uint32_t lid = 0; lid < idx.K; lid++) {
            const float* c = &idx.centroids[(size_t)lid * idx.D];
            float d2 = 0.0f;
            for (uint32_t j = 0; j < idx.D; j++) {
                float diff = c[j] - q_aug[j];
                d2 += diff * diff;
            }
            d2_lid.push_back({d2, lid});
        }

        uint32_t take_lists = std::min<uint32_t>(idx.nprobe, idx.K);
        if (take_lists == 0) continue;

        // FIX: nth_element requires nth in [begin, end), so nth cannot be end().
        // If take_lists == K, we take all lists: just sort the whole vector.
        if (take_lists < idx.K) {
            std::nth_element(
                d2_lid.begin(), d2_lid.begin() + take_lists, d2_lid.end(), cmp_d2
            );
            d2_lid.resize(take_lists);
            std::sort(d2_lid.begin(), d2_lid.end(), cmp_d2);
        } else {
            // take_lists == K
            std::sort(d2_lid.begin(), d2_lid.end(), cmp_d2);
        }

        struct Best { uint16_t s; uint32_t gid; };
        Best best[16];
        uint32_t nb = 0;

        auto push_best = [&](uint16_t s, uint32_t gid) {
            uint32_t pos = nb;
            if (nb < topk) {
                best[nb++] = {s, gid};
            } else {
                Best worst = best[nb - 1];
                if (s < worst.s || (s == worst.s && gid >= worst.gid)) return;
                best[nb - 1] = {s, gid};
                pos = nb - 1;
            }
            while (pos > 0) {
                Best a = best[pos - 1];
                Best b = best[pos];
                if (a.s > b.s) break;
                if (a.s == b.s && a.gid <= b.gid) break;
                best[pos - 1] = b;
                best[pos] = a;
                pos--;
            }
        };

        // scan postings
        for (uint32_t i = 0; i < take_lists; i++) {
            const uint32_t lid = d2_lid[i].second;
            if (lid + 1 >= idx.list_offsets.size())
                throw std::runtime_error("list_offsets OOB at lid=" + std::to_string(lid));

            const uint64_t start = idx.list_offsets[lid];
            const uint64_t end   = idx.list_offsets[lid + 1];

            // extra guards: prevent postings OOB even if data is corrupted
            if (end < start)
                throw std::runtime_error("list_offsets not monotonic at lid=" + std::to_string(lid));
            if (end > (uint64_t)idx.postings.size())
                throw std::runtime_error("list_offsets exceeds postings.size at lid=" + std::to_string(lid));

            for (uint64_t p = start; p < end; p++) {
                const uint32_t gid = idx.postings[(size_t)p];
                if (gid >= idx.Nmap) continue;

                const uint8_t mx_u8 = idx.map_xy[(size_t)gid * 2 + 0];
                const uint8_t my_u8 = idx.map_xy[(size_t)gid * 2 + 1];
                const int mx = (int)mx_u8;
                const int my = (int)my_u8;

                // ROI window (+/-2 grid)
                if (mx < qx - 2 || mx > qx + 2 || my < qy - 2 || my > qy + 2) continue;

                const uint8_t* cdesc = &idx.map_desc[(size_t)gid * 64];
                const uint16_t s = score_equal_nonzero_64(q, cdesc);
                push_best(s, gid);
            }
        }

        for (uint32_t t = 0; t < nb; t++) {
            checksum ^= (uint64_t(best[t].gid) << (t % 16));
            checksum ^= (uint64_t(best[t].s)   << ((t + 7) % 16));
        }
    }

    return checksum;
}

// ------------------------
// loaders aligned with export_gem5.py
// ------------------------
static IVFIndex load_map_dir(const std::string& map_dir) {
    IVFIndex idx;

    // optional: read map.json for alpha/beta/use_idf (if present)
    if (file_exists(map_dir + "/map.json")) {
        std::string js = slurp_text(map_dir + "/map.json");
        double v;
        if (json_get_number(js, "alpha_value", v)) idx.alpha = (float)v;
        if (json_get_number(js, "beta_presence", v)) idx.beta  = (float)v;
        if (json_get_number(js, "use_idf", v)) idx.flags = ((int)v) ? 1u : 0u;
    } else {
        idx.alpha = 1.0f;
        idx.beta  = 5.0f;
        idx.flags = 1u;
    }

    // weights files optional
    idx.val_scale = load_bin_vec_optional<float>(map_dir + "/ivf_val_scale.bin");
    idx.idf_w     = load_bin_vec_optional<float>(map_dir + "/ivf_idf_w.bin");
    if (!idx.val_scale.empty() && idx.val_scale.size() != 64) throw std::runtime_error("ivf_val_scale must be 64 floats");
    if (!idx.idf_w.empty()     && idx.idf_w.size()     != 64) throw std::runtime_error("ivf_idf_w must be 64 floats");

    // centroids
    idx.centroids = load_bin_vec<float>(map_dir + "/ivf_centroids.bin");
    if (idx.centroids.empty()) throw std::runtime_error("ivf_centroids is empty");

    if (idx.centroids.size() % 128 == 0) idx.D = 128;
    else if (idx.centroids.size() % 64 == 0) idx.D = 64;
    else throw std::runtime_error("ivf_centroids size not divisible by 128 or 64 (cannot infer D)");

    idx.K = (uint32_t)(idx.centroids.size() / idx.D);
    if (idx.K == 0) throw std::runtime_error("K inferred as 0 (centroids corrupt?)");

    // OFFSETS: export is uint32(K+1). Immediate fix: read as uint32.
    idx.list_offsets = load_offsets_from_u32(map_dir + "/ivf_postings_offsets.bin");
    if (idx.list_offsets.size() != (size_t)idx.K + 1)
        throw std::runtime_error("ivf_postings_offsets size must be K+1");

    // postings map ids: int32(total) in export -> cast to u32
    auto p_i32 = load_bin_vec<int32_t>(map_dir + "/ivf_postings_map_ids.bin");
    idx.postings.resize(p_i32.size());
    for (size_t i = 0; i < p_i32.size(); i++) {
        if (p_i32[i] < 0) throw std::runtime_error("ivf_postings_map_ids contains negative id at i=" + std::to_string(i));
        idx.postings[i] = (uint32_t)p_i32[i];
    }

    // NEW: validate offsets monotonic and within postings
    if (idx.list_offsets.front() != 0)
        throw std::runtime_error("offsets[0] must be 0");
    for (uint32_t i = 0; i < idx.K; i++) {
        uint64_t a = idx.list_offsets[i];
        uint64_t b = idx.list_offsets[i + 1];
        if (b < a)
            throw std::runtime_error("offsets not monotonic at i=" + std::to_string(i));
        if (b > (uint64_t)idx.postings.size())
            throw std::runtime_error("offsets exceed postings.size at i=" + std::to_string(i));
    }
    if (idx.list_offsets.back() != (uint64_t)idx.postings.size())
        throw std::runtime_error("offsets.back != postings.size (mismatch)");

    // map arrays
    idx.map_xy   = load_bin_vec<uint8_t>(map_dir + "/map_geo_grid.bin"); // N*2
    idx.map_desc = load_bin_vec<uint8_t>(map_dir + "/map_desc_q4.bin");  // N*64
    if (idx.map_desc.size() % 64 != 0) throw std::runtime_error("map_desc_q4 size must be Nmap*64");
    idx.Nmap = (uint32_t)(idx.map_desc.size() / 64);
    if (idx.Nmap == 0) throw std::runtime_error("Nmap inferred as 0 (map_desc corrupt?)");
    if (idx.map_xy.size() != (size_t)idx.Nmap * 2) throw std::runtime_error("map_geo_grid size must be Nmap*2");

    // default params
    idx.nprobe = std::min<uint32_t>(idx.K, 32u);
    idx.top_k  = 3;

    idx.build_weights();
    return idx;
}

static QueryPack load_query_dir(const std::string& qdir) {
    QueryPack qp;

    qp.pose_grid = load_bin_vec<uint8_t>(qdir + "/query_step_pose_grid.bin"); // S*2

    // counts: export writes uint16
    auto c16 = load_bin_vec<uint16_t>(qdir + "/query_step_counts.bin");
    qp.step_counts.resize(c16.size());
    for (size_t i = 0; i < c16.size(); i++) qp.step_counts[i] = (uint32_t)c16[i];

    // OFFSETS: export writes uint32(S+1). Immediate fix: read as uint32.
    qp.step_offsets = load_offsets_from_u32(qdir + "/query_step_offsets.bin"); // S+1
    qp.qdesc        = load_bin_vec<uint8_t>(qdir + "/query_desc_q4.bin");      // totalQ*64

    qp.S = (uint32_t)qp.step_counts.size();
    if (qp.S == 0) throw std::runtime_error("S inferred as 0 (step_counts empty?)");
    if (qp.pose_grid.size() != (size_t)qp.S * 2) throw std::runtime_error("pose_grid size != S*2");
    if (qp.step_offsets.size() != (size_t)qp.S + 1) throw std::runtime_error("step_offsets size != S+1");

    // NEW: validate step_offsets monotonic
    if (qp.step_offsets.front() != 0)
        throw std::runtime_error("query step_offsets[0] must be 0");
    for (uint32_t s = 0; s < qp.S; s++) {
        uint64_t a = qp.step_offsets[s];
        uint64_t b = qp.step_offsets[s + 1];
        if (b < a)
            throw std::runtime_error("step_offsets not monotonic at step " + std::to_string(s));
    }

    const uint64_t totalQ = qp.step_offsets.back();
    if (qp.qdesc.size() != (size_t)totalQ * 64) {
        throw std::runtime_error(
            "query_desc_q4 size != totalQ*64. "
            "Your query_step_offsets might be wrong, or query_desc_q4.bin mismatch."
        );
    }

    // existing consistency check: (b-a) == counts
    for (uint32_t s = 0; s < qp.S; s++) {
        uint64_t a = qp.step_offsets[s];
        uint64_t b = qp.step_offsets[s + 1];
        uint64_t c = (uint64_t)qp.step_counts[s];
        if (b < a || (b - a) != c) {
            throw std::runtime_error("step_counts mismatch step_offsets at step " + std::to_string(s));
        }
    }
    return qp;
}

// ------------------------
// CLI
// ------------------------
static void usage() {
    std::cerr
      << "usage:\n"
      << "  ./pf_kernel_ivf_dir_roi <map_dir> <query_dir> [--max-steps N] [--nprobe N] [--topk K]\n"
      << "                           [--alpha A] [--beta B] [--use_idf 0|1]\n"
      << "example:\n"
      << "  ./pf_kernel_ivf_dir_roi export_gem5/2013-01-10/map export_gem5/2013-01-10/query --max-steps 100 --nprobe 32 --topk 3\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 1; }
    std::string map_dir = argv[1];
    std::string qdir    = argv[2];

    uint32_t override_nprobe = 0;
    uint32_t override_topk   = 0;
    bool override_use_idf = false; uint32_t use_idf_val = 1;
    bool override_alpha = false; float alpha_val = 1.0f;
    bool override_beta  = false; float beta_val  = 5.0f;
    uint64_t requested_steps = 0;
    bool has_requested_steps = false;

    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--max-steps" && i + 1 < argc) { requested_steps = std::stoull(argv[++i]); has_requested_steps = true; }
        else if (a == "--nprobe" && i + 1 < argc) override_nprobe = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--topk" && i + 1 < argc) override_topk = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--use_idf" && i + 1 < argc) { override_use_idf = true; use_idf_val = (uint32_t)std::stoul(argv[++i]); }
        else if (a == "--alpha" && i + 1 < argc)   { override_alpha = true; alpha_val = std::stof(argv[++i]); }
        else if (a == "--beta" && i + 1 < argc)    { override_beta  = true; beta_val  = std::stof(argv[++i]); }
        else { std::cerr << "unknown arg: " << a << "\n"; usage(); return 1; }
    }

    IVFIndex idx = load_map_dir(map_dir);
    QueryPack qp = load_query_dir(qdir);

    if (override_nprobe) idx.nprobe = std::min<uint32_t>(override_nprobe, idx.K);
    if (override_topk)   idx.top_k  = override_topk;
    if (override_use_idf) idx.flags = (use_idf_val ? 1u : 0u);
    if (override_alpha) idx.alpha = alpha_val;
    if (override_beta)  idx.beta  = beta_val;

    idx.build_weights();

    std::cerr << "[map] K=" << idx.K << " D=" << idx.D << " Nmap=" << idx.Nmap
              << " postings=" << idx.postings.size()
              << " alpha=" << idx.alpha << " beta=" << idx.beta
              << " use_idf=" << ((idx.flags & 1u) ? 1 : 0)
              << " nprobe=" << idx.nprobe << " top_k=" << idx.top_k << "\n";
    const uint64_t available_steps = qp.S;
    const uint64_t run_steps = has_requested_steps ?
        std::min<uint64_t>(requested_steps, available_steps) :
        available_steps;
    std::cerr << "[query] steps(S)=" << available_steps
              << " totalQ=" << qp.step_offsets.back()
              << " run_steps=" << run_steps << "\n";

    uint64_t total_cs = 0;
    const uint64_t totalQ = qp.step_offsets.back();

    for (uint64_t s = 0; s < run_steps; s++) {
        uint8_t qrx = qp.pose_grid[(size_t)s * 2 + 0];
        uint8_t qry = qp.pose_grid[(size_t)s * 2 + 1];
        uint32_t Q  = qp.step_counts[s];
        uint64_t offQ = qp.step_offsets[s];

        // NEW: runtime guard (extra safe; should always pass if files are consistent)
        if (offQ + (uint64_t)Q > totalQ) {
            throw std::runtime_error(
                "runtime guard failed: offQ+Q > totalQ at step " + std::to_string(s) +
                " (offQ=" + std::to_string(offQ) + ", Q=" + std::to_string(Q) +
                ", totalQ=" + std::to_string(totalQ) + ")"
            );
        }

        FrameQueries fr;
        fr.frame_id = s;
        fr.Q = Q;
        fr.qrx = qrx;
        fr.qry = qry;
        fr.qdesc = &qp.qdesc[(size_t)offQ * 64];

        m5_reset_stats(0, 0);
        m5_work_begin(1, (uint64_t)s);

        total_cs ^= matcher_ivf_roi(idx, fr);

        m5_work_end(1, (uint64_t)s);
        m5_dump_stats(0, 0);

        const uint64_t completed_steps = s + 1;
        if (should_print_progress(completed_steps, run_steps)) {
            std::cerr << "processed steps: "
                      << completed_steps << "/" << run_steps << "\n";
        }
    }

    std::cerr << "done. steps=" << run_steps << " checksum=" << total_cs << "\n";
    return 0;
}
