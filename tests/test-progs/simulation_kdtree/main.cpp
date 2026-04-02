// src/pf_kernel_roi.cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "nanoflann.hpp"

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

static void usage(const char* prog) {
    std::cerr << "usage: " << prog
              << " <map.bin> <frames.bin> [--max-steps N]\n";
}

struct MapData {
    uint32_t N = 0;
    float polevar = 0.f;
    float d_max = 1.f;
    float T_w_o[16]{};
    std::vector<float> xy; // 2*N
};

struct Frame {
    uint32_t frame_id = 0;
    uint32_t P = 0;
    uint32_t M = 0;
    uint64_t seed = 0;
    std::vector<float> particles_xyp; // 3*P
    std::vector<float> weights;       // P
    std::vector<float> obs_xy;        // 2*M
};

static void read_exact(std::ifstream& f, void* buf, size_t n) {
    f.read(reinterpret_cast<char*>(buf), n);
    if (!f || (size_t)f.gcount() != n) throw std::runtime_error("read_exact failed");
}

MapData load_map(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open map.bin");

    char magic[8];
    uint32_t ver, N;
    float polevar, dmax;
    read_exact(f, magic, 8);
    read_exact(f, &ver, 4);
    read_exact(f, &N, 4);
    read_exact(f, &polevar, 4);
    read_exact(f, &dmax, 4);

    if (std::memcmp(magic, "PFMAPv1\0", 8) != 0) throw std::runtime_error("bad map magic");
    if (ver != 1) throw std::runtime_error("bad map version");

    MapData m;
    m.N = N; m.polevar = polevar; m.d_max = dmax;
    read_exact(f, m.T_w_o, sizeof(float) * 16);

    m.xy.resize(size_t(2) * N);
    read_exact(f, m.xy.data(), sizeof(float) * m.xy.size());
    return m;
}

struct FramesReader {
    std::ifstream f;
    FramesReader(const std::string& path): f(path, std::ios::binary) {
        if (!f) throw std::runtime_error("cannot open frames.bin");
        char magic[8];
        uint32_t ver, reserved;
        read_exact(f, magic, 8);
        read_exact(f, &ver, 4);
        read_exact(f, &reserved, 4);
        if (std::memcmp(magic, "PFFRMSv1", 8) != 0) throw std::runtime_error("bad frames magic");
        if (ver != 1) throw std::runtime_error("bad frames version");
    }

    bool read_one(Frame& out) {
        if (!f) return false;

        uint32_t frame_id, P, M, flags;
        uint64_t seed = 0;

        f.read(reinterpret_cast<char*>(&frame_id), 4);
        if (!f) return false; // EOF ok

        read_exact(f, &P, 4);
        read_exact(f, &M, 4);
        read_exact(f, &flags, 4);
        if (flags & 1) read_exact(f, &seed, 8);

        out.frame_id = frame_id;
        out.P = P;
        out.M = M;
        out.seed = seed;

        out.particles_xyp.resize(size_t(3) * P);
        out.weights.resize(size_t(P));
        out.obs_xy.resize(size_t(2) * M);

        read_exact(f, out.particles_xyp.data(), sizeof(float) * out.particles_xyp.size());
        read_exact(f, out.weights.data(), sizeof(float) * out.weights.size());
        read_exact(f, out.obs_xy.data(), sizeof(float) * out.obs_xy.size());
        return true;
    }
};

static uint64_t count_total_frames(const std::string& path) {
    FramesReader reader(path);
    Frame frame;
    uint64_t total = 0;
    while (reader.read_one(frame)) {
        ++total;
    }
    return total;
}

struct PointCloud2f {
    const std::vector<float>* xy = nullptr; // [x0,y0,...]
    size_t kdtree_get_point_count() const { return xy->size() / 2; }
    float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return (*xy)[2 * idx + dim];
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

static inline float normal_pdf(float d, float sigma) {
    const float inv_sqrt_2pi = 0.3989422804014327f;
    float z = d / sigma;
    return (inv_sqrt_2pi / sigma) * std::exp(-0.5f * z * z);
}

// Return a checksum to prevent optimization
uint64_t measurement_update(
    const nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud2f>, PointCloud2f, 2
    >& kdtree,
    const MapData& map,
    Frame& fr
) {
    const float sigma = std::sqrt(map.polevar);
    const float dmax = map.d_max;

    for (uint32_t i = 0; i < fr.P; i++) {
        float x   = fr.particles_xyp[3*i + 0];
        float y   = fr.particles_xyp[3*i + 1];
        float yaw = fr.particles_xyp[3*i + 2];
        float c = std::cos(yaw), s = std::sin(yaw);

        float wmul = 1.0f;

        for (uint32_t j = 0; j < fr.M; j++) {
            float xr = fr.obs_xy[2*j + 0];
            float yr = fr.obs_xy[2*j + 1];

            float qw[2];
            qw[0] = x + c * xr - s * yr;
            qw[1] = y + s * xr + c * yr;

            size_t idx = size_t(-1);
            float dist2 = 0.f;
            nanoflann::KNNResultSet<float> rs(1);
            rs.init(&idx, &dist2);
            kdtree.findNeighbors(rs, qw, nanoflann::SearchParams());

            float d = std::sqrt(dist2);
            if (!(d <= dmax)) d = dmax;

            float p = normal_pdf(d, sigma);
            wmul *= (p + 0.2f);
        }

        fr.weights[i] *= wmul;
    }

    float sumw = 0.f;
    for (float w : fr.weights) sumw += w;
    if (sumw > 0) for (float& w : fr.weights) w /= sumw;

    // checksum: xor the bit patterns of the first few weights
    uint64_t cs = 0;
    const uint32_t take = std::min<uint32_t>(fr.P, 16);
    for (uint32_t i = 0; i < take; i++) {
        uint32_t bits;
        static_assert(sizeof(float) == sizeof(uint32_t));
        std::memcpy(&bits, &fr.weights[i], sizeof(uint32_t));
        cs ^= (uint64_t(bits) << (i % 8));
    }
    return cs;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    uint64_t requested_steps = 0;
    bool has_requested_steps = false;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--max-steps" && i + 1 < argc) {
            requested_steps = std::stoull(argv[++i]);
            has_requested_steps = true;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    MapData map = load_map(argv[1]);

    PointCloud2f pc{&map.xy};
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud2f>, PointCloud2f, 2
    >;
    KDTree kdtree(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(16));
    kdtree.buildIndex();

    const uint64_t available_frames = count_total_frames(argv[2]);
    const uint64_t total_frames = has_requested_steps ?
        std::min<uint64_t>(requested_steps, available_frames) :
        available_frames;
    std::cerr << "[info] frames=" << available_frames
              << " run_frames=" << total_frames << "\n";
    FramesReader rdr(argv[2]);
    Frame fr;

    uint64_t total_checksum = 0;
    size_t cnt = 0;

    while (cnt < total_frames && rdr.read_one(fr)) {
        // ROI: per-frame kernel only
        m5_reset_stats(0, 0);
        m5_work_begin(1, static_cast<uint64_t>(fr.frame_id));

        total_checksum ^= measurement_update(kdtree, map, fr);

        m5_work_end(1, static_cast<uint64_t>(fr.frame_id));
        m5_dump_stats(0, 0);

        cnt++;
        if (should_print_progress(cnt, total_frames)) {
            std::cerr << "processed frames: "
                      << cnt << "/" << total_frames << "\n";
        }
    }

    std::cerr << "done. frames=" << cnt << " checksum=" << total_checksum << "\n";
    return 0;
}
