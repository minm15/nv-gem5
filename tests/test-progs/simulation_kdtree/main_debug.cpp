// pf_debug_random_frames.cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <chrono>

#include "nanoflann/include/nanoflann.hpp"

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
    std::vector<float> particles_xyp; // 3*P : x,y,yaw
    std::vector<float> weights;       // P
    std::vector<float> obs_xy;        // 2*M : xr, yr
};

// ---------- Binary reader helpers ----------
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

// ---------- KD-tree adaptor for nanoflann ----------
struct PointCloud2f {
    const std::vector<float>* xy = nullptr; // [x0,y0,x1,y1,...]
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

static float sum_weights(const std::vector<float>& w) {
    float s = 0.f;
    for (float x : w) s += x;
    return s;
}

static float neff(const std::vector<float>& w) {
    // Python: 1 / (sum(w^2) * count)
    // 這裡 count = P
    float s2 = 0.f;
    for (float x : w) s2 += x * x;
    if (s2 <= 0.f) return 0.f;
    return 1.f / (s2 * float(w.size()));
}

static void minmax_weights(const std::vector<float>& w, float& mn, float& mx) {
    mn = +std::numeric_limits<float>::infinity();
    mx = -std::numeric_limits<float>::infinity();
    for (float x : w) {
        mn = std::min(mn, x);
        mx = std::max(mx, x);
    }
}

// 2/3/4 kernel: observation transform + NN + weighting + normalize
template <typename KDTreeT>
void measurement_update(
    const KDTreeT& kdtree,
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

            // (2) transform obs -> world (SE2)
            float qw[2];
            qw[0] = x + c * xr - s * yr;
            qw[1] = y + s * xr + c * yr;

            // (3) NN query
            size_t idx = size_t(-1);
            float dist2 = 0.f;
            nanoflann::KNNResultSet<float> rs(1);
            rs.init(&idx, &dist2);
            kdtree.findNeighbors(rs, qw, nanoflann::SearchParameters());

            float d = std::sqrt(dist2);
            if (!(d <= dmax)) d = dmax;

            // (4) weight update
            float p = normal_pdf(d, sigma);
            wmul *= (p + 0.2f);
        }

        fr.weights[i] *= wmul;
    }

    // normalize
    float sumw = sum_weights(fr.weights);
    if (sumw > 0) {
        for (float& w : fr.weights) w /= sumw;
    }
}

static void print_one_transform_example(const Frame& fr) {
    if (fr.P == 0 || fr.M == 0) return;
    float x   = fr.particles_xyp[0];
    float y   = fr.particles_xyp[1];
    float yaw = fr.particles_xyp[2];
    float c = std::cos(yaw), s = std::sin(yaw);
    float xr = fr.obs_xy[0];
    float yr = fr.obs_xy[1];
    float xw = x + c * xr - s * yr;
    float yw = y + s * xr + c * yr;
    std::cerr << "  example: particle0(x,y,yaw)=("
              << x << "," << y << "," << yaw << "), obs0=("
              << xr << "," << yr << ") -> world=("
              << xw << "," << yw << ")\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ./pf_debug <map.bin> <frames.bin> [K=5] [seed=123]\n";
        return 1;
    }
    int K = (argc >= 4) ? std::stoi(argv[3]) : 5;
    uint64_t seed = (argc >= 5) ? (uint64_t)std::stoull(argv[4]) : 123ULL;
    if (K <= 0) K = 1;

    MapData map = load_map(argv[1]);

    PointCloud2f pc{&map.xy};
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud2f>, PointCloud2f, 2
    >;
    KDTree kdtree(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(16));
    kdtree.buildIndex();

    FramesReader rdr(argv[2]);

    // Reservoir sampling: 隨機抽 K 個 frame（等機率）
    std::mt19937_64 rng(seed);
    std::vector<Frame> sampled;
    sampled.reserve((size_t)K);

    Frame fr;
    uint64_t seen = 0;
    while (rdr.read_one(fr)) {
        seen++;
        if ((int)sampled.size() < K) {
            sampled.push_back(fr);
        } else {
            std::uniform_int_distribution<uint64_t> dist(1, seen);
            uint64_t r = dist(rng);
            if (r <= (uint64_t)K) {
                sampled[(size_t)(r - 1)] = fr;
            }
        }
    }

    std::cerr << "total frames in file: " << seen << "\n";
    std::cerr << "sampled frames: " << sampled.size() << " (K=" << K << ", seed=" << seed << ")\n\n";

    // Debug each sampled frame
    for (size_t t = 0; t < sampled.size(); t++) {
        Frame& f = sampled[t];

        uint64_t transforms = (uint64_t)f.P * (uint64_t)f.M;

        float sumw0 = sum_weights(f.weights);
        float neff0 = neff(f.weights);
        float mn0, mx0; minmax_weights(f.weights, mn0, mx0);

        std::cerr << "=== sample[" << t << "] frame_id=" << f.frame_id
                  << " P=" << f.P << " M=" << f.M
                  << " transformations(P*M)=" << transforms << " ===\n";
        std::cerr << "  weights(before): sum=" << sumw0
                  << " neff=" << neff0
                  << " min=" << mn0 << " max=" << mx0 << "\n";

        print_one_transform_example(f);

        auto t0 = std::chrono::high_resolution_clock::now();
        measurement_update(kdtree, map, f);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        float sumw1 = sum_weights(f.weights);
        float neff1 = neff(f.weights);
        float mn1, mx1; minmax_weights(f.weights, mn1, mx1);

        std::cerr << "  weights(after ): sum=" << sumw1
                  << " neff=" << neff1
                  << " min=" << mn1 << " max=" << mx1 << "\n";
        std::cerr << "  measurement_update time ~ " << us << " us (host timing)\n\n";
    }

    return 0;
}
