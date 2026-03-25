#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>

#ifdef USE_M5
  #include <gem5/m5ops.h>
#else
  static inline void m5_reset_stats(uint64_t, uint64_t) {}
  static inline void m5_dump_stats (uint64_t, uint64_t) {}
  static inline void m5_work_begin (uint64_t, uint64_t) {}
  static inline void m5_work_end   (uint64_t, uint64_t) {}
#endif

static void read_exact(std::ifstream& f, void* buf, size_t n) {
    f.read(reinterpret_cast<char*>(buf), n);
    if (!f || (size_t)f.gcount() != n) throw std::runtime_error("read_exact failed");
}

struct IVFIndex {
    uint32_t K = 0;
    uint32_t nprobe = 0;
    uint32_t top_k = 3;
    uint32_t Nmap = 0;

    float alpha = 1.0f;
    float beta  = 5.0f;
    uint32_t flags = 0; // bit0: use_idf

    // arrays
    std::vector<float> val_scale;   // 64
    std::vector<float> idf_w;       // 64
    std::vector<float> centroids;   // K*128

    std::vector<uint8_t> map_xy;    // Nmap*2
    std::vector<uint8_t> map_desc;  // Nmap*64

    std::vector<uint32_t> list_sizes;   // K
    std::vector<uint64_t> list_offsets; // K+1 prefix sum
    std::vector<uint32_t> postings;     // sum(list_sizes)

    // precomputed weights for q_aug building
    float wv[64]; // alpha * idf * val_scale
    float wz[64]; // beta  * idf

    void build_weights() {
        const bool use_idf = (flags & 1u) != 0;
        for (int d = 0; d < 64; d++) {
            const float idf = use_idf ? idf_w[d] : 1.0f;
            wv[d] = alpha * idf * val_scale[d];
            wz[d] = beta  * idf;
        }
    }
};

struct FrameQueries {
    uint32_t frame_id = 0;
    uint32_t Q = 0;
    uint8_t qrx = 0, qry = 0;
    std::vector<uint8_t> qdesc; // Q*64
};

static inline uint16_t score_equal_nonzero_64(const uint8_t* q, const uint8_t* c) {
    uint16_t s = 0;
    for (int d = 0; d < 64; d++) {
        const uint8_t cv = c[d];
        s += (cv != 0 && cv == q[d]) ? 1 : 0;
    }
    return s;
}

static IVFIndex read_ivf_header(std::ifstream& f) {
    IVFIndex idx;

    char magic[8];
    uint32_t ver = 0;
    read_exact(f, magic, 8);
    read_exact(f, &ver, 4);

    if (std::memcmp(magic, "PFIVFv1\0", 8) != 0) throw std::runtime_error("bad magic");
    if (ver != 1) throw std::runtime_error("bad version");

    read_exact(f, &idx.K, 4);
    read_exact(f, &idx.nprobe, 4);
    read_exact(f, &idx.top_k, 4);

    read_exact(f, &idx.Nmap, 4);

    read_exact(f, &idx.alpha, 4);
    read_exact(f, &idx.beta, 4);
    read_exact(f, &idx.flags, 4);

    idx.val_scale.resize(64);
    idx.idf_w.resize(64);
    read_exact(f, idx.val_scale.data(), 64 * sizeof(float));
    read_exact(f, idx.idf_w.data(),     64 * sizeof(float));

    // centroids: K*128 float
    idx.centroids.resize(size_t(idx.K) * 128);
    read_exact(f, idx.centroids.data(), idx.centroids.size() * sizeof(float));

    // map arrays
    idx.map_xy.resize(size_t(idx.Nmap) * 2);
    idx.map_desc.resize(size_t(idx.Nmap) * 64);
    read_exact(f, idx.map_xy.data(),   idx.map_xy.size() * sizeof(uint8_t));
    read_exact(f, idx.map_desc.data(), idx.map_desc.size() * sizeof(uint8_t));

    // list_sizes
    idx.list_sizes.resize(idx.K);
    read_exact(f, idx.list_sizes.data(), size_t(idx.K) * sizeof(uint32_t));

    // postings
    uint64_t total = 0;
    idx.list_offsets.resize(size_t(idx.K) + 1);
    idx.list_offsets[0] = 0;
    for (uint32_t i = 0; i < idx.K; i++) {
        total += idx.list_sizes[i];
        idx.list_offsets[i + 1] = total;
    }

    idx.postings.resize((size_t)total);
    if (total) read_exact(f, idx.postings.data(), (size_t)total * sizeof(uint32_t));

    idx.build_weights();
    return idx;
}

// frame record after header:
// u32 frame_id, u32 Q, u8 qrx, u8 qry, u16 rsv, then Q*64 bytes queries
static bool read_one_frame_queries(std::ifstream& f, FrameQueries& out) {
    if (!f) return false;
    int c = f.peek();
    if (c == EOF) return false;

    uint32_t frame_id = 0, Q = 0;
    uint8_t qrx = 0, qry = 0;
    uint16_t rsv = 0;

    read_exact(f, &frame_id, 4);
    read_exact(f, &Q, 4);
    read_exact(f, &qrx, 1);
    read_exact(f, &qry, 1);
    read_exact(f, &rsv, 2);

    out.frame_id = frame_id;
    out.Q = Q;
    out.qrx = qrx;
    out.qry = qry;

    out.qdesc.resize(size_t(Q) * 64);
    if (Q) read_exact(f, out.qdesc.data(), out.qdesc.size());
    return true;
}

// ROI kernel: centroid -> nprobe lists -> fetch postings -> geo -> score -> top-k
uint64_t matcher_ivf_roi(const IVFIndex& idx, const FrameQueries& fr) {
    const int qx = (int)fr.qrx;
    const int qy = (int)fr.qry;
    const uint32_t topk = idx.top_k;

    uint64_t checksum = 0;

    // temp buffers
    std::vector<std::pair<float, uint32_t>> d2_lid;
    d2_lid.reserve(idx.K);

    for (uint32_t qi = 0; qi < fr.Q; qi++) {
        const uint8_t* q = &fr.qdesc[size_t(qi) * 64];

        // ---- build q_aug (128 floats) ----
        float q_aug[128];
        for (int d = 0; d < 64; d++) {
            const float qv = float(q[d]);               // uint8 -> float
            const float z  = (q[d] != 0) ? 1.0f : 0.0f; // presence
            q_aug[d]      = idx.wv[d] * qv;             // alpha*idf*val_scale * qv
            q_aug[64 + d] = idx.wz[d] * z;              // beta*idf * z
        }

        // ---- centroid distances ----
        d2_lid.clear();
        for (uint32_t lid = 0; lid < idx.K; lid++) {
            const float* c = &idx.centroids[size_t(lid) * 128];
            float d2 = 0.0f;
            for (int j = 0; j < 128; j++) {
                float diff = c[j] - q_aug[j];
                d2 += diff * diff;
            }
            d2_lid.push_back({d2, lid});
        }

        const uint32_t take_lists = std::min<uint32_t>(idx.nprobe, idx.K);
        if (take_lists == 0) continue;

        std::nth_element(
            d2_lid.begin(), d2_lid.begin() + take_lists, d2_lid.end(),
            [](const auto& a, const auto& b){ return a.first < b.first; }
        );
        d2_lid.resize(take_lists);
        std::sort(
            d2_lid.begin(), d2_lid.end(),
            [](const auto& a, const auto& b){ return a.first < b.first; }
        );

        // ---- scan postings from selected lists, apply geo + score, keep top-k ----
        struct Best { uint16_t s; uint32_t gid; };
        Best best[16]; // top_k is usually small, so reserve 16 slots
        uint32_t nb = 0;

        auto push_best = [&](uint16_t s, uint32_t gid) {
            // insert into best[] sorted desc by score, tie break smaller gid
            uint32_t pos = nb;
            if (nb < topk) {
                best[nb++] = {s, gid};
            } else {
                // if worse than current worst, skip
                Best worst = best[nb - 1];
                if (s < worst.s || (s == worst.s && gid >= worst.gid)) return;
                best[nb - 1] = {s, gid};
                pos = nb - 1;
            }
            // bubble up
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

        for (uint32_t i = 0; i < take_lists; i++) {
            const uint32_t lid = d2_lid[i].second;
            const uint64_t start = idx.list_offsets[lid];
            const uint64_t end   = idx.list_offsets[lid + 1];

            for (uint64_t p = start; p < end; p++) {
                const uint32_t gid = idx.postings[(size_t)p];
                if (gid >= idx.Nmap) continue; // safety

                // geo filter using map_xy_u8
                const uint8_t mx_u8 = idx.map_xy[size_t(gid) * 2 + 0];
                const uint8_t my_u8 = idx.map_xy[size_t(gid) * 2 + 1];
                const int mx = (int)mx_u8;
                const int my = (int)my_u8;

                if (mx < qx - 2 || mx > qx + 2 || my < qy - 2 || my > qy + 2) continue;

                // score with map_desc_u8
                const uint8_t* cdesc = &idx.map_desc[size_t(gid) * 64];
                const uint16_t s = score_equal_nonzero_64(q, cdesc);
                push_best(s, gid);
            }
        }

        // checksum: xor selected gids + score
        for (uint32_t t = 0; t < nb; t++) {
            checksum ^= (uint64_t(best[t].gid) << (t % 16));
            checksum ^= (uint64_t(best[t].s)   << ((t + 7) % 16));
        }
    }

    return checksum;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ./pf_kernel_ivf_roi <pfivf_roi.bin>\n";
        return 1;
    }

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) throw std::runtime_error("cannot open bin");

    IVFIndex idx = read_ivf_header(f);

    FrameQueries fr;
    size_t cnt = 0;
    uint64_t total_cs = 0;

    while (read_one_frame_queries(f, fr)) {
        m5_reset_stats(0, 0);
        m5_work_begin(1, (uint64_t)fr.frame_id);

        total_cs ^= matcher_ivf_roi(idx, fr);

        m5_work_end(1, (uint64_t)fr.frame_id);
        m5_dump_stats(0, 0);

        cnt++;
        if (cnt % 50 == 0) std::cerr << "processed frames: " << cnt << "\n";
    }

    std::cerr << "done. frames=" << cnt << " checksum=" << total_cs << "\n";
    return 0;
}