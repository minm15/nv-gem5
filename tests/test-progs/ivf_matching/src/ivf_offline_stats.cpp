// src/ivf_offline_stats.cpp
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "msim_config.hpp"
#include "map_loader.hpp"
#include "query_loader.hpp"

#include "ivf_bins.hpp"
#include "ivf_layout.hpp"

// ---------------------------
// helpers
// ---------------------------
static uint32_t parse_u32_or(const char* s, uint32_t defv)
{
    if (!s) return defv;
    char* endp = nullptr;
    unsigned long v = std::strtoul(s, &endp, 10);
    if (!endp || *endp != '\0') return defv;
    return static_cast<uint32_t>(v);
}

static std::vector<std::tuple<uint16_t,uint16_t,uint16_t>>
bucket_all_desc_regions(const msim::IvfPlacement& place, uint32_t bucket_id)
{
    std::set<std::tuple<uint16_t,uint16_t,uint16_t>> uniq;

    if (bucket_id >= place.bucket.size()) return {};
    const auto& bp = place.bucket[bucket_id];
    for (uint32_t g = 0; g < bp.desc_groups; ++g) {
        uint16_t b=0, m=0, a=0;
        msim::map_desc_bucket_group_to_region(place, bucket_id, g, b, m, a);
        uniq.insert({b,m,a});
    }

    return std::vector<std::tuple<uint16_t,uint16_t,uint16_t>>(uniq.begin(), uniq.end());
}

static void print_usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " [step_idx] [nprobe] [print_all]\n"
        << "  step_idx   : which query step to inspect (default: random)\n"
        << "  nprobe     : how many IVF lists to probe (default: 32)\n"
        << "  print_all  : 1 => print mapping for ALL lists, 0 => only probed + top heavy (default: 0)\n";
}

// ---------------------------
// select_lists_cpu (copied behavior)
// ---------------------------
static std::vector<uint32_t> select_lists_cpu_debug(const msim::IvfMapBins& map,
                                                    const uint8_t* qdesc64,
                                                    uint32_t nprobe)
{
    const uint32_t K = map.model.nlist;
    if (K == 0) return {};
    if (nprobe == 0 || nprobe > K) nprobe = K;

    const uint32_t dim = map.model.dim;
    if (dim == 0 || map.model.centroids.size() != static_cast<size_t>(K * dim)) {
        throw std::runtime_error("select_lists_cpu_debug: invalid centroids");
    }

    // ---- build query augmented vector ----
    std::vector<float> q(dim, 0.0f);

    const float alpha   = map.model.alpha_value;
    const float beta    = map.model.beta_presence;
    const bool  use_idf = (map.model.use_idf != 0);

    auto vs = [&](int d) -> float {
        return map.model.val_scale.empty() ? 1.0f : map.model.val_scale.at(static_cast<size_t>(d));
    };
    auto iw = [&](int d) -> float {
        if (!use_idf) return 1.0f;
        return map.model.idf_w.empty() ? 1.0f : map.model.idf_w.at(static_cast<size_t>(d));
    };

    // common case: aug128 = [value64, presence64]
    if (dim >= 128) {
        for (int d = 0; d < 64; ++d) {
            const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
            const float idf  = iw(d);
            const float val  = alpha * idf * vs(d) * static_cast<float>(qv);
            const float pres = beta  * idf * (qv != 0 ? 1.0f : 0.0f);
            q[static_cast<size_t>(d)]      = val;
            q[static_cast<size_t>(64 + d)] = pres;
        }
    } else {
        const uint32_t D = std::min<uint32_t>(dim, 64);
        for (uint32_t d = 0; d < D; ++d) q[d] = static_cast<float>(qdesc64[d] & 0x0Fu);
    }

    // ---- top-nprobe by L2 ----
    struct Pair { float dist; uint32_t id; };
    std::vector<Pair> best;
    best.reserve(nprobe);

    for (uint32_t k = 0; k < K; ++k) {
        const float* c = &map.model.centroids[static_cast<size_t>(k) * dim];
        float s = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            const float diff = q[i] - c[i];
            s += diff * diff;
        }

        if (best.size() < nprobe) {
            best.push_back({s, k});
            if (best.size() == nprobe) {
                std::nth_element(best.begin(), best.end() - 1, best.end(),
                                 [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
            }
        } else if (s < best.back().dist) {
            best.back() = {s, k};
            std::nth_element(best.begin(), best.end() - 1, best.end(),
                             [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
        }
    }

    std::sort(best.begin(), best.end(),
              [](const Pair& a, const Pair& b){ return a.dist < b.dist; });

    std::vector<uint32_t> out;
    out.reserve(best.size());
    for (const auto& p : best) out.push_back(p.id);
    return out;
}

static float centroid_dist_debug(const msim::IvfMapBins& map, const uint8_t* qdesc64, uint32_t kid)
{
    const uint32_t dim = map.model.dim;
    const float* c = &map.model.centroids[static_cast<size_t>(kid) * dim];

    std::vector<float> q(dim, 0.0f);

    const float alpha   = map.model.alpha_value;
    const float beta    = map.model.beta_presence;
    const bool  use_idf = (map.model.use_idf != 0);

    auto vs = [&](int d) -> float {
        return map.model.val_scale.empty() ? 1.0f : map.model.val_scale.at(static_cast<size_t>(d));
    };
    auto iw = [&](int d) -> float {
        if (!use_idf) return 1.0f;
        return map.model.idf_w.empty() ? 1.0f : map.model.idf_w.at(static_cast<size_t>(d));
    };

    if (dim >= 128) {
        for (int d = 0; d < 64; ++d) {
            const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
            const float idf  = iw(d);
            q[static_cast<size_t>(d)]      = alpha * idf * vs(d) * static_cast<float>(qv);
            q[static_cast<size_t>(64 + d)] = beta  * idf * (qv != 0 ? 1.0f : 0.0f);
        }
    } else {
        const uint32_t D = std::min<uint32_t>(dim, 64);
        for (uint32_t d = 0; d < D; ++d) q[d] = static_cast<float>(qdesc64[d] & 0x0Fu);
    }

    float s = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float diff = q[i] - c[i];
        s += diff * diff;
    }
    return s;
}

// ---------------------------
// mapping dump
// ---------------------------
static void dump_bucket_mapping(const msim::IvfPlacement& place,
                               uint32_t bucket_id,
                               bool verbose_groups)
{
    const auto& bp = place.bucket.at(bucket_id);

    // desc groups -> (bank,mat,array)
    std::set<std::tuple<uint16_t,uint16_t,uint16_t>> desc_regions;
    for (uint32_t g = 0; g < bp.desc_groups; ++g) {
        uint16_t b,m,a;
        msim::map_desc_bucket_group_to_region(place, bucket_id, g, b, m, a);
        desc_regions.insert({b,m,a});
    }

    // geo groups -> (bank,mat,array,row_base)
    std::set<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>> geo_regions;
    for (uint32_t g = 0; g < bp.geo_groups; ++g) {
        uint16_t b,m,a,row;
        msim::map_geo_bucket_group_to_region(place, bucket_id, g, b, m, a, row);
        geo_regions.insert({b,m,a,row});
    }

    std::cout << "  [bucket " << bucket_id << "] map_count=" << bp.map_count
              << " desc_groups=" << bp.desc_groups
              << " geo_groups="  << bp.geo_groups
              << "\n";

    std::cout << "    desc_regions(unique)=" << desc_regions.size() << " :";
    for (const auto& t : desc_regions) {
        auto [b,m,a] = t;
        std::cout << " (" << b << "," << m << "," << a << ")";
    }
    std::cout << "\n";

    std::cout << "    geo_regions(unique)=" << geo_regions.size() << " :";
    for (const auto& t : geo_regions) {
        auto [b,m,a,row] = t;
        std::cout << " (" << b << "," << m << "," << a << ",row=" << row << ")";
    }
    std::cout << "\n";

    if (verbose_groups) {
        std::cout << "    desc_group->region:";
        for (uint32_t g = 0; g < bp.desc_groups; ++g) {
            uint16_t b,m,a;
            msim::map_desc_bucket_group_to_region(place, bucket_id, g, b, m, a);
            std::cout << " g" << g << "=" << b << "/" << m << "/" << a;
        }
        std::cout << "\n";

        std::cout << "    geo_group->region:";
        for (uint32_t g = 0; g < bp.geo_groups; ++g) {
            uint16_t b,m,a,row;
            msim::map_geo_bucket_group_to_region(place, bucket_id, g, b, m, a, row);
            std::cout << " g" << g << "=" << b << "/" << m << "/" << a << "(row=" << row << ")";
        }
        std::cout << "\n";
    }
}

// ---------------------------
// main
// ---------------------------
int main(int argc, char** argv)
{
    uint32_t step_idx_arg = (argc >= 2) ? parse_u32_or(argv[1], UINT32_MAX) : UINT32_MAX;
    uint32_t nprobe       = (argc >= 3) ? parse_u32_or(argv[2], 32) : 32;
    uint32_t print_all    = (argc >= 4) ? parse_u32_or(argv[3], 0) : 0;

    if (argc >= 2 && std::string(argv[1]) == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    const std::string map_dir   = msim::kMapDir;
    const std::string query_dir = msim::kQueryDir;

    std::cout << "[map]   dir=" << map_dir << "\n";
    std::cout << "[query] dir=" << query_dir << "\n";
    std::cout << "[cfg] nprobe=" << nprobe << " print_all=" << print_all << "\n";

    // ---- load map/query ----
    msim::MapLoader map_loader;
    map_loader.load_from_folder(map_dir);

    msim::QueryLoader query_loader;
    query_loader.load_from_folder(query_dir);

    // ---- build IvfMapBins ----
    msim::IvfMapBins map;
    map.N        = static_cast<uint32_t>(map_loader.n());
    map.geo_max  = map_loader.geo().max;

    map.desc_q4  = map_loader.desc_q4_u8();
    map.geo_grid = map_loader.geo_grid_u8();

    if (!map_loader.ivf().enabled) {
        std::cerr << "[fatal] map.ivf is not enabled in map.json\n";
        return 1;
    }

    map.postings.nlist   = map_loader.ivf().nlist;
    map.postings.offsets = map_loader.ivf().postings_offsets_u32;
    map.postings.map_ids = map_loader.ivf().postings_map_ids_i32;

    map.model.nlist      = map_loader.ivf().nlist;
    map.model.dim        = map_loader.ivf().dim;
    map.model.centroids  = map_loader.ivf().centroids_f32;

    map.model.alpha_value   = map_loader.ivf().alpha_value;
    map.model.beta_presence = map_loader.ivf().beta_presence;
    map.model.use_idf       = map_loader.ivf().use_idf ? 1 : 0;
    map.model.val_scale     = map_loader.ivf().val_scale_f32;
    map.model.idf_w         = map_loader.ivf().idf_w_f32;

    std::cout << "[info] N=" << map.N
              << " K(nlist)=" << map.postings.nlist
              << " postings_total=" << map.postings.map_ids.size()
              << " dim=" << map.model.dim
              << " alpha=" << map.model.alpha_value
              << " beta=" << map.model.beta_presence
              << " use_idf=" << (map.model.use_idf ? 1 : 0)
              << "\n";

    std::cout << "[info] query steps=" << query_loader.steps()
              << " total_rows=" << query_loader.total_query_desc_rows()
              << "\n";

    // ---- placement ----
    const msim::IvfPlacement place = msim::compute_ivf_placement(map);
    std::cout << "[place] total_banks=" << place.total_banks
              << " desc: bank0=" << place.desc_bank0 << " bank_count=" << place.desc_bank_count
              << " geo: bank0="  << place.geo_bank0  << " bank_count="  << place.geo_bank_count
              << "\n";

    // ---- precompute: how many unique (bank,mat,array) each list touches (descriptor side) ----
    std::vector<uint32_t> arrays_per_list(place.bucket.size(), 0);
    for (uint32_t lid = 0; lid < place.bucket.size(); ++lid) {
        const auto& bp = place.bucket[lid];
        std::set<std::tuple<uint16_t,uint16_t,uint16_t>> regs;
        for (uint32_t g = 0; g < bp.desc_groups; ++g) {
            uint16_t b,m,a;
            msim::map_desc_bucket_group_to_region(place, lid, g, b, m, a);
            regs.insert({b,m,a});
        }
        arrays_per_list[lid] = static_cast<uint32_t>(regs.size());
    }

    // ---- loop over steps (frames) ----
    const size_t S = query_loader.steps();
    if (S == 0) {
        std::cerr << "[fatal] query has 0 steps\n";
        return 1;
    }

    // step_idx_arg == UINT32_MAX => ALL frames
    size_t s0 = 0, s1 = S;
    if (step_idx_arg != UINT32_MAX) {
        s0 = static_cast<size_t>(step_idx_arg % static_cast<uint32_t>(S));
        s1 = s0 + 1;
    }

    // fixed model assumptions (your stated counting model)
    constexpr uint64_t ROWS_PER_ARRAY = 512;
    constexpr uint64_t D_EQ = 64;
    constexpr uint64_t ORS_PER_ARRAY   = ROWS_PER_ARRAY * D_EQ; // 512 rows * 64 dims
    constexpr uint64_t TEMP_WRITES_PER_ARRAY = D_EQ;            // 1 mask per dim

    for (size_t step_idx = s0; step_idx < s1; ++step_idx) {
        const msim::QueryStepView st = query_loader.step(step_idx);

        uint64_t total_array_probes_in_frame = 0;

        for (size_t i = 0; i < st.m; ++i) {
            const uint8_t* qdesc = st.desc_ptr + i * 64u;
            std::vector<uint32_t> lists = select_lists_cpu_debug(map, qdesc, nprobe);

            for (uint32_t lid : lists) {
                const uint32_t na = (lid < arrays_per_list.size()) ? arrays_per_list[lid] : 0;
                total_array_probes_in_frame += static_cast<uint64_t>(na);
            }
        }

        const uint64_t or_count   = total_array_probes_in_frame * ORS_PER_ARRAY;
        const uint64_t temp_write = total_array_probes_in_frame * TEMP_WRITES_PER_ARRAY;

        std::cout << "[frame(step) " << step_idx << "]"
                  << " gx=" << static_cast<int>(st.gx)
                  << " gy=" << static_cast<int>(st.gy)
                  << " m(qdesc)=" << st.m
                  << " total_array_probes=" << total_array_probes_in_frame
                  << " OR=" << or_count
                  << " temp_write=" << temp_write
                  << "\n";
    }

    return 0;
}