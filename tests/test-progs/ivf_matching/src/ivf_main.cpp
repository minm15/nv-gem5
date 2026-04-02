// src/ivf_main.cpp
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

#include "gem5/m5ops.h"

#include "cim_api.hpp"
#include "map_loader.hpp"
#include "query_loader.hpp"

#include "ivf_layout.hpp"
#include "ivf_map_writer.hpp"
#include "ivf_matcher.hpp"

#include "msim_config.hpp"

static bool should_print_progress(uint64_t current, uint64_t total)
{
    if (total == 0) return false;
    if (total <= 10) return true;

    const uint64_t interval = (total + 9) / 10;
    return current == total || (current % interval) == 0;
}

static void usage(const char* prog)
{
    std::cerr << "usage: " << prog << " [--max-steps N]\n";
}

static std::vector<uint32_t> build_order_centroid_nn(const msim::IvfMapBins& map, uint32_t start = 0)
{
    const uint32_t K   = map.model.nlist;
    const uint32_t dim = map.model.dim;
    if (K == 0) return {};
    if (map.model.centroids.size() != static_cast<size_t>(K) * dim) {
        throw std::runtime_error("build_order_centroid_nn: invalid centroids");
    }
    start = (K ? (start % K) : 0);

    auto dist2 = [&](uint32_t a, uint32_t b) -> float {
        const float* ca = &map.model.centroids[static_cast<size_t>(a) * dim];
        const float* cb = &map.model.centroids[static_cast<size_t>(b) * dim];
        float s = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            const float d = ca[i] - cb[i];
            s += d * d;
        }
        return s;
    };

    std::vector<uint32_t> order;
    order.reserve(K);
    std::vector<uint8_t> used(K, 0);

    uint32_t cur = start;
    used[cur] = 1;
    order.push_back(cur);

    for (uint32_t t = 1; t < K; ++t) {
        uint32_t best = UINT32_MAX;
        float best_d  = 0.0f;
        for (uint32_t cand = 0; cand < K; ++cand) {
            if (used[cand]) continue;
            const float d = dist2(cur, cand);
            if (best == UINT32_MAX || d < best_d) {
                best = cand;
                best_d = d;
            }
        }
        if (best == UINT32_MAX) break;
        used[best] = 1;
        order.push_back(best);
        cur = best;
    }

    for (uint32_t i = 0; i < K; ++i) if (!used[i]) order.push_back(i);
    return order;
}

int main(int argc, char** argv)
{
    uint64_t requested_steps = 0;
    bool has_requested_steps = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--max-steps" && i + 1 < argc) {
            requested_steps = std::stoull(argv[++i]);
            has_requested_steps = true;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    // ---- hardcoded by msim_config ----
    const std::string map_dir   = msim::kMapDir;
    const std::string query_dir = msim::kQueryDir;

    // ---- fixed params ----
    static constexpr uint32_t kNProbe = 8;
    static constexpr uint32_t kMaxGroupsPerList = 50; // 0=all; cap for faster debug/ROI

    try {
        // ---- load map/query ----
        msim::MapLoader map_loader;
        map_loader.load_from_folder(map_dir);

        msim::QueryLoader query_loader;
        query_loader.load_from_folder(query_dir);

        std::cout << "[info] map_dir=" << map_dir << "\n";
        std::cout << "[info] query_dir=" << query_dir << "\n";

        std::cout << "[info] map N=" << map_loader.n()
                  << " desc_bits=" << static_cast<int>(map_loader.desc_bits())
                  << " ivf=" << (map_loader.ivf().enabled ? "on" : "off")
                  << " nlist=" << (map_loader.ivf().enabled ? map_loader.ivf().nlist : 0)
                  << "\n";

        const uint64_t available_steps = query_loader.steps();
        const uint64_t run_steps = has_requested_steps ?
            std::min<uint64_t>(requested_steps, available_steps) :
            available_steps;

        std::cout << "[info] query steps=" << available_steps
                  << " total_rows=" << query_loader.total_query_desc_rows()
                  << " run_steps=" << run_steps
                  << "\n";

        std::cout << "[info] nprobe=" << kNProbe
                  << " max_groups_per_list=" << kMaxGroupsPerList
                  << "\n";

        if (!map_loader.ivf().enabled) {
            throw std::runtime_error("ivf_main: map.ivf is not enabled in map.json");
        }

        // ---- build IvfMapBins from loaded content ----
        msim::IvfMapBins map;

        map.N = static_cast<uint32_t>(map_loader.n());
        map.geo_max = map_loader.geo().max;

        // Dense base arrays (indexed by global map_id)
        map.desc_q4  = map_loader.desc_q4_u8();
        map.geo_grid = map_loader.geo_grid_u8();

        // postings
        map.postings.nlist   = map_loader.ivf().nlist;
        map.postings.offsets = map_loader.ivf().postings_offsets_u32;
        map.postings.map_ids = map_loader.ivf().postings_map_ids_i32;

        // model
        map.model.nlist     = map_loader.ivf().nlist;
        map.model.dim       = map_loader.ivf().dim;
        map.model.centroids = map_loader.ivf().centroids_f32;

        map.model.alpha_value   = map_loader.ivf().alpha_value;
        map.model.beta_presence = map_loader.ivf().beta_presence;
        map.model.use_idf       = map_loader.ivf().use_idf ? 1 : 0;
        map.model.val_scale     = map_loader.ivf().val_scale_f32;
        map.model.idf_w         = map_loader.ivf().idf_w_f32;

        // ---- placement ----
        std::vector<uint32_t> order = build_order_centroid_nn(map, 0);
        const msim::IvfPlacement place = msim::compute_ivf_placement(map, order);
        // const msim::IvfPlacement place = msim::compute_ivf_placement(map);

        // ---- init CIM ----
        CimModule cim;

        // ---- write map to CIM (NOT in ROI) ----
        msim::IvfMapWriter writer(cim, place);
        writer.write_all(map);

        // ---- matcher ----
        msim::IvfMatcher matcher(cim, map, place);

        // ---- run (ROI covers match_one_query_desc, including CPU bucket selection inside it) ----
        uint64_t total_queries = 0;
        volatile uint64_t checksum = 0; // prevent over-optimization

        for (uint64_t s = 0; s < run_steps; ++s) {
            const msim::QueryStepView st = query_loader.step(s);

            // ROI for this step: all query desc in this step
            m5_reset_stats(0, 0);
            m5_work_begin(1, static_cast<uint64_t>(s));

            // for (size_t i = 0; i < st.m; ++i) {
            //     const uint8_t* qdesc = st.desc_ptr + i * 64u;

            //     const msim::MatchResult r =
            //         matcher.match_one_query_desc(st.gx, st.gy, qdesc, kNProbe, kMaxGroupsPerList);

            //     checksum += static_cast<uint64_t>(static_cast<uint32_t>(r.best_map_id)) +
            //                 static_cast<uint64_t>(r.best_mismatch);
            //     total_queries++;
            // }

            const auto results = matcher.match_one_step(st.gx, st.gy, st.desc_ptr, st.m, kNProbe, kMaxGroupsPerList);

            for (const auto& r : results) {
                checksum += static_cast<uint64_t>(static_cast<uint32_t>(r.best_map_id)) +
                            static_cast<uint64_t>(r.best_mismatch);
                total_queries++;
            }

            m5_work_end(1, static_cast<uint64_t>(s));
            m5_dump_stats(0, 0);

            const uint64_t completed_steps = s + 1;
            if (should_print_progress(completed_steps, run_steps)) {
                std::cerr << "processed steps: "
                          << completed_steps << "/" << run_steps << "\n";
            }
        }

        std::cout << "[done] run_steps=" << run_steps
                  << " total_queries=" << total_queries
                  << " checksum=" << checksum << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}
