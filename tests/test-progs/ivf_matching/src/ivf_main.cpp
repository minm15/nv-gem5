// src/ivf_main.cpp
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>

#include "gem5/m5ops.h"

#include "cim_api.hpp"
#include "map_loader.hpp"
#include "query_loader.hpp"

#include "ivf_layout.hpp"
#include "ivf_map_writer.hpp"
#include "ivf_matcher.hpp"

#include "msim_config.hpp"

int main()
{
    // ---- hardcoded by msim_config ----
    const std::string map_dir   = msim::kMapDir;
    const std::string query_dir = msim::kQueryDir;

    // ---- fixed params ----
    static constexpr uint32_t kNProbe = 32;
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

        std::cout << "[info] query steps=" << query_loader.steps()
                  << " total_rows=" << query_loader.total_query_desc_rows()
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
        const msim::IvfPlacement place = msim::compute_ivf_placement(map);

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

        for (size_t s = 0; s < query_loader.steps(); ++s) {
            const msim::QueryStepView st = query_loader.step(s);

            // ROI for this step: all query desc in this step
            m5_reset_stats(0, 0);
            m5_work_begin(1, static_cast<uint64_t>(s));

            for (size_t i = 0; i < st.m; ++i) {
                const uint8_t* qdesc = st.desc_ptr + i * 64u;

                const msim::MatchResult r =
                    matcher.match_one_query_desc(st.gx, st.gy, qdesc, kNProbe, kMaxGroupsPerList);

                checksum += static_cast<uint64_t>(static_cast<uint32_t>(r.best_map_id)) +
                            static_cast<uint64_t>(r.best_mismatch);
                total_queries++;
            }

            m5_work_end(1, static_cast<uint64_t>(s));
            m5_dump_stats(0, 0);

            std::cout << "[step] s=" << s << " m=" << st.m << "\n";
        }

        std::cout << "[done] total_queries=" << total_queries
                  << " checksum=" << checksum << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}