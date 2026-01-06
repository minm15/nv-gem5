#include "cim_api.hpp"
#include <gem5/m5ops.h>

#include "msim_config.hpp"
#include "msim_bins.hpp"
#include "layout.hpp"
#include "nvm_map_writer.hpp"
#include "noivf_matcher.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <algorithm>

static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x18000000;
static constexpr uintptr_t CMD_BASE  = 0x20000000;

int main(int argc, char** argv)
{
    std::cout << "[info] Begin main" << "\n";
    uint32_t max_steps = 100;
    uint32_t max_desc_per_step = 50;
    uint32_t max_groups = 0;

    if (argc >= 2) max_steps = static_cast<uint32_t>(std::stoul(argv[1]));
    if (argc >= 3) max_desc_per_step = static_cast<uint32_t>(std::stoul(argv[2]));
    if (argc >= 4) max_groups = static_cast<uint32_t>(std::stoul(argv[3]));

    auto* rw  = reinterpret_cast<volatile uint64_t*>(DATA_BASE);
    auto* tmp = reinterpret_cast<volatile uint64_t*>(TEMP_BASE);
    auto* cmd = reinterpret_cast<volatile uint64_t*>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);
    std::cout << "[info] Set geometry" << "\n";
    cim.setGeometry(msim::kBankBits, msim::kMatBits, msim::kArrayBits, msim::kRowBits, msim::kColBits);

    std::cout << "[info] Loading map bins from: " << msim::kMapDir << "\n";
    const msim::MapBins map = msim::load_map_bins(msim::kMapDir);
    std::cout << "[info] map N=" << map.N << "\n";

    std::cout << "[info] Loading query bins from: " << msim::kQueryDir << "\n";
    const msim::QueryBins query = msim::load_query_bins(msim::kQueryDir);
    std::cout << "[info] query steps=" << query.steps << " total_rows=" << query.total_rows << "\n";

    const msim::Placement place = msim::compute_placement(map.N);
    std::cout << "[info] banks: total=" << place.total_banks
              << " desc_bank0=" << place.desc_bank0 << " desc_bank_count=" << place.desc_bank_count
              << " geo_bank0=" << place.geo_bank0 << " geo_bank_count=" << place.geo_bank_count
              << "\n";

    std::cout << "[info] Writing map to CIM...\n";
    msim::NvmMapWriter writer(cim, place);
    writer.write_all(map);
    std::cout << "[info] Done writing map.\n";

    msim::NoIvfMatcher matcher(cim, place, map.geo_max);

    const uint32_t steps = std::min<uint32_t>(query.steps, max_steps);
    for (uint32_t s = 0; s < steps; ++s) {
        const uint8_t qgx = query.step_pose[static_cast<size_t>(s) * 2 + 0];
        const uint8_t qgy = query.step_pose[static_cast<size_t>(s) * 2 + 1];

        const uint32_t beg = query.offsets[s];
        const uint32_t end = query.offsets[s + 1];
        const uint32_t cnt = end - beg;

        const uint32_t run_cnt = std::min<uint32_t>(cnt, max_desc_per_step);
        std::cout << "[step] s=" << s << " qgx=" << static_cast<int>(qgx) << " qgy=" << static_cast<int>(qgy)
                  << " desc_cnt=" << cnt << " run=" << run_cnt << "\n";

        // Before running all descriptors in this frame
        m5_reset_stats(0, 0);     // reset stats at current tick
        m5_work_begin(1, s);      // optional: mark ROI begin (workid=1, work=step index)

        // (optional) avoid printing inside ROI if you want cleaner timing
        for (uint32_t i = 0; i < run_cnt; ++i) {
            const uint32_t qi = beg + i;
            const uint8_t* qdesc = &query.desc_q4[static_cast<size_t>(qi) * 64];

            const msim::MatchResult r = matcher.match_one_query_desc(qgx, qgy, qdesc, map.N, max_groups);

            // If you care about pure matching time, move prints outside ROI.
            // std::cout << ...;
        }

        m5_work_end(1, s);        // optional: mark ROI end
        m5_dump_stats(0, 0);      // dump stats for this frame
    }

    return 0;
}