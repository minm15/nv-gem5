// ivf_bank_check.cpp
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "msim_config.hpp"
#include "map_loader.hpp"

namespace {

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

static void print_percentiles(const std::vector<uint32_t>& v_sorted,
                              const std::string& name)
{
    if (v_sorted.empty()) return;

    auto pick = [&](double p) -> uint32_t {
        const double x = p * (static_cast<double>(v_sorted.size() - 1));
        const size_t i = static_cast<size_t>(x);
        return v_sorted[i];
    };

    std::cout << "[dist] " << name
              << " min=" << v_sorted.front()
              << " p50=" << pick(0.50)
              << " p90=" << pick(0.90)
              << " p99=" << pick(0.99)
              << " max=" << v_sorted.back()
              << "\n";
}

} // namespace

int main()
{
    // ---- mapping constants (match your ivf_layout.cpp assumptions) ----
    static constexpr uint32_t kTotalBanks       = 16;
    static constexpr uint32_t kMatsPerBank      = 16;
    static constexpr uint32_t kArraysPerMat     = 16;
    static constexpr uint32_t kArraysPerBank    = kMatsPerBank * kArraysPerMat; // 256

    static constexpr uint32_t kLanesPerGroup    = 512; // one desc-group = 512 lanes
    static constexpr uint32_t kGeoGroupsPerArray = 16; // one geo-array holds 16 geo-groups

    const std::string map_dir = msim::kMapDir;

    try {
        msim::MapLoader ml;
        ml.load_from_folder(map_dir);

        if (!ml.ivf().enabled) {
            throw std::runtime_error("map.json has no ivf section (ivf not enabled)");
        }

        const uint32_t K = ml.ivf().nlist;
        const auto& off = ml.ivf().postings_offsets_u32;

        if (off.size() != static_cast<size_t>(K) + 1u) {
            throw std::runtime_error("invalid postings_offsets size (expect K+1)");
        }

        // ---- per-bucket stats ----
        std::vector<uint32_t> map_counts;
        map_counts.reserve(K);

        std::vector<uint32_t> desc_groups;
        desc_groups.reserve(K);

        std::vector<uint32_t> geo_arrays_per_bucket;
        geo_arrays_per_bucket.reserve(K);

        uint64_t sum_maps = 0;
        uint64_t sum_desc_arrays = 0;
        uint64_t sum_geo_arrays  = 0;

        struct TopItem {
            uint32_t bucket;
            uint32_t map_count;
            uint32_t dgroups;
            uint32_t garrays;
        };
        std::vector<TopItem> top;
        top.reserve(K);

        for (uint32_t b = 0; b < K; ++b) {
            const uint32_t a = off[b];
            const uint32_t z = off[b + 1u];
            if (z < a) throw std::runtime_error("postings_offsets not monotonic");

            const uint32_t cnt = z - a;
            const uint32_t dg  = ceil_div_u32(cnt, kLanesPerGroup);
            const uint32_t ga  = ceil_div_u32(dg, kGeoGroupsPerArray);

            map_counts.push_back(cnt);
            desc_groups.push_back(dg);
            geo_arrays_per_bucket.push_back(ga);

            sum_maps       += cnt;
            sum_desc_arrays += dg; // 1 desc-group uses 1 desc-array
            sum_geo_arrays  += ga; // geo packed 16 groups per geo-array

            top.push_back({b, cnt, dg, ga});
        }

        // ---- bank estimation (same as compute_ivf_placement) ----
        const uint32_t desc_banks = ceil_div_u32(static_cast<uint32_t>(sum_desc_arrays), kArraysPerBank);
        const uint32_t geo_banks  = ceil_div_u32(static_cast<uint32_t>(sum_geo_arrays),  kArraysPerBank);
        const uint32_t total_banks_need = desc_banks + geo_banks;

        // ---- print summary ----
        std::cout << "[map] dir=" << map_dir << "\n";
        std::cout << "[map] N=" << ml.n()
                  << " nlist(K)=" << K
                  << " postings_total=" << sum_maps
                  << "\n";

        std::cout << "[pack] arrays_per_bank=" << kArraysPerBank
                  << " lanes_per_group=" << kLanesPerGroup
                  << " geo_groups_per_array=" << kGeoGroupsPerArray
                  << "\n";

        std::cout << "[need] total_desc_arrays=" << sum_desc_arrays
                  << " => desc_banks=" << desc_banks
                  << "\n";
        std::cout << "[need] total_geo_arrays=" << sum_geo_arrays
                  << " => geo_banks=" << geo_banks
                  << "\n";
        std::cout << "[need] total_banks=" << total_banks_need
                  << " (limit=" << kTotalBanks << ")"
                  << (total_banks_need > kTotalBanks ? "  **EXCEEDED**" : "  OK")
                  << "\n";

        // ---- distributions ----
        {
            std::vector<uint32_t> mc = map_counts;
            std::sort(mc.begin(), mc.end());
            print_percentiles(mc, "map_count_per_list");
        }
        {
            std::vector<uint32_t> dg = desc_groups;
            std::sort(dg.begin(), dg.end());
            print_percentiles(dg, "desc_groups_per_list");
        }
        {
            std::vector<uint32_t> ga = geo_arrays_per_bucket;
            std::sort(ga.begin(), ga.end());
            print_percentiles(ga, "geo_arrays_per_list");
        }

        // ---- show top 12 buckets by map_count ----
        std::sort(top.begin(), top.end(),
                  [](const TopItem& x, const TopItem& y) { return x.map_count > y.map_count; });

        const size_t show = std::min<size_t>(12, top.size());
        std::cout << "[top] largest buckets:\n";
        std::cout << "  bucket   map_count   desc_groups   geo_arrays\n";
        for (size_t i = 0; i < show; ++i) {
            const auto& t = top[i];
            std::cout << "  " << std::setw(6) << t.bucket
                      << "  " << std::setw(10) << t.map_count
                      << "  " << std::setw(11) << t.dgroups
                      << "  " << std::setw(10) << t.garrays
                      << "\n";
        }

        // ---- quick sanity note ----
        if (K >= 4096) {
            std::cout << "[note] if K=4096 and every list has >=1 desc_group, "
                         "desc_banks will be at least 16, leaving no bank for geo.\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}