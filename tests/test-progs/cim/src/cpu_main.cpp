#include <gem5/m5ops.h>

#include "cpu_match.hpp"
#include "workload_gen.hpp"

#include <cstdint>
#include <iostream>

static inline uint64_t ts_now()
{
    return m5_rpns();
}

struct ScopeTimer {
    const char* label;
    uint64_t t0;
    explicit ScopeTimer(const char* l) : label(l), t0(ts_now()) {}
    ~ScopeTimer() {
        const uint64_t t1 = ts_now();
        std::cout << "[MEASURE] " << label << " delta=" << (t1 - t0) << " (m5_rpns units)\n";
    }
};

int main()
{
    const int M = 1u << 19;
    const int Q = 50;
    const uint32_t seed = 1;
    const int range = 30;

    lab::Workload wl = lab::load_or_make_bin_workload(M, Q, seed, range);

    lab::CpuInvIndex idx;
    {
        ScopeTimer t("build_inv_index (excluded from ROI)");
        idx = lab::build_inv_index(wl);
    }

    std::cout << "\n[ROI] begin (CPU inverted index: argmax match, query!=0 only)\n";
    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);

    lab::CpuResults rr;
    {
        ScopeTimer t("ROI total");
        rr = lab::run_cpu_inverted_match(wl, idx);
    }

    m5_work_end(0, 0);
    m5_dump_stats(0, 0);
    std::cout << "[ROI] end\n\n";

    std::cout << "Results:\n";
    for (int qi = 0; qi < wl.Q; ++qi) {
        const auto& r = rr.per_query[static_cast<size_t>(qi)];
        std::cout << "  qi=" << qi
                  << " best_lane=" << r.best_lane
                  << " best_mismatch=" << r.best_mismatch
                  << " best_match=" << r.best_match
                  << " compared_dims=" << r.compared_dims
                  << "\n";
    }

    std::cout << "DONE\n";
    return 0;
}