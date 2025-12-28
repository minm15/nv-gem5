#include "cim_api.hpp"
#include <gem5/m5ops.h>

#include "nvm_writer.hpp"
#include "roi_match.hpp"
#include "workload_gen.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x14000000;
static constexpr uintptr_t CMD_BASE  = 0x18000000;

static inline uint64_t ts_now() { return m5_rpns(); }

struct ScopeTimer {
    const char* label;
    uint64_t t0;
    explicit ScopeTimer(const char* l) : label(l), t0(ts_now()) {}
    ~ScopeTimer()
    {
        const uint64_t t1 = ts_now();
        std::cout << "[MEASURE] " << label << " delta=" << (t1 - t0)
                  << " (m5_rpns units)\n";
    }
};

// --- Descriptor row mapping: 64 dims, 4-bit values; we store orig(4) + inv(4) rows per dim.
static inline uint16_t desc_orig_row(int d, int b)
{
    return static_cast<uint16_t>(d * (2 * lab::kValueBits) + b);
}
static inline uint16_t desc_inv_row(int d, int b)
{
    return static_cast<uint16_t>(d * (2 * lab::kValueBits) + lab::kValueBits + b);
}

static inline uint8_t get_lane_bit(const uint8_t* row_bytes, int lane)
{
    const int byte_i = lane >> 3;
    const int bit_i  = lane & 7;
    return static_cast<uint8_t>((row_bytes[byte_i] >> bit_i) & 1u);
}

static inline uint32_t golden_mismatch_one(const lab::Workload& wl, size_t qi, size_t map_i)
{
    uint32_t mis = 0;
    for (int d = 0; d < lab::kDims; ++d) {
        const uint8_t mv = wl.map_desc[map_i][static_cast<size_t>(d)] & 0x0Fu;
        const uint8_t qv = wl.query_desc[qi][static_cast<size_t>(d)] & 0x0Fu;
        mis += (mv != qv);
    }
    return mis;
}

static inline lab::RoiPerQuery golden_match_one(const lab::Workload& wl, size_t qi)
{
    lab::RoiPerQuery best;
    best.best_global = -1;
    best.best_mismatch = 0xFFFFFFFFu;

    for (size_t i = 0; i < wl.M; ++i) {
        const uint32_t mis = golden_mismatch_one(wl, qi, i);
        if (mis < best.best_mismatch ||
            (mis == best.best_mismatch && static_cast<int>(i) < best.best_global)) {
            best.best_global = static_cast<int>(i);
            best.best_mismatch = mis;
        }
    }
    return best;
}

// For a single query, recompute the ROI/CIM mismatch for every map,
// and print: global_id roi_mis golden_mis delta
static void print_roi_vs_golden_scores_for_query(
    CimModule& cim,
    const lab::Workload& wl,
    const lab::Placement& place,
    size_t qi)
{
    // Mapping assumptions (must match your writer/group_to_desc_bank_array):
    // - 4 descriptor banks (bank_desc0..3)
    // - mats per bank = 2^kMatBits, arrays per mat = 2^kArrayBits
    // - each (bank,mat,array) stores one group of 512 lanes (global maps)
    const uint32_t num_mats   = (1u << lab::NvmWriter::kMatBits);
    const uint32_t num_arrays = (1u << lab::NvmWriter::kArrayBits);
    const uint32_t groups_per_desc_bank = num_mats * num_arrays;

    const uint16_t desc_banks[4] = {
        static_cast<uint16_t>(place.bank_desc0),
        static_cast<uint16_t>(place.bank_desc1),
        static_cast<uint16_t>(place.bank_desc2),
        static_cast<uint16_t>(place.bank_desc3),
    };

    const size_t lanes_per_group = lab::NvmWriter::kLanesPerGroup; // 512
    const size_t num_groups = (wl.M + lanes_per_group - 1) / lanes_per_group;

    std::array<uint8_t, lab::NvmWriter::kRowBytes> tempRow{};
    std::vector<uint16_t> rows;
    rows.reserve(lab::kValueBits);

    std::cout << "[DUMP] qi=" << qi << " M=" << wl.M << "\n";
    std::cout << "global_id roi_mis golden_mis delta\n";

    for (size_t g = 0; g < num_groups; ++g) {
        const uint32_t g_u32 = static_cast<uint32_t>(g);
        const uint32_t bank_sel = g_u32 / groups_per_desc_bank; // 0..3
        const uint32_t within   = g_u32 % groups_per_desc_bank;
        const uint32_t mat      = within / num_arrays;
        const uint32_t array    = within % num_arrays;

        if (bank_sel >= 4) break; // M exceeds capacity of 4 desc banks

        const uint16_t bank = desc_banks[bank_sel];

        // Per-lane mismatch counters (0..64). We only need 0..64 so u8 is fine.
        std::array<uint8_t, 512> lane_mis{};
        lane_mis.fill(0);

        // Re-run the same per-dim OR + copy_temp_to_cpu, but keep per-lane counts.
        for (int d = 0; d < lab::kDims; ++d) {
            rows.clear();
            const uint8_t qv = wl.query_desc[qi][static_cast<size_t>(d)] & 0x0Fu;

            for (int b = 0; b < lab::kValueBits; ++b) {
                const int qbit = (qv >> b) & 1u;
                const uint16_t r = qbit ? desc_inv_row(d, b) : desc_orig_row(d, b);
                rows.push_back(r);
            }

            cim.OR(rows,
                   0xffu,
                   CimModule::Mask::bank(bank),
                   CimModule::Mask::colsAll(),
                   0,
                   CimModule::Mask::mat(static_cast<uint16_t>(mat)),
                   CimModule::Mask::array(static_cast<uint16_t>(array)));

            cim.copy_temp_to_cpu(tempRow.data(),
                                 bank,
                                 static_cast<uint16_t>(mat),
                                 static_cast<uint16_t>(array),
                                 0,
                                 lab::NvmWriter::kRowBytes);

            for (size_t lane = 0; lane < lanes_per_group; ++lane) {
                lane_mis[lane] = static_cast<uint8_t>(
                    lane_mis[lane] + get_lane_bit(tempRow.data(), static_cast<int>(lane)));
            }
        }

        const size_t base = g * lanes_per_group;
        for (size_t lane = 0; lane < lanes_per_group; ++lane) {
            const size_t global = base + lane;
            if (global >= wl.M) break;

            const uint32_t roi_mis  = lane_mis[lane];
            const uint32_t gold_mis = golden_mismatch_one(wl, qi, global);

            std::cout << global << ' '
                      << roi_mis << ' '
                      << gold_mis << ' '
                      << static_cast<int32_t>(roi_mis) - static_cast<int32_t>(gold_mis)
                      << "\n";
        }
    }
}

int main()
{
    std::cout << "in the main\n";

    auto* rw  = reinterpret_cast<volatile uint64_t*>(DATA_BASE);
    auto* tmp = reinterpret_cast<volatile uint64_t*>(TEMP_BASE);
    auto* cmd = reinterpret_cast<volatile uint64_t*>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);

    std::cout << "Setting geometry...\n";
    cim.setGeometry(lab::NvmWriter::kBankBits,
                    lab::NvmWriter::kMatBits,
                    lab::NvmWriter::kArrayBits,
                    lab::NvmWriter::kRowBits,
                    lab::NvmWriter::kColBits);

    const size_t M = 1u << 19;
    const size_t Q = 50;
    const uint32_t seed = 1;
    const uint8_t default_range = 30;

    std::cout << "Loading map/query bin file...\n";
    lab::Workload wl = lab::load_or_make_bin_workload(M, Q, seed, default_range);
    std::cout << "Finish loading map/query bin file...\n";

    lab::Placement place;
    place.bank_desc0 = 0;
    place.bank_desc1 = 1;
    place.bank_desc2 = 2;
    place.bank_desc3 = 3;
    place.bank_pos   = 4;
    place.mat        = 0;

    lab::NvmWriter writer(place);

    std::cout << "Writing NVM: desc banks 0..3, positions bank 4 ...\n";
    {
        ScopeTimer t("write_all (excluded from ROI)");
        writer.write_all(cim, wl);
    }

    std::cout << "\n[ROI] begin (geo gating from bank4, full scan)\n";
    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);

    lab::RoiResults rr;
    {
        ScopeTimer t("ROI total");
        rr = lab::run_roi_match(cim, wl, writer.placement());
    }

    m5_work_end(0, 0);
    m5_dump_stats(0, 0);
    std::cout << "[ROI] end\n\n";

    // Golden verification + print per-map scores for mismatched queries.
    // size_t ok = 0, fail = 0;
    // for (size_t qi = 0; qi < wl.Q; ++qi) {
    //     const auto& roi = rr.per_query[qi];
    //     const auto gold = golden_match_one(wl, qi);

    //     const bool match = (roi.best_global == gold.best_global) &&
    //                        (roi.best_mismatch == gold.best_mismatch);

    //     if (match) {
    //         ++ok;
    //     } else {
    //         ++fail;
    //         std::cout << "[MISMATCH] qi=" << qi
    //                   << " ROI(best_global=" << roi.best_global << ", mis=" << roi.best_mismatch << ")"
    //                   << " vs GOLDEN(best_global=" << gold.best_global << ", mis=" << gold.best_mismatch << ")\n";

    //         // 依你需求：把這個 mismatch 的 qi 對每個 map 的 mismatch 分數全部印出來
    //         print_roi_vs_golden_scores_for_query(cim, wl, writer.placement(), qi);
    //         std::cout << "[DUMP] end qi=" << qi << "\n\n";
    //     }
    // }

    // std::cout << "\n[CHECK] OK=" << ok << " FAIL=" << fail << " (Q=" << wl.Q << ")\n";

    // std::cout << "Results:\n";
    // for (size_t qi = 0; qi < wl.Q; ++qi) {
    //     const auto& r = rr.per_query[qi];
    //     const uint32_t best_match =
    //         (r.best_global >= 0 && r.best_mismatch != 0xFFFFFFFFu)
    //             ? static_cast<uint32_t>(lab::kDims - r.best_mismatch)
    //             : 0;
    //     std::cout << "  qi=" << qi
    //               << " best_global=" << r.best_global
    //               << " best_mismatch=" << r.best_mismatch
    //               << " best_match=" << best_match
    //               << "\n";
    // }

    std::cout << "DONE\n";
    return 0;
}