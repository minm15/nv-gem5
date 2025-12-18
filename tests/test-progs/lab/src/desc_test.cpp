#include "cim_api.hpp"
#include <gem5/m5ops.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// ====== mapped in python ======
static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x14000000;
static constexpr uintptr_t CMD_BASE  = 0x18000000;

// ====== geometry (must match CimHandler params) ======
static constexpr unsigned BANK_BITS  = 3;  // 8 banks
static constexpr unsigned MAT_BITS   = 4;  // 16 mats
static constexpr unsigned ARRAY_BITS = 4;  // 16 arrays
static constexpr unsigned ROW_BITS   = 9;  // 512 rows
static constexpr unsigned COL_BITS   = 6;  // 64B/row

static constexpr size_t ROW_BYTES = (1ull << COL_BITS); // 64B (=512 bits)

// ====== workload ======
static constexpr int NUM_MAP_LOCAL = 512; // 512 maps => 512 lanes
static constexpr int NUM_QUERIES   = 4;   // 4 queries
static constexpr int DIMS          = 64;  // 64 dims
static constexpr int VALUE_BITS    = 4;   // 4-bit value per dim

// Layout in RW rows (512 rows total):
// per dim: [orig bit0..3] + [inv bit0..3] => 8 rows per dim
static inline uint16_t orig_row(int d, int b) {
    return static_cast<uint16_t>(d * (2 * VALUE_BITS) + b);
}
static inline uint16_t inv_row (int d, int b) {
    return static_cast<uint16_t>(d * (2 * VALUE_BITS) + VALUE_BITS + b);
}
static_assert(DIMS * (2 * VALUE_BITS) == (1 << ROW_BITS),
              "This config must exactly fill 512 RW rows (512x512 layout).");

// TEMP destination row (in TEMP window). overwrite per-dim and read back immediately.
static constexpr uint16_t TEMP_DEST_ROW = 0;

// ====== A: tick marker (sim tick visible in gem5 debug log) ======
static constexpr uintptr_t MARK_BASE = TEMP_BASE + 0x2000;
static inline void tick_mark(uint64_t tag)
{
    auto *p = reinterpret_cast<volatile uint64_t *>(MARK_BASE);
    p[0] = tag;
    asm volatile("" ::: "memory");
}

// ====== measurement helpers (host time) ======
static inline uint64_t ts_now() { return m5_rpns(); }

struct ScopeTimer {
    const char* label;
    uint64_t t0;
    explicit ScopeTimer(const char* l) : label(l), t0(ts_now()) {}
    ~ScopeTimer() {
        const uint64_t t1 = ts_now();
        std::cout << "[MEASURE] " << label << " delta=" << (t1 - t0)
                  << " (m5_rpns units)\n";
    }
};

static inline void set_bit(uint8_t *row64B, int map_id, bool v)
{
    const int byte = map_id >> 3;
    const int bit  = map_id & 7;
    const uint8_t m = static_cast<uint8_t>(1u << bit);
    if (v) row64B[byte] |= m;
    else   row64B[byte] &= static_cast<uint8_t>(~m);
}

// ====== deterministic data generator (4-bit values) ======
static inline uint8_t map_val(int m, int d)
{
    uint32_t x = static_cast<uint32_t>(m) * 1103515245u
               + static_cast<uint32_t>(d) * 12345u
               + 0x9e3779b9u;
    x ^= (x >> 16);
    return static_cast<uint8_t>(x & 0xFu);
}

static inline uint8_t query_val(int qi, int d)
{
    uint32_t x = static_cast<uint32_t>(qi) * 0x13579bdu
               + static_cast<uint32_t>(d) * 0x2468aceu
               + 0x7u;
    x ^= (x >> 13);
    return static_cast<uint8_t>(x & 0xFu);
}

// ====== CPU bit-sliced ripple-carry counter + argmin-on-planes ======
static constexpr int LANES_WORDS = 8; // 8 * 64 = 512

static inline void load_mask_u64(const uint8_t* mask64B, uint64_t out[LANES_WORDS])
{
    std::memcpy(out, mask64B, 64);
}

// K = ceil(log2(DIMS+1)) ; DIMS=64 => 7 bits (0..64)
static constexpr int COUNTER_BITS =
    (DIMS + 1 <= 2)   ? 1 :
    (DIMS + 1 <= 4)   ? 2 :
    (DIMS + 1 <= 8)   ? 3 :
    (DIMS + 1 <= 16)  ? 4 :
    (DIMS + 1 <= 32)  ? 5 :
    (DIMS + 1 <= 64)  ? 6 :
    (DIMS + 1 <= 128) ? 7 : 8;

using Planes = std::array<std::array<uint64_t, LANES_WORDS>, COUNTER_BITS>;

static inline void planes_zero(Planes& p)
{
    for (int k = 0; k < COUNTER_BITS; ++k)
        for (int w = 0; w < LANES_WORDS; ++w)
            p[k][w] = 0;
}

static inline void planes_add_mask(Planes& p, const uint64_t mask_words[LANES_WORDS])
{
    uint64_t carry[LANES_WORDS];
    for (int w = 0; w < LANES_WORDS; ++w) carry[w] = mask_words[w];

    for (int k = 0; k < COUNTER_BITS; ++k) {
        for (int w = 0; w < LANES_WORDS; ++w) {
            const uint64_t a = p[k][w];
            const uint64_t c = carry[w];
            p[k][w] = a ^ c;
            carry[w] = a & c;
        }
        uint64_t acc = 0;
        for (int w = 0; w < LANES_WORDS; ++w) acc |= carry[w];
        if (acc == 0) break;
    }
}

static inline bool words_any_nonzero(const uint64_t w[LANES_WORDS])
{
    uint64_t acc = 0;
    for (int i = 0; i < LANES_WORDS; ++i) acc |= w[i];
    return acc != 0;
}

static inline int ctz64(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while ((x & 1ull) == 0ull) { x >>= 1; ++n; }
    return n;
#endif
}

static inline int planes_argmin_lane(const Planes& p)
{
    uint64_t cand[LANES_WORDS];
    for (int w = 0; w < LANES_WORDS; ++w) cand[w] = ~0ull;

    for (int k = COUNTER_BITS - 1; k >= 0; --k) {
        uint64_t zeros[LANES_WORDS];
        for (int w = 0; w < LANES_WORDS; ++w) {
            zeros[w] = cand[w] & ~p[k][w];
        }
        if (words_any_nonzero(zeros)) {
            for (int w = 0; w < LANES_WORDS; ++w) cand[w] = zeros[w];
        } else {
            for (int w = 0; w < LANES_WORDS; ++w) cand[w] = cand[w] & p[k][w];
        }
    }

    for (int w = 0; w < LANES_WORDS; ++w) {
        uint64_t x = cand[w];
        if (x) return w * 64 + ctz64(x);
    }
    return -1;
}

static inline uint32_t planes_get_lane_value(const Planes& p, int lane)
{
    const int w = lane >> 6;
    const int b = lane & 63;
    uint32_t v = 0;
    for (int k = 0; k < COUNTER_BITS; ++k) {
        v |= static_cast<uint32_t>(((p[k][w] >> b) & 1ull) << k);
    }
    return v;
}

// ====== golden (CPU brute) ======
static inline uint32_t golden_mismatch_count(int m, const std::array<uint8_t, DIMS>& q)
{
    uint32_t mis = 0;
    for (int d = 0; d < DIMS; ++d) {
        const uint8_t mv = map_val(m, d) & 0xFu;
        const uint8_t qv = q[d] & 0xFu;
        if (mv != qv) ++mis;
    }
    return mis;
}

int main()
{
    auto *rw  = reinterpret_cast<volatile uint64_t *>(DATA_BASE);
    auto *tmp = reinterpret_cast<volatile uint64_t *>(TEMP_BASE);
    auto *cmd = reinterpret_cast<volatile uint64_t *>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, MAT_BITS, ARRAY_BITS, ROW_BITS, COL_BITS);

    constexpr uint16_t TEST_BANK  = 0;
    constexpr uint16_t TEST_MAT   = 0;
    constexpr uint16_t TEST_ARRAY = 0;

    // ---- Build queries ----
    std::array<std::array<uint8_t, DIMS>, NUM_QUERIES> queries{};
    for (int qi = 0; qi < NUM_QUERIES; ++qi) {
        for (int d = 0; d < DIMS; ++d) {
            queries[qi][d] = query_val(qi, d) & 0xFu;
        }
    }

    // ---- Fill RW rows ----
    alignas(64) std::array<uint8_t, ROW_BYTES> row_o{};
    alignas(64) std::array<uint8_t, ROW_BYTES> row_i{};

    std::cout << "Filling RW (bank0, mat0, array0, 512 rows total) ...\n";

    for (int d = 0; d < DIMS; ++d) {
        for (int b = 0; b < VALUE_BITS; ++b) {
            row_o.fill(0);
            row_i.fill(0);

            for (int m = 0; m < NUM_MAP_LOCAL; ++m) {
                const uint8_t v = map_val(m, d) & 0xFu;
                const bool bit = ((v >> b) & 1u) != 0u;
                set_bit(row_o.data(), m, bit);
                set_bit(row_i.data(), m, !bit);
            }

            cim.copy_to_cim(TEST_BANK, TEST_MAT, TEST_ARRAY, orig_row(d, b),
                            row_o.data(), ROW_BYTES);
            cim.copy_to_cim(TEST_BANK, TEST_MAT, TEST_ARRAY, inv_row(d, b),
                            row_i.data(), ROW_BYTES);
        }
    }

    // ============================================================
    // ROI: from queries -> 4*64*(build rows + cim.OR + read mask + ripple add) + argmin
    // ============================================================
    alignas(64) std::array<uint8_t, ROW_BYTES> tempRow{};
    std::vector<uint16_t> rows;
    rows.reserve(VALUE_BITS);

    volatile uint64_t sink = 0;
    std::array<int, NUM_QUERIES> best_lane{};
    std::array<uint32_t, NUM_QUERIES> best_mismatch{};

    std::cout << "\n[ROI] begin (query -> CIM OR/read -> ripple -> argmin)\n";
    tick_mark(0xC0000000ull);

    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);

    {
        ScopeTimer t_all("ROI total: 4*64*(rows+OR+read+ripple) + argmin");

        for (int qi = 0; qi < NUM_QUERIES; ++qi) {
            ScopeTimer t_q("  per-query: 64*(rows+OR+read+ripple) + argmin");

            Planes planes{};
            planes_zero(planes);

            for (int d = 0; d < DIMS; ++d) {
                // ---- build rows for this dim based on query bits ----
                rows.clear();
                const uint8_t qv = queries[qi][d] & 0xFu;
                for (int b = 0; b < VALUE_BITS; ++b) {
                    const int qbit = (qv >> b) & 1u;
                    rows.push_back(static_cast<uint16_t>(qbit ? inv_row(d, b)
                                                             : orig_row(d, b)));
                }

                // ---- CIM OR -> TEMP row ----
                cim.OR(rows,
                       /*byte_mask=*/0xffu,
                       /*bank_mask=*/CimModule::Mask::bank(TEST_BANK),
                       /*column_mask=*/CimModule::Mask::colsAll(),
                       /*dest=*/TEMP_DEST_ROW,
                       /*mat_mask=*/CimModule::Mask::mat(TEST_MAT),
                       /*array_mask=*/CimModule::Mask::array(TEST_ARRAY));

                // ---- read back mismatch mask ----
                cim.copy_temp_to_cpu(tempRow.data(),
                                     TEST_BANK, TEST_MAT, TEST_ARRAY,
                                     TEMP_DEST_ROW,
                                     ROW_BYTES);

                // ---- ripple add into bit-sliced counter ----
                uint64_t mask_words[LANES_WORDS];
                load_mask_u64(tempRow.data(), mask_words);
                planes_add_mask(planes, mask_words);
            }

            // ---- argmin on planes ----
            const int lane = planes_argmin_lane(planes);
            const uint32_t mis = planes_get_lane_value(planes, lane);

            best_lane[qi] = lane;
            best_mismatch[qi] = mis;

            sink += static_cast<uint64_t>(lane);
            sink += static_cast<uint64_t>(mis);

            std::cout << "[RESULT] qi=" << qi
                      << " best_lane=" << lane
                      << " best_mismatch=" << mis
                      << " best_match=" << (DIMS - mis)
                      << "\n";
        }
    }

    m5_work_end(0, 0);
    m5_dump_stats(0, 0);

    tick_mark(0xC0000001ull);
    std::cout << "[DBG] sink=" << (uint64_t)sink << "\n";
    std::cout << "[ROI] end\n\n";

    // ===== ROI OUT: golden check =====
    std::cout << "Golden compare (excluded from ROI) ...\n";
    for (int qi = 0; qi < NUM_QUERIES; ++qi) {
        uint32_t best_mis = 0xffffffffu;
        int best_m = -1;

        for (int m = 0; m < NUM_MAP_LOCAL; ++m) {
            const uint32_t mis = golden_mismatch_count(m, queries[qi]);
            if (mis < best_mis) {
                best_mis = mis;
                best_m = m;
            }
        }

        if (best_mismatch[qi] != best_mis || best_lane[qi] != best_m) {
            std::cout << "[FAIL]\n"
                      << "  qi=" << qi
                      << " got(best_lane=" << best_lane[qi]
                      << ", best_mismatch=" << best_mismatch[qi] << ")\n"
                      << "  exp(best_lane=" << best_m
                      << ", best_mismatch=" << best_mis << ")\n";
            return 1;
        }
    }

    std::cout << "ALL PASS (ROI measures only CPU ripple+argmin)\n";
    return 0;
}