#include "cim_api.hpp"
#include <gem5/m5ops.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
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

static constexpr size_t ROW_BYTES = (1ull << COL_BITS); // 64B

static constexpr int NUM_MAP_LOCAL = 512;
static constexpr int DIMS       = 64;
static constexpr int VALUE_BITS = 4;

static inline uint16_t orig_row(int d, int b) { return static_cast<uint16_t>(d * 8 + b); }
static inline uint16_t inv_row (int d, int b) { return static_cast<uint16_t>(d * 8 + (b + 4)); }
static_assert(DIMS * 8 == (1 << ROW_BITS), "64 dims * 8 rows must fill 512 rows");

static constexpr uint16_t TEMP_BASE_ROW = 0;
static_assert(TEMP_BASE_ROW + DIMS <= (1u << ROW_BITS), "TEMP rows overflow");

static inline void set_bit(uint8_t *row64B, int map_id, bool v)
{
    const int byte = map_id >> 3;
    const int bit  = map_id & 7;
    const uint8_t m = static_cast<uint8_t>(1u << bit);
    if (v) row64B[byte] |= m;
    else   row64B[byte] &= static_cast<uint8_t>(~m);
}

// deterministic 4-bit map value generator
static inline uint8_t map_val(uint16_t bank, uint16_t mat, uint16_t array,
                              int local_map, int d)
{
    uint32_t x = 0x9E3779B9u;
    x ^= (static_cast<uint32_t>(bank)  * 0xA341316Cu);
    x ^= (static_cast<uint32_t>(mat)   * 0xC8013EA4u);
    x ^= (static_cast<uint32_t>(array) * 0xAD90777Du);
    x ^= (static_cast<uint32_t>(local_map) * 0x7E95761Eu);
    x ^= (static_cast<uint32_t>(d)     * 0xF39CC060u);

    x ^= (x >> 16);
    x *= 0x85EBCA6Bu;
    x ^= (x >> 13);
    x *= 0xC2B2AE35u;
    x ^= (x >> 16);

    return static_cast<uint8_t>(x & 0xFu);
}

static inline uint8_t golden_score(uint16_t bank, uint16_t mat, uint16_t array,
                                   int local_map,
                                   const std::array<uint8_t, DIMS> &q)
{
    uint32_t s = 0;
    for (int d = 0; d < DIMS; ++d) {
        const uint8_t mv = map_val(bank, mat, array, local_map, d);
        if ((mv & 0xFu) != (q[d] & 0xFu)) s++;
    }
    return static_cast<uint8_t>(s);
}

int main()
{
    auto *rw  = reinterpret_cast<volatile uint64_t *>(DATA_BASE);
    auto *tmp = reinterpret_cast<volatile uint64_t *>(TEMP_BASE);
    auto *cmd = reinterpret_cast<volatile uint64_t *>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, MAT_BITS, ARRAY_BITS, ROW_BITS, COL_BITS);

    // only test bank0 + mat0
    constexpr uint16_t TEST_BANK = 0;
    constexpr uint16_t TEST_MAT  = 0;
    const uint16_t arrays = static_cast<uint16_t>(1u << ARRAY_BITS);  // 16

    // ---- Build 4 queries (ROI OUT) ----
    std::array<std::array<uint8_t, DIMS>, 4> queries{};

    for (int d = 0; d < DIMS; ++d) {
        queries[0][d] = map_val(TEST_BANK, TEST_MAT, 0, 0, d);
    }
    queries[1] = queries[0];
    queries[1][0] = static_cast<uint8_t>((queries[1][0] + 1u) & 0xFu);

    std::mt19937 rng(20251216);
    std::uniform_int_distribution<int> dist(0, 15);
    for (int d = 0; d < DIMS; ++d) {
        queries[2][d] = static_cast<uint8_t>(dist(rng));
        queries[3][d] = static_cast<uint8_t>(dist(rng));
    }

    // ---- Fill RW for bank0/mat0 only (ROI OUT) ----
    alignas(64) std::array<uint8_t, ROW_BYTES> row_o{};
    alignas(64) std::array<uint8_t, ROW_BYTES> row_i{};

    std::cout << "Filling RW (bank0, mat0, all arrays) ..." << std::endl;

    for (uint16_t array = 0; array < arrays; ++array) {
        std::cout << "  fill array=" << array << "\n";

        for (int d = 0; d < DIMS; ++d) {
            for (int b = 0; b < VALUE_BITS; ++b) {
                row_o.fill(0);
                row_i.fill(0);

                for (int m = 0; m < NUM_MAP_LOCAL; ++m) {
                    const uint8_t v = map_val(TEST_BANK, TEST_MAT, array, m, d);
                    const bool bit = ((v >> b) & 1u) != 0u;
                    set_bit(row_o.data(), m, bit);
                    set_bit(row_i.data(), m, !bit);
                }

                cim.copy_to_cim(TEST_BANK, TEST_MAT, array, orig_row(d, b),
                                row_o.data(), ROW_BYTES);
                cim.copy_to_cim(TEST_BANK, TEST_MAT, array, inv_row(d, b),
                                row_i.data(), ROW_BYTES);
            }
        }
    }

    // ===== ROI: 只量測 (OR commands + temp readback + score reconstruction) =====
    // 儲存 ROI 產出的 scores，ROI 外再做 golden compare
    std::vector<uint8_t> all_scores;
    all_scores.resize(size_t(4) * size_t(arrays) * size_t(NUM_MAP_LOCAL), 0);

    alignas(64) std::array<std::array<uint8_t, ROW_BYTES>, DIMS> tempRows{};
    alignas(64) std::array<uint8_t, NUM_MAP_LOCAL> scores_cim{};

    // 重用 rows vector，避免每次都 malloc（雖然這段仍在 ROI 內，但至少穩定）
    std::vector<uint16_t> rows;
    rows.reserve(4);

    // ROI 開始：只算這段
    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);

    for (int qi = 0; qi < 4; ++qi) {
        for (uint16_t array = 0; array < arrays; ++array) {

            // 1) per-dim OR (4 rows) -> TEMP row = d
            for (int d = 0; d < DIMS; ++d) {
                rows.clear();
                for (int b = 0; b < VALUE_BITS; ++b) {
                    const int qbit = (queries[qi][d] >> b) & 1u;
                    rows.push_back(static_cast<uint16_t>(qbit ? inv_row(d, b)
                                                             : orig_row(d, b)));
                }

                cim.OR(rows,
                       /*byte_mask=*/0xffu,
                       /*bank_mask=*/CimModule::Mask::bank(TEST_BANK),
                       /*column_mask=*/CimModule::Mask::colsAll(),
                       /*dest=*/static_cast<uint16_t>(TEMP_BASE_ROW + d),
                       /*mat_mask=*/CimModule::Mask::mat(TEST_MAT),
                       /*array_mask=*/CimModule::Mask::array(array));
            }

            // 2) read back 64 temp rows
            for (int d = 0; d < DIMS; ++d) {
                cim.copy_temp_to_cpu(tempRows[d].data(),
                                     TEST_BANK, TEST_MAT, array,
                                     static_cast<uint16_t>(TEMP_BASE_ROW + d),
                                     ROW_BYTES);
            }

            // 3) reconstruct scores (per map = sum mismatch bits across dims)
            scores_cim.fill(0);
            for (int d = 0; d < DIMS; ++d) {
                for (int byte = 0; byte < static_cast<int>(ROW_BYTES); ++byte) {
                    uint8_t v = tempRows[d][byte];
                    if (!v) continue;
                    const int base = byte * 8;
                    for (int bit = 0; bit < 8; ++bit) {
                        if ((v >> bit) & 1u) {
                            scores_cim[base + bit] += 1;
                        }
                    }
                }
            }

            // ROI 內：只存結果，不做 golden compare
            const size_t base =
                (size_t(qi) * size_t(arrays) + size_t(array)) * size_t(NUM_MAP_LOCAL);
            std::memcpy(all_scores.data() + base, scores_cim.data(), size_t(NUM_MAP_LOCAL));
        }
    }

    // ROI 結束：只算到這裡
    m5_work_end(0, 0);
    m5_dump_stats(0, 0);

    // ===== ROI OUT: golden compare（排除時間）=====
    // std::cout << "Golden compare (excluded from ROI) ..." << std::endl;

    // for (int qi = 0; qi < 4; ++qi) {
    //     for (uint16_t array = 0; array < arrays; ++array) {

    //         const size_t base =
    //             (size_t(qi) * size_t(arrays) + size_t(array)) * size_t(NUM_MAP_LOCAL);

    //         for (int m = 0; m < NUM_MAP_LOCAL; ++m) {
    //             const uint8_t got = all_scores[base + size_t(m)];
    //             const uint8_t exp = golden_score(TEST_BANK, TEST_MAT, array, m, queries[qi]);
    //             if (got != exp) {
    //                 std::cout << "[FAIL]\n"
    //                           << "  qi=" << qi
    //                           << " bank=" << TEST_BANK
    //                           << " mat=" << TEST_MAT
    //                           << " array=" << array
    //                           << " local_map=" << m
    //                           << " got_score=" << int(got)
    //                           << " exp_score=" << int(exp)
    //                           << "\n";
    //                 return 1;
    //             }
    //         }
    //     }
    // }

    // const uint64_t expect_maps = uint64_t(arrays) * uint64_t(NUM_MAP_LOCAL); // 16*512=8192
    // std::cout << "\nALL PASS ✅\n"
    //           << "checked maps = " << (expect_maps * 4)
    //           << " comparisons\n";
    return 0;
}