#include "cim_api.hpp"
#include <gem5/m5ops.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// ====== You mapped these in python ======
static constexpr uintptr_t DATA_BASE = 0x10000000; // readWriteAddress
static constexpr uintptr_t TEMP_BASE = 0x14000000; // resultTemporaryBufferAddress
static constexpr uintptr_t CMD_BASE  = 0x15000000; // commandWriteAddress

// ====== Geometry ======
static constexpr int BANK_BITS   = 5;  // 32 banks
static constexpr int COL_BITS    = 6;  // 2^6 bytes per row = 64B = 512 bits
static constexpr int ROW_BYTES   = 1 << COL_BITS; // 64B

static constexpr int NUM_MAP     = 512; // bitvector width
static constexpr int NUM_QUERY   = 4;
static constexpr int VALUE_BITS  = 4;

static constexpr int DIMS           = 64;
static constexpr int DIMS_PER_BANK  = 32;
static constexpr int NUM_BANKS_USED = 2;

// 32 dim * 8 rows = 256 rows
static constexpr int MAP_ROWS_PER_BANK = DIMS_PER_BANK * 8; // 256

static inline void set_bit(uint8_t *row64B, int idx, bool v) {
    int byte = idx >> 3;
    int bit  = idx & 7; // LSB-first: map0->bit0
    uint8_t m = uint8_t(1u << bit);
    if (v) row64B[byte] |= m;
    else   row64B[byte] &= uint8_t(~m);
}

static inline bool get_bit(const uint8_t *row64B, int idx) {
    int byte = idx >> 3;
    int bit  = idx & 7;
    return (row64B[byte] >> bit) & 1u;
}

// bank-local row mapping
static inline uint8_t orig_row_local(int local_d, int b) { return uint8_t(local_d * 8 + b); }
static inline uint8_t inv_row_local (int local_d, int b) { return uint8_t(local_d * 8 + (b + 4)); }

// dim -> bank/local_d
static inline uint8_t dim_bank(int d)  { return uint8_t(d / DIMS_PER_BANK); } // 0 or 1
static inline int     dim_local(int d) { return d % DIMS_PER_BANK; }         // 0..31

// golden: mismatch mask across DIMS (LSB=d0)
static inline uint64_t golden_mask(const std::array<uint8_t, DIMS> &map,
                                  const std::array<uint8_t, DIMS> &q)
{
    uint64_t m = 0;
    for (int d = 0; d < DIMS; ++d) {
        if ((map[d] & 0xF) != (q[d] & 0xF)) m |= (1ull << d);
    }
    return m;
}

int main() {
    static_assert(ROW_BYTES == 64, "expect 64B row");
    static_assert(NUM_MAP == 512, "expect 512 maps");
    static_assert(DIMS == 64, "expect 64 dims");
    static_assert(DIMS_PER_BANK * 8 == 256, "32 dims per bank must fill 256 rows");
    static_assert(NUM_BANKS_USED == 2, "using 2 banks (bank0 + bank1)");
    static_assert(MAP_ROWS_PER_BANK == 256, "expect 256 rows used per bank");

    auto *rw  = reinterpret_cast<volatile uint64_t *>(DATA_BASE);
    auto *tmp = reinterpret_cast<volatile uint64_t *>(TEMP_BASE);
    auto *cmd = reinterpret_cast<volatile uint64_t *>(CMD_BASE);

    // 依照你目前 cim_api.hpp：CimModule(read_write, temp, command)
    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, COL_BITS);

    // ============ (ROI 外) 產生資料：maps + queries ============
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 15);

    std::array<std::array<uint8_t, DIMS>, NUM_MAP> maps{};
    std::array<std::array<uint8_t, DIMS>, NUM_QUERY> queries{};

    for (int i = 0; i < NUM_MAP; ++i)
        for (int d = 0; d < DIMS; ++d)
            maps[i][d] = uint8_t(dist(rng));

    for (int q = 0; q < NUM_QUERY; ++q)
        for (int d = 0; d < DIMS; ++d)
            queries[q][d] = uint8_t(dist(rng));

    // ============ (ROI 外) 寫入 maps 到 bank0/bank1 ============
    for (int d = 0; d < DIMS; ++d) {
        uint8_t bank = dim_bank(d);   // 0 or 1
        int ld       = dim_local(d);  // 0..31

        for (int b = 0; b < VALUE_BITS; ++b) {
            uint8_t row_o[ROW_BYTES]; std::memset(row_o, 0, ROW_BYTES);
            uint8_t row_i[ROW_BYTES]; std::memset(row_i, 0, ROW_BYTES);

            for (int m = 0; m < NUM_MAP; ++m) {
                bool bit = (maps[m][d] >> b) & 1u;
                set_bit(row_o, m, bit);
                set_bit(row_i, m, !bit);
            }

            cim.copy_to_cim(bank, orig_row_local(ld, b), row_o, ROW_BYTES);
            cim.copy_to_cim(bank, inv_row_local (ld, b), row_i, ROW_BYTES);
        }
    }

    // ============ ROI：只量你要的內容 ============
    // ROI 期間：OR(cmd MMIO) + temp load + reconstruct+popcount + cim internal latency
    // ROI 外：golden compare + print
    alignas(64) std::array<std::array<uint64_t, NUM_MAP>, NUM_QUERY> masks_out{};
    alignas(64) std::array<std::array<uint8_t,  NUM_MAP>, NUM_QUERY> scores_out{};

    // Reset stats so "measured" stats start here
    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);

    for (int qi = 0; qi < NUM_QUERY; ++qi) {

        // 1) OR commands (CPU -> cmd MMIO), results go to TEMP[bank][dest=ld]
        for (int d = 0; d < DIMS; ++d) {
            uint8_t bank = dim_bank(d);
            int ld       = dim_local(d);

            std::vector<uint8_t> rows;
            rows.reserve(4);
            for (int b = 0; b < VALUE_BITS; ++b) {
                int qbit = (queries[qi][d] >> b) & 1u;
                rows.push_back(qbit ? inv_row_local(ld, b) : orig_row_local(ld, b));
            }

            cim.OR(rows,
                   /*byte_mask=*/0xff,
                   /*bank_mask=*/CimModule::Mask::bank(bank),
                   /*column_mask=*/CimModule::Mask::colsAll(),
                   /*dest=*/uint8_t(ld)); // TEMP row = ld (0..31) within that bank
        }

        // 2) temp read back (CPU load temp)
        std::array<std::array<uint8_t, ROW_BYTES>, DIMS> tempRows{};
        for (int d = 0; d < DIMS; ++d) {
            uint8_t bank = dim_bank(d);
            int ld       = dim_local(d);
            cim.copy_temp_to_cpu(tempRows[d].data(), bank, uint16_t(ld), ROW_BYTES);
        }

        // 3) reconstruct mask + popcount
        for (int m = 0; m < NUM_MAP; ++m) {
            uint64_t mask = 0;
            for (int d = 0; d < DIMS; ++d) {
                if (get_bit(tempRows[d].data(), m))
                    mask |= (1ull << d);
            }
            masks_out[qi][m]  = mask;
            scores_out[qi][m] = static_cast<uint8_t>(__builtin_popcountll(mask));
        }
    }

    m5_work_end(0, 0);
    m5_dump_stats(0, 0);
    // ============ ROI END ============

    // ============ (ROI 外) correctness check + print ============
    bool all_ok = true;
    for (int qi = 0; qi < NUM_QUERY; ++qi) {
        for (int m = 0; m < NUM_MAP; ++m) {
            uint64_t g = golden_mask(maps[m], queries[qi]);
            if (masks_out[qi][m] != g) {
                all_ok = false;
                std::cout << "[FAIL] q=" << qi << " map=" << m
                          << " mask=0x"   << std::hex << masks_out[qi][m]
                          << " golden=0x" << g << std::dec
                          << " pop="  << int(scores_out[qi][m])
                          << " gpop=" << __builtin_popcountll(g)
                          << "\n";
                break;
            }
        }
        std::cout << "[Query " << qi << "] " << (all_ok ? "PASS" : "FAIL") << "\n";
        if (!all_ok) break;
    }

    std::cout << (all_ok ? "\nALL PASS ✅\n" : "\nTEST FAIL ❌\n");
    return all_ok ? 0 : 1;
}