#include "cim_api.hpp"
#include <gem5/m5ops.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
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

static constexpr size_t ROW_BYTES = (1ull << COL_BITS); // 64B

static constexpr uint16_t TEST_BANK  = 0;
static constexpr uint16_t TEST_MAT   = 0;
static constexpr uint16_t TEST_ARRAY = 0;

static constexpr uint16_t ROW0 = 0;
static constexpr uint16_t ROW1 = 1;
static constexpr uint16_t ROW2 = 2;

static constexpr uint16_t TEMP_DEST_ROW = 0;

static void dump_hex_64B(const char* tag, const uint8_t* buf)
{
    std::cout << tag << " (64B):\n";
    for (int i = 0; i < 64; ++i) {
        if (i % 16 == 0) std::cout << "  ";
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << (unsigned)buf[i] << ((i % 16 == 15) ? "\n" : " ");
    }
    std::cout << std::dec;
}

static bool equal_64B(const uint8_t* a, const uint8_t* b)
{
    return std::memcmp(a, b, 64) == 0;
}

static void cpu_or_64B(uint8_t* out, const uint8_t* a, const uint8_t* b)
{
    for (int i = 0; i < 64; ++i) out[i] = (uint8_t)(a[i] | b[i]);
}

int main()
{
    auto *rw  = reinterpret_cast<volatile uint64_t *>(DATA_BASE);
    auto *tmp = reinterpret_cast<volatile uint64_t *>(TEMP_BASE);
    auto *cmd = reinterpret_cast<volatile uint64_t *>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, MAT_BITS, ARRAY_BITS, ROW_BITS, COL_BITS);

    // --- Prepare 3 rows of test data (64B each) ---
    alignas(64) std::array<uint8_t, ROW_BYTES> r0{};
    alignas(64) std::array<uint8_t, ROW_BYTES> r1{};
    alignas(64) std::array<uint8_t, ROW_BYTES> r2{};
    alignas(64) std::array<uint8_t, ROW_BYTES> out{};
    alignas(64) std::array<uint8_t, ROW_BYTES> exp{};

    // 讓 pattern 一眼可驗證（只在前幾個 byte 放不同 bit，其他 0）
    // r0: 0x01 0x02 0x04 0x08 ...
    // r1: 0x10 0x20 0x40 0x80 ...
    // r2: 0x55 0xAA 0x55 0xAA ...
    for (int i = 0; i < 16; ++i) {
        r0[i] = (uint8_t)(1u << (i % 8));
        r1[i] = (uint8_t)(0x10u << (i % 4)); // 0x10,0x20,0x40,0x80 repeat
        r2[i] = (uint8_t)((i % 2 == 0) ? 0x55u : 0xAAu);
    }

    std::cout << "Write 3 rows into bank0/mat0/array0: row0,row1,row2\n";
    dump_hex_64B("r0", r0.data());
    dump_hex_64B("r1", r1.data());
    dump_hex_64B("r2", r2.data());

    // --- Copy to CIM RW rows ---
    cim.copy_to_cim(TEST_BANK, TEST_MAT, TEST_ARRAY, ROW0, r0.data(), ROW_BYTES);
    cim.copy_to_cim(TEST_BANK, TEST_MAT, TEST_ARRAY, ROW1, r1.data(), ROW_BYTES);
    cim.copy_to_cim(TEST_BANK, TEST_MAT, TEST_ARRAY, ROW2, r2.data(), ROW_BYTES);

    auto do_or_and_read = [&](uint16_t a, uint16_t b, const uint8_t* ra, const uint8_t* rb) {
        std::vector<uint16_t> rows;
        rows.push_back(a);
        rows.push_back(b);

        // CIM OR -> TEMP_DEST_ROW
        cim.OR(rows,
               /*byte_mask=*/0xffu,
               /*bank_mask=*/CimModule::Mask::bank(TEST_BANK),
               /*column_mask=*/CimModule::Mask::colsAll(),
               /*dest=*/TEMP_DEST_ROW,
               /*mat_mask=*/CimModule::Mask::mat(TEST_MAT),
               /*array_mask=*/CimModule::Mask::array(TEST_ARRAY));

        // read back
        cim.copy_temp_to_cpu(out.data(),
                             TEST_BANK, TEST_MAT, TEST_ARRAY,
                             TEMP_DEST_ROW,
                             ROW_BYTES);

        // expected
        cpu_or_64B(exp.data(), ra, rb);

        std::cout << "\nOR(row" << a << ", row" << b << ") -> TEMP row " << TEMP_DEST_ROW << "\n";
        dump_hex_64B("out", out.data());
        dump_hex_64B("exp", exp.data());

        if (!equal_64B(out.data(), exp.data())) {
            std::cout << "[FAIL] OR(row" << a << ",row" << b << ") mismatch!\n";
            return false;
        }
        std::cout << "[PASS]\n";
        return true;
    };

    // --- Do 3 ORs (pairwise) ---
    bool ok = true;
    ok = ok && do_or_and_read(ROW0, ROW1, r0.data(), r1.data());
    ok = ok && do_or_and_read(ROW0, ROW2, r0.data(), r2.data());
    ok = ok && do_or_and_read(ROW1, ROW2, r1.data(), r2.data());

    if (!ok) return 1;

    std::cout << "\nALL PASS\n";
    return 0;
}