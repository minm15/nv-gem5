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
static constexpr unsigned BANK_BITS  = 3;  // 16 banks
static constexpr unsigned MAT_BITS   = 4;  // 16 mats
static constexpr unsigned ARRAY_BITS = 4;  // 16 arrays
static constexpr unsigned ROW_BITS   = 9;  // 512 rows
static constexpr unsigned COL_BITS   = 6;  // 64B/row

static constexpr size_t ROW_BYTES = (1ull << COL_BITS);

// Test target
static constexpr uint16_t TEST_MAT   = 0;
static constexpr uint16_t TEST_ARRAY = 0;
static constexpr uint16_t ROW0 = 0;
static constexpr uint16_t ROW1 = 1;
static constexpr uint16_t TEMP_DEST_ROW = 0;

static constexpr size_t HEX_DUMP_BYTES = 32;

static bool equal_64B(const uint8_t* a, const uint8_t* b)
{
    return std::memcmp(a, b, ROW_BYTES) == 0;
}

static void cpu_or_64B(uint8_t* out, const uint8_t* a, const uint8_t* b)
{
    for (size_t i = 0; i < ROW_BYTES; ++i) out[i] = static_cast<uint8_t>(a[i] | b[i]);
}

static void dump_hex_prefix(const char* tag, const uint8_t* buf, size_t n)
{
    std::cout << tag << " (first " << n << "B):\n  ";
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<unsigned>(buf[i]) << ((i % 16 == 15) ? "\n  " : " ");
    }
    std::cout << std::dec << "\n";
}

static void fill_pattern(std::array<uint8_t, ROW_BYTES>& r0,
                         std::array<uint8_t, ROW_BYTES>& r1)
{
    r0.fill(0);
    r1.fill(0);

    // Easy-to-verify patterns.
    // r0: 01 02 04 08 ... (repeat)
    // r1: 10 20 40 80 ... (repeat)
    for (int i = 0; i < 32; ++i) {
        r0[static_cast<size_t>(i)] = static_cast<uint8_t>(1u << (i % 8));
        r1[static_cast<size_t>(i)] = static_cast<uint8_t>(0x10u << (i % 4));
    }
}

static bool test_one_bank(CimModule& cim, uint16_t bank)
{
    alignas(64) std::array<uint8_t, ROW_BYTES> r0{};
    alignas(64) std::array<uint8_t, ROW_BYTES> r1{};
    alignas(64) std::array<uint8_t, ROW_BYTES> exp{};
    alignas(64) std::array<uint8_t, ROW_BYTES> rb0{};
    alignas(64) std::array<uint8_t, ROW_BYTES> rb1{};
    alignas(64) std::array<uint8_t, ROW_BYTES> out{};

    fill_pattern(r0, r1);
    cpu_or_64B(exp.data(), r0.data(), r1.data());

    std::cout << "\n==== Bank " << bank << " test ====\n";
    dump_hex_prefix("r0(write)", r0.data(), HEX_DUMP_BYTES);
    dump_hex_prefix("r1(write)", r1.data(), HEX_DUMP_BYTES);
    dump_hex_prefix("exp_or", exp.data(), HEX_DUMP_BYTES);

    // 1) Write into RW
    cim.copy_to_cim(bank, TEST_MAT, TEST_ARRAY, ROW0, r0.data(), ROW_BYTES);
    cim.copy_to_cim(bank, TEST_MAT, TEST_ARRAY, ROW1, r1.data(), ROW_BYTES);

    // 2) Readback from RW (sanity)
    cim.copy_to_cpu(rb0.data(), bank, TEST_MAT, TEST_ARRAY, ROW0, ROW_BYTES);
    cim.copy_to_cpu(rb1.data(), bank, TEST_MAT, TEST_ARRAY, ROW1, ROW_BYTES);

    if (!equal_64B(r0.data(), rb0.data()) || !equal_64B(r1.data(), rb1.data())) {
        std::cout << "[FAIL] RW readback mismatch\n";
        dump_hex_prefix("rb0(read)", rb0.data(), HEX_DUMP_BYTES);
        dump_hex_prefix("rb1(read)", rb1.data(), HEX_DUMP_BYTES);
        return false;
    }
    std::cout << "[PASS] RW readback OK\n";

    // 3) CIM OR(row0,row1) -> TEMP_DEST_ROW
    std::vector<uint16_t> rows;
    rows.push_back(ROW0);
    rows.push_back(ROW1);

    cim.OR(rows,
           0xffu,
           CimModule::Mask::bank(bank),
           CimModule::Mask::colsAll(),
           TEMP_DEST_ROW,
           CimModule::Mask::mat(TEST_MAT),
           CimModule::Mask::array(TEST_ARRAY));

    // 4) Readback from TEMP
    cim.copy_temp_to_cpu(out.data(), bank, TEST_MAT, TEST_ARRAY, TEMP_DEST_ROW, ROW_BYTES);

    if (!equal_64B(out.data(), exp.data())) {
        std::cout << "[FAIL] TEMP OR result mismatch\n";
        dump_hex_prefix("out(temp)", out.data(), HEX_DUMP_BYTES);

        bool all_zero = true;
        for (size_t i = 0; i < ROW_BYTES; ++i) {
            if (out[i] != 0) { all_zero = false; break; }
        }
        if (all_zero) {
            std::cout << "[hint] temp is all zero -> OR not written to temp, or temp addressing mismatch.\n";
        }
        return false;
    }

    std::cout << "[PASS] CIM OR -> TEMP OK\n";
    return true;
}

int main()
{
    auto* rw  = reinterpret_cast<volatile uint64_t*>(DATA_BASE);
    auto* tmp = reinterpret_cast<volatile uint64_t*>(TEMP_BASE);
    auto* cmd = reinterpret_cast<volatile uint64_t*>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, MAT_BITS, ARRAY_BITS, ROW_BITS, COL_BITS);

    bool ok0  = test_one_bank(cim, 0);
    bool ok15 = test_one_bank(cim, 7);

    std::cout << "\n==== Summary ====\n";
    std::cout << "bank0 : " << (ok0  ? "PASS" : "FAIL") << "\n";
    std::cout << "bank15: " << (ok15 ? "PASS" : "FAIL") << "\n";

    return (ok0 && ok15) ? 0 : 1;
}