#include "cim_api.hpp"
#include <gem5/m5ops.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x14000000;
static constexpr uintptr_t CMD_BASE  = 0x18000000;

static constexpr unsigned BANK_BITS  = 3;
static constexpr unsigned MAT_BITS   = 4;
static constexpr unsigned ARRAY_BITS = 4;
static constexpr unsigned ROW_BITS   = 9;
static constexpr unsigned COL_BITS   = 6;

static constexpr size_t ROW_BYTES = (1ull << COL_BITS);

static constexpr uint16_t ROW_A = 0;
static constexpr uint16_t ROW_B = 1;

static constexpr uint16_t DEST_AND = 10;
static constexpr uint16_t DEST_OR  = 11;
static constexpr uint16_t DEST_XOR = 12;

static void fill_pattern(uint8_t *buf, size_t n, uint32_t seed)
{
    uint32_t x = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        buf[i] = static_cast<uint8_t>((x >> 24) ^ (i * 13u));
    }
}

static void ref_and(uint8_t *out, const uint8_t *a, const uint8_t *b, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = static_cast<uint8_t>(a[i] & b[i]);
}

static void ref_or(uint8_t *out, const uint8_t *a, const uint8_t *b, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = static_cast<uint8_t>(a[i] | b[i]);
}

static void ref_xor(uint8_t *out, const uint8_t *a, const uint8_t *b, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = static_cast<uint8_t>(a[i] ^ b[i]);
}

static bool equal_buf(const uint8_t *x, const uint8_t *y, size_t n, size_t &bad_i)
{
    for (size_t i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            bad_i = i;
            return false;
        }
    }
    return true;
}

static void dump_mismatch(const char *tag,
                          uint16_t bank,
                          uint16_t mat,
                          uint16_t array,
                          const uint8_t *got,
                          const uint8_t *exp,
                          size_t n)
{
    size_t bad_i = 0;
    (void)equal_buf(got, exp, n, bad_i);

    std::cout << "[FAIL] " << tag
              << " bank=" << bank
              << " mat=" << mat
              << " array=" << array
              << "\n  first mismatch at byte " << bad_i
              << " got=0x" << std::hex << static_cast<int>(got[bad_i])
              << " exp=0x" << static_cast<int>(exp[bad_i]) << std::dec
              << "\n";
}

int main()
{
    static_assert(ROW_BYTES == 64, "expect 64B row");

    auto *rw  = reinterpret_cast<volatile uint64_t *>(DATA_BASE);
    auto *tmp = reinterpret_cast<volatile uint64_t *>(TEMP_BASE);
    auto *cmd = reinterpret_cast<volatile uint64_t *>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, MAT_BITS, ARRAY_BITS, ROW_BITS, COL_BITS);

    alignas(64) std::array<uint8_t, ROW_BYTES> a{};
    alignas(64) std::array<uint8_t, ROW_BYTES> b{};
    alignas(64) std::array<uint8_t, ROW_BYTES> got{};
    alignas(64) std::array<uint8_t, ROW_BYTES> exp{};

    const uint16_t banks = 1u << BANK_BITS;
    const uint16_t mats  = 1u << MAT_BITS;
    const uint16_t arrays = 1u << ARRAY_BITS;

    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);

    for (uint16_t bank = 0; bank < banks; bank++) {
        for (uint16_t mat = 0; mat < mats; mat++) {
            for (uint16_t array = 0; array < arrays; array++) {

                uint32_t seedA = (static_cast<uint32_t>(bank) << 24)
                               ^ (static_cast<uint32_t>(mat)  << 16)
                               ^ (static_cast<uint32_t>(array) << 8)
                               ^ 0xA5u;
                uint32_t seedB = seedA ^ 0x5Au;

                fill_pattern(a.data(), a.size(), seedA);
                fill_pattern(b.data(), b.size(), seedB);

                cim.copy_to_cim(bank, mat, array, ROW_A, a.data(), ROW_BYTES);
                cim.copy_to_cim(bank, mat, array, ROW_B, b.data(), ROW_BYTES);

                std::vector<uint16_t> rows{ROW_A, ROW_B};

                cim.AND(rows,
                        0xffu,
                        CimModule::Mask::bank(bank),
                        CimModule::Mask::colsAll(),
                        DEST_AND,
                        CimModule::Mask::mat(mat),
                        CimModule::Mask::array(array));

                cim.OR(rows,
                       0xffu,
                       CimModule::Mask::bank(bank),
                       CimModule::Mask::colsAll(),
                       DEST_OR,
                       CimModule::Mask::mat(mat),
                       CimModule::Mask::array(array));

                cim.XOR(rows,
                        0xffu,
                        CimModule::Mask::bank(bank),
                        CimModule::Mask::colsAll(),
                        DEST_XOR,
                        CimModule::Mask::mat(mat),
                        CimModule::Mask::array(array));
            }
        }
    }

    m5_work_end(0, 0);
    m5_dump_stats(0, 0);

    bool ok = true;
    uint64_t total = 0;
    uint64_t failed = 0;

    for (uint16_t bank = 0; bank < banks; bank++) {
        for (uint16_t mat = 0; mat < mats; mat++) {
            for (uint16_t array = 0; array < arrays; array++) {
                total++;

                uint32_t seedA = (static_cast<uint32_t>(bank) << 24)
                               ^ (static_cast<uint32_t>(mat)  << 16)
                               ^ (static_cast<uint32_t>(array) << 8)
                               ^ 0xA5u;
                uint32_t seedB = seedA ^ 0x5Au;

                fill_pattern(a.data(), a.size(), seedA);
                fill_pattern(b.data(), b.size(), seedB);

                ref_and(exp.data(), a.data(), b.data(), ROW_BYTES);
                cim.copy_temp_to_cpu(got.data(), bank, mat, array, DEST_AND, ROW_BYTES);
                size_t bad_i = 0;
                if (!equal_buf(got.data(), exp.data(), ROW_BYTES, bad_i)) {
                    ok = false;
                    failed++;
                    dump_mismatch("AND", bank, mat, array, got.data(), exp.data(), ROW_BYTES);
                    goto done_check;
                }

                ref_or(exp.data(), a.data(), b.data(), ROW_BYTES);
                cim.copy_temp_to_cpu(got.data(), bank, mat, array, DEST_OR, ROW_BYTES);
                if (!equal_buf(got.data(), exp.data(), ROW_BYTES, bad_i)) {
                    ok = false;
                    failed++;
                    dump_mismatch("OR", bank, mat, array, got.data(), exp.data(), ROW_BYTES);
                    goto done_check;
                }

                ref_xor(exp.data(), a.data(), b.data(), ROW_BYTES);
                cim.copy_temp_to_cpu(got.data(), bank, mat, array, DEST_XOR, ROW_BYTES);
                if (!equal_buf(got.data(), exp.data(), ROW_BYTES, bad_i)) {
                    ok = false;
                    failed++;
                    dump_mismatch("XOR", bank, mat, array, got.data(), exp.data(), ROW_BYTES);
                    goto done_check;
                }
            }
        }
    }

done_check:
    std::cout << (ok ? "ALL PASS\n" : "TEST FAIL\n");
    std::cout << "checked arrays = " << total
              << " failed = " << failed << "\n";
    return ok ? 0 : 1;
}