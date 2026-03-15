#include "cim_api.hpp"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x18000000;
static constexpr uintptr_t CMD_BASE  = 0x20000000;

namespace {

static constexpr unsigned kBankBits = 4;
static constexpr unsigned kMatBits = 4;
static constexpr unsigned kArrayBits = 4;
static constexpr unsigned kRowBits = 9;
static constexpr unsigned kColBits = 6;
static constexpr size_t kRowBytes = (1u << kColBits);

struct Region {
    uint16_t bank;
    uint16_t mat;
    uint16_t array;
};

std::vector<uint8_t> make_pattern_row(uint32_t seed)
{
    std::vector<uint8_t> row(kRowBytes, 0u);
    for (size_t i = 0; i < row.size(); ++i) {
        const uint32_t value = seed * 29u + static_cast<uint32_t>(i) * 17u + ((i & 1u) ? 0x55u : 0xaau);
        row[i] = static_cast<uint8_t>(value & 0xffu);
    }
    return row;
}

std::vector<uint8_t> reduce_rows(const std::vector<std::vector<uint8_t>>& rows,
                                 char op)
{
    std::vector<uint8_t> out(kRowBytes, 0u);
    if (rows.empty()) return out;

    out = rows.front();
    for (size_t r = 1; r < rows.size(); ++r) {
        for (size_t i = 0; i < out.size(); ++i) {
            if (op == '|') out[i] = static_cast<uint8_t>(out[i] | rows[r][i]);
            else if (op == '&') out[i] = static_cast<uint8_t>(out[i] & rows[r][i]);
            else if (op == '^') out[i] = static_cast<uint8_t>(out[i] ^ rows[r][i]);
        }
    }
    return out;
}

std::string summarize_row_prefix(const std::vector<uint8_t>& row, size_t count = 8)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    const size_t n = std::min(count, row.size());
    for (size_t i = 0; i < n; ++i) {
        if (i != 0u) oss << " ";
        oss << std::setw(2) << static_cast<unsigned>(row[i]);
    }
    return oss.str();
}

bool expect_equal(const std::string& label,
                  const std::vector<uint8_t>& actual,
                  const std::vector<uint8_t>& expected)
{
    if (actual == expected) {
        std::cout << "[pass] " << label << "\n";
        return true;
    }

    size_t first_diff = 0u;
    while (first_diff < actual.size() && actual[first_diff] == expected[first_diff]) {
        ++first_diff;
    }

    std::cerr << "[mismatch] " << label
              << " first_diff_byte=" << first_diff
              << " actual_prefix=[" << summarize_row_prefix(actual) << "]"
              << " expected_prefix=[" << summarize_row_prefix(expected) << "]"
              << "\n";
    return false;
}

bool read_rw_row(CimModule& cim,
                 const Region& region,
                 uint16_t row,
                 std::vector<uint8_t>& out)
{
    out.assign(kRowBytes, 0u);
    cim.copy_to_cpu(out.data(), region.bank, region.mat, region.array, row, kRowBytes);
    return true;
}

bool read_temp_row(CimModule& cim,
                   const Region& region,
                   uint16_t row,
                   std::vector<uint8_t>& out)
{
    out.assign(kRowBytes, 0u);
    cim.copy_temp_to_cpu(out.data(), region.bank, region.mat, region.array, row, kRowBytes);
    return true;
}

bool read_temp_block(CimModule& cim,
                     const Region& region,
                     uint16_t start_row,
                     size_t num_rows,
                     std::vector<uint8_t>& out)
{
    out.assign(num_rows * kRowBytes, 0u);
    cim.copy_temp_block_to_cpu(out.data(),
                               region.bank, region.mat, region.array,
                               start_row, num_rows);
    return true;
}

void write_rw_row(CimModule& cim,
                  const Region& region,
                  uint16_t row,
                  const std::vector<uint8_t>& data)
{
    cim.copy_to_cim(region.bank, region.mat, region.array, row,
                    const_cast<uint8_t*>(data.data()), kRowBytes);
}

} // namespace

int main()
{
    auto* rw = reinterpret_cast<volatile uint64_t*>(DATA_BASE);
    auto* tmp = reinterpret_cast<volatile uint64_t*>(TEMP_BASE);
    auto* cmd = reinterpret_cast<volatile uint64_t*>(CMD_BASE);

    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(kBankBits, kMatBits, kArrayBits, kRowBits, kColBits);

    const Region region_a{0u, 0u, 0u};
    const Region region_b{1u, 2u, 3u};

    bool ok = true;

    const auto row10 = make_pattern_row(10u);
    const auto row11 = make_pattern_row(11u);
    const auto row12 = make_pattern_row(12u);
    const auto row13 = make_pattern_row(13u);
    const auto zero_row = std::vector<uint8_t>(kRowBytes, 0u);

    write_rw_row(cim, region_a, 10u, row10);
    write_rw_row(cim, region_a, 11u, row11);
    write_rw_row(cim, region_a, 12u, row12);
    write_rw_row(cim, region_a, 13u, row13);
    write_rw_row(cim, region_a, 20u, zero_row);

    write_rw_row(cim, region_b, 30u, row10);
    write_rw_row(cim, region_b, 31u, row11);
    write_rw_row(cim, region_b, 32u, row12);
    write_rw_row(cim, region_b, 33u, row13);

    {
        std::vector<uint8_t> actual;
        read_rw_row(cim, region_b, 31u, actual);
        ok &= expect_equal("copy_to_cim + copy_to_cpu roundtrip", actual, row11);
    }

    {
        cim.OR({10u, 11u}, 0xffu,
               CimModule::Mask::bank(region_a.bank),
               CimModule::Mask::colsAll(),
               5u,
               CimModule::Mask::mat(region_a.mat),
               CimModule::Mask::array(region_a.array));

        std::vector<uint8_t> actual;
        read_temp_row(cim, region_a, 5u, actual);
        ok &= expect_equal("OR of 2 rows", actual, reduce_rows({row10, row11}, '|'));
    }

    {
        cim.OR({30u, 31u, 32u, 33u}, 0xffu,
               CimModule::Mask::bank(region_b.bank),
               CimModule::Mask::colsAll(),
               6u,
               CimModule::Mask::mat(region_b.mat),
               CimModule::Mask::array(region_b.array));

        std::vector<uint8_t> actual;
        read_temp_row(cim, region_b, 6u, actual);
        ok &= expect_equal("OR of 4 rows on nonzero bank/mat/array",
                           actual, reduce_rows({row10, row11, row12, row13}, '|'));
    }

    {
        cim.OR({10u, 11u}, 0xffu,
               CimModule::Mask::bank(region_a.bank),
               CimModule::Mask::colsAll(),
               7u,
               CimModule::Mask::mat(region_a.mat),
               CimModule::Mask::array(region_a.array));
        cim.OR({12u, 13u}, 0xffu,
               CimModule::Mask::bank(region_a.bank),
               CimModule::Mask::colsAll(),
               7u,
               CimModule::Mask::mat(region_a.mat),
               CimModule::Mask::array(region_a.array));

        std::vector<uint8_t> actual;
        read_temp_row(cim, region_a, 7u, actual);
        ok &= expect_equal("OR overwrites reused temp dest",
                           actual, reduce_rows({row12, row13}, '|'));
    }

    {
        cim.OR({10u, 11u}, 0xffu,
               CimModule::Mask::bank(region_a.bank),
               CimModule::Mask::colsAll(),
               21u,
               CimModule::Mask::mat(region_a.mat),
               CimModule::Mask::array(region_a.array));
        cim.OR({11u, 12u}, 0xffu,
               CimModule::Mask::bank(region_a.bank),
               CimModule::Mask::colsAll(),
               22u,
               CimModule::Mask::mat(region_a.mat),
               CimModule::Mask::array(region_a.array));
        cim.OR({12u, 13u}, 0xffu,
               CimModule::Mask::bank(region_a.bank),
               CimModule::Mask::colsAll(),
               23u,
               CimModule::Mask::mat(region_a.mat),
               CimModule::Mask::array(region_a.array));
        cim.OR({10u, 13u}, 0xffu,
               CimModule::Mask::bank(region_a.bank),
               CimModule::Mask::colsAll(),
               24u,
               CimModule::Mask::mat(region_a.mat),
               CimModule::Mask::array(region_a.array));

        std::vector<uint8_t> block_actual;
        read_temp_block(cim, region_a, 21u, 4u, block_actual);

        std::vector<uint8_t> block_expected;
        for (uint16_t row = 21u; row <= 24u; ++row) {
            std::vector<uint8_t> single;
            read_temp_row(cim, region_a, row, single);
            block_expected.insert(block_expected.end(), single.begin(), single.end());
        }

        ok &= expect_equal("copy_temp_block_to_cpu matches per-row temp reads",
                           block_actual, block_expected);
    }

    {
        cim.AND({10u, 11u, 12u}, 0xffu,
                CimModule::Mask::bank(region_a.bank),
                CimModule::Mask::colsAll(),
                8u,
                CimModule::Mask::mat(region_a.mat),
                CimModule::Mask::array(region_a.array));

        std::vector<uint8_t> actual;
        read_temp_row(cim, region_a, 8u, actual);
        ok &= expect_equal("AND of 3 rows", actual, reduce_rows({row10, row11, row12}, '&'));
    }

    {
        cim.XOR({10u, 11u, 12u, 13u}, 0xffu,
                CimModule::Mask::bank(region_a.bank),
                CimModule::Mask::colsAll(),
                9u,
                CimModule::Mask::mat(region_a.mat),
                CimModule::Mask::array(region_a.array));

        std::vector<uint8_t> actual;
        read_temp_row(cim, region_a, 9u, actual);
        ok &= expect_equal("XOR of 4 rows", actual, reduce_rows({row10, row11, row12, row13}, '^'));
    }

    {
        cim.OR({30u, 31u}, 0xffu,
               CimModule::Mask::bank(region_b.bank),
               CimModule::Mask::colsAll(),
               40u,
               CimModule::Mask::mat(region_b.mat),
               CimModule::Mask::array(region_b.array));
        cim.OR({31u, 32u}, 0xffu,
               CimModule::Mask::bank(region_b.bank),
               CimModule::Mask::colsAll(),
               41u,
               CimModule::Mask::mat(region_b.mat),
               CimModule::Mask::array(region_b.array));
        cim.OR({32u, 33u}, 0xffu,
               CimModule::Mask::bank(region_b.bank),
               CimModule::Mask::colsAll(),
               42u,
               CimModule::Mask::mat(region_b.mat),
               CimModule::Mask::array(region_b.array));

        std::vector<uint8_t> block_actual;
        read_temp_block(cim, region_b, 40u, 3u, block_actual);

        const auto expect40 = reduce_rows({row10, row11}, '|');
        const auto expect41 = reduce_rows({row11, row12}, '|');
        const auto expect42 = reduce_rows({row12, row13}, '|');

        std::vector<uint8_t> block_expected;
        block_expected.reserve(3u * kRowBytes);
        block_expected.insert(block_expected.end(), expect40.begin(), expect40.end());
        block_expected.insert(block_expected.end(), expect41.begin(), expect41.end());
        block_expected.insert(block_expected.end(), expect42.begin(), expect42.end());

        ok &= expect_equal("copy_temp_block_to_cpu matches expected OR results on nonzero region",
                           block_actual, block_expected);
    }

    if (!ok) {
        std::cerr << "[fatal] CIM primitive verification failed for IVF-required ops\n";
        return 1;
    }

    std::cout << "[pass] all IVF-required CIM primitive checks passed\n";
    return 0;
}
