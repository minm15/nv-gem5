#include "cim_api.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

using RegionMap = std::unordered_map<CimModule::RegionKey,
                                     std::vector<uint8_t>,
                                     CimModule::RegionKeyHash>;

static inline std::vector<uint16_t> expand_mask16(uint16_t mask, unsigned bits)
{
    std::vector<uint16_t> ids;
    const uint16_t count = static_cast<uint16_t>(1u << bits);
    for (uint16_t i = 0; i < count; ++i) {
        if (((mask >> i) & 1u) != 0u) ids.push_back(i);
    }
    return ids;
}

static inline std::vector<uint16_t> expand_mask32(uint32_t mask, unsigned bits)
{
    std::vector<uint16_t> ids;
    const uint16_t count = static_cast<uint16_t>(1u << bits);
    for (uint16_t i = 0; i < count; ++i) {
        if (((mask >> i) & 1u) != 0u) ids.push_back(i);
    }
    return ids;
}

static std::vector<uint8_t>& region_buffer(RegionMap& storage,
                                           const CimModule::RegionKey& key,
                                           size_t rows_per_region,
                                           size_t row_bytes)
{
    auto it = storage.find(key);
    if (it == storage.end()) {
        it = storage.emplace(key, std::vector<uint8_t>(rows_per_region * row_bytes, 0u)).first;
    }
    return it->second;
}

static const std::vector<uint8_t>* find_region_buffer(const RegionMap& storage,
                                                      const CimModule::RegionKey& key)
{
    const auto it = storage.find(key);
    return it == storage.end() ? nullptr : &it->second;
}

static inline void validate_row_index(uint16_t row, unsigned row_bits)
{
    const uint16_t rows = static_cast<uint16_t>(1u << row_bits);
    if (row >= rows) throw std::runtime_error("cim_api_emu: row out of range");
}

} // namespace

CimModule::CommandEncode::CommandEncode(volatile uint64_t *const command_address)
    : commandAddress(command_address)
{
    operation_type = 0xffu;
    operation_flag_mask = 0u;
    byte_mask = 0xffu;
    bank_mask16 = 0xffffu;
    mat_mask32 = 0xffffffffu;
    array_mask32 = 0xffffffffu;
    column_mask = 0xffffffffffffffffull;
    dest = 0u;
    for (auto& row : row_number) row = 0xffffu;
}

void
CimModule::CommandEncode::print()
{
}

void
CimModule::CommandEncode::issue()
{
}

CimModule::CimModule(volatile uint64_t *const read_write_address,
                     volatile uint64_t *const temp_address,
                     volatile uint64_t *const command_address)
    : readWriteAddress(read_write_address),
      tempAddress(temp_address),
      commandWriteAddress(command_address),
      bankBits(0),
      matBits(0),
      arrayBits(0),
      rowBits(0),
      columnBits(0)
{
}

void
CimModule::setGeometry(unsigned num_bank_bits,
                       unsigned num_mat_bits,
                       unsigned num_array_bits,
                       unsigned num_row_bits,
                       unsigned num_column_bits)
{
    bankBits = num_bank_bits;
    matBits = num_mat_bits;
    arrayBits = num_array_bits;
    rowBits = num_row_bits;
    columnBits = num_column_bits;

    checkGeometryReady();
    rwStorage_.clear();
    tempStorage_.clear();
}

uint64_t
CimModule::Mask::bank(uint16_t b)
{
    if (b >= 16u) return 0ull;
    return (1ull << b);
}

uint64_t
CimModule::Mask::banksAll(unsigned bank_bits)
{
    const uint64_t n = 1ull << bank_bits;
    return (n >= 16ull) ? 0xffffull : ((1ull << n) - 1ull);
}

uint32_t
CimModule::Mask::mat(uint16_t m)
{
    if (m >= 32u) return 0u;
    return (1u << m);
}

uint32_t
CimModule::Mask::matsAll(unsigned mat_bits)
{
    const uint64_t n = 1ull << mat_bits;
    return (n >= 32ull) ? 0xffffffffu : static_cast<uint32_t>((1ull << n) - 1ull);
}

uint32_t
CimModule::Mask::array(uint16_t a)
{
    if (a >= 32u) return 0u;
    return (1u << a);
}

uint32_t
CimModule::Mask::arraysAll(unsigned array_bits)
{
    const uint64_t n = 1ull << array_bits;
    return (n >= 32ull) ? 0xffffffffu : static_cast<uint32_t>((1ull << n) - 1ull);
}

uint64_t
CimModule::Mask::colsAll()
{
    return 0xffffffffffffffffull;
}

void
CimModule::checkGeometryReady() const
{
    if (bankBits == 0 || bankBits > 16) throw std::runtime_error("cim_api_emu: invalid bankBits");
    if (matBits == 0 || matBits > 32) throw std::runtime_error("cim_api_emu: invalid matBits");
    if (arrayBits == 0 || arrayBits > 32) throw std::runtime_error("cim_api_emu: invalid arrayBits");
    if (rowBits == 0 || rowBits > 16) throw std::runtime_error("cim_api_emu: invalid rowBits");
    if (columnBits < kByteBits || columnBits > 63) throw std::runtime_error("cim_api_emu: invalid columnBits");
}

uint16_t
CimModule::normalizeBankMask16(uint64_t bank_mask) const
{
    const uint64_t valid = Mask::banksAll(bankBits);
    if (bank_mask == kAllOnes64) return static_cast<uint16_t>(valid & 0xffffull);
    if ((bank_mask & ~valid) != 0ull || bank_mask == 0ull) {
        throw std::runtime_error("cim_api_emu: invalid bank mask");
    }
    return static_cast<uint16_t>(bank_mask & 0xffffull);
}

uint32_t
CimModule::normalizeMatMask32(uint32_t mat_mask) const
{
    const uint32_t valid = Mask::matsAll(matBits);
    if (mat_mask == 0xffffffffu) return valid;
    if ((mat_mask & ~valid) != 0u || mat_mask == 0u) {
        throw std::runtime_error("cim_api_emu: invalid mat mask");
    }
    return mat_mask;
}

uint32_t
CimModule::normalizeArrayMask32(uint32_t array_mask) const
{
    const uint32_t valid = Mask::arraysAll(arrayBits);
    if (array_mask == 0xffffffffu) return valid;
    if ((array_mask & ~valid) != 0u || array_mask == 0u) {
        throw std::runtime_error("cim_api_emu: invalid array mask");
    }
    return array_mask;
}

uint64_t
CimModule::normalizeColumnMask(uint64_t column_mask) const
{
    const unsigned chunk_bits = columnBits - kByteBits;
    const uint64_t chunks = 1ull << chunk_bits;
    const uint64_t valid = (chunks >= 64ull) ? 0xffffffffffffffffull : ((1ull << chunks) - 1ull);

    if (column_mask == kAllOnes64) return valid;
    if ((column_mask & ~valid) != 0ull || column_mask == 0ull) {
        throw std::runtime_error("cim_api_emu: invalid column mask");
    }
    return column_mask;
}

uintptr_t
CimModule::calcRegionOffsetBytes(uint16_t bank,
                                 uint16_t mat,
                                 uint16_t array,
                                 uint16_t row) const
{
    checkGeometryReady();

    if (bank >= (1u << bankBits) ||
        mat >= (1u << matBits) ||
        array >= (1u << arrayBits) ||
        row >= (1u << rowBits)) {
        throw std::runtime_error("cim_api_emu: region address out of range");
    }

    uintptr_t offs = 0;
    offs += (static_cast<uintptr_t>(bank)  << (matBits + arrayBits + rowBits + columnBits));
    offs += (static_cast<uintptr_t>(mat)   << (arrayBits + rowBits + columnBits));
    offs += (static_cast<uintptr_t>(array) << (rowBits + columnBits));
    offs += (static_cast<uintptr_t>(row)   << columnBits);
    return offs;
}

void
CimModule::generateCommand(uint8_t,
                           const std::vector<uint16_t>&,
                           uint8_t,
                           uint64_t,
                           uint64_t,
                           uint16_t,
                           uint32_t,
                           uint32_t)
{
    throw std::runtime_error("cim_api_emu: generateCommand should not be called directly");
}

namespace {

enum class ReduceOp {
    And,
    Or,
    Xor,
};

static void reduce_rows(RegionMap& rw_storage,
                        RegionMap& temp_storage,
                        ReduceOp op,
                        const std::vector<uint16_t>& banks,
                        const std::vector<uint16_t>& mats,
                        const std::vector<uint16_t>& arrays,
                        unsigned row_bits,
                        unsigned column_bits,
                        const std::vector<uint16_t>& rows,
                        uint8_t byte_mask,
                        uint64_t column_mask,
                        uint16_t dest)
{
    const uint64_t col_sel = column_mask;
    validate_row_index(dest, row_bits);

    const size_t row_bytes = static_cast<size_t>(1ull << column_bits);
    const size_t rows_per_region = static_cast<size_t>(1ull << row_bits);
    const size_t chunks = row_bytes / 8u;

    for (uint16_t bank : banks) {
        for (uint16_t mat : mats) {
            for (uint16_t array : arrays) {
                const CimModule::RegionKey key{bank, mat, array};
                auto& temp_region = region_buffer(temp_storage, key, rows_per_region, row_bytes);
                uint8_t* dest_row = temp_region.data() + static_cast<size_t>(dest) * row_bytes;

                for (size_t chunk = 0; chunk < chunks; ++chunk) {
                    if (((col_sel >> chunk) & 1ull) == 0ull) continue;
                    for (size_t byte = 0; byte < 8u; ++byte) {
                        if (((byte_mask >> byte) & 1u) == 0u) continue;

                        const size_t idx = chunk * 8u + byte;
                        uint8_t acc = 0u;
                        bool first = true;

                        for (uint16_t row : rows) {
                            validate_row_index(row, row_bits);
                            const auto* region = find_region_buffer(rw_storage, key);
                            const uint8_t value = region == nullptr
                                ? 0u
                                : (*region)[static_cast<size_t>(row) * row_bytes + idx];

                            if (first) {
                                acc = value;
                                first = false;
                            } else if (op == ReduceOp::And) {
                                acc = static_cast<uint8_t>(acc & value);
                            } else if (op == ReduceOp::Or) {
                                acc = static_cast<uint8_t>(acc | value);
                            } else {
                                acc = static_cast<uint8_t>(acc ^ value);
                            }
                        }

                        dest_row[idx] = first ? 0u : acc;
                    }
                }
            }
        }
    }
}

} // namespace

void
CimModule::AND(const std::vector<uint16_t>& rows,
               uint8_t byte_mask,
               uint64_t bank_mask,
               uint64_t column_mask,
               uint16_t dest,
               uint32_t mat_mask,
               uint32_t array_mask)
{
    checkGeometryReady();
    const uint16_t bank_sel = normalizeBankMask16(bank_mask);
    const uint32_t mat_sel = normalizeMatMask32(mat_mask);
    const uint32_t array_sel = normalizeArrayMask32(array_mask);
    const uint64_t col_sel = normalizeColumnMask(column_mask);

    reduce_rows(rwStorage_, tempStorage_, ReduceOp::And,
                expand_mask16(bank_sel, bankBits),
                expand_mask32(mat_sel, matBits),
                expand_mask32(array_sel, arrayBits),
                rowBits,
                columnBits,
                rows, byte_mask, col_sel, dest);
}

void
CimModule::OR(const std::vector<uint16_t>& rows,
              uint8_t byte_mask,
              uint64_t bank_mask,
              uint64_t column_mask,
              uint16_t dest,
              uint32_t mat_mask,
              uint32_t array_mask)
{
    checkGeometryReady();
    const uint16_t bank_sel = normalizeBankMask16(bank_mask);
    const uint32_t mat_sel = normalizeMatMask32(mat_mask);
    const uint32_t array_sel = normalizeArrayMask32(array_mask);
    const uint64_t col_sel = normalizeColumnMask(column_mask);

    reduce_rows(rwStorage_, tempStorage_, ReduceOp::Or,
                expand_mask16(bank_sel, bankBits),
                expand_mask32(mat_sel, matBits),
                expand_mask32(array_sel, arrayBits),
                rowBits,
                columnBits,
                rows, byte_mask, col_sel, dest);
}

void
CimModule::XOR(const std::vector<uint16_t>& rows,
               uint8_t byte_mask,
               uint64_t bank_mask,
               uint64_t column_mask,
               uint16_t dest,
               uint32_t mat_mask,
               uint32_t array_mask)
{
    checkGeometryReady();
    const uint16_t bank_sel = normalizeBankMask16(bank_mask);
    const uint32_t mat_sel = normalizeMatMask32(mat_mask);
    const uint32_t array_sel = normalizeArrayMask32(array_mask);
    const uint64_t col_sel = normalizeColumnMask(column_mask);

    reduce_rows(rwStorage_, tempStorage_, ReduceOp::Xor,
                expand_mask16(bank_sel, bankBits),
                expand_mask32(mat_sel, matBits),
                expand_mask32(array_sel, arrayBits),
                rowBits,
                columnBits,
                rows, byte_mask, col_sel, dest);
}

void
CimModule::COPY(const uint16_t& dest,
                const uint16_t& src,
                const uint8_t& rotate_left,
                const uint8_t& byte_mask,
                const uint64_t& bank_mask,
                const uint64_t& column_mask,
                const uint32_t& mat_mask,
                const uint32_t& array_mask)
{
    checkGeometryReady();

    const uint16_t bank_sel = normalizeBankMask16(bank_mask);
    const uint32_t mat_sel = normalizeMatMask32(mat_mask);
    const uint32_t array_sel = normalizeArrayMask32(array_mask);
    const uint64_t col_sel = normalizeColumnMask(column_mask);
    validate_row_index(dest, rowBits);
    validate_row_index(src, rowBits);

    const size_t row_bytes = static_cast<size_t>(1ull << columnBits);
    const size_t rows_per_region = static_cast<size_t>(1ull << rowBits);
    const size_t chunks = row_bytes / 8u;
    const unsigned rot = static_cast<unsigned>(rotate_left) % static_cast<unsigned>(row_bytes);

    const auto banks = expand_mask16(bank_sel, bankBits);
    const auto mats = expand_mask32(mat_sel, matBits);
    const auto arrays = expand_mask32(array_sel, arrayBits);

    for (uint16_t bank : banks) {
        for (uint16_t mat : mats) {
            for (uint16_t array : arrays) {
                const RegionKey key{bank, mat, array};
                auto& temp_region = region_buffer(tempStorage_, key, rows_per_region, row_bytes);
                const auto* rw_region = find_region_buffer(rwStorage_, key);
                uint8_t* dest_row = temp_region.data() + static_cast<size_t>(dest) * row_bytes;

                for (size_t chunk = 0; chunk < chunks; ++chunk) {
                    if (((col_sel >> chunk) & 1ull) == 0ull) continue;
                    for (size_t byte = 0; byte < 8u; ++byte) {
                        if (((byte_mask >> byte) & 1u) == 0u) continue;
                        const size_t idx = chunk * 8u + byte;
                        const size_t src_idx = (idx + row_bytes - rot) % row_bytes;
                        const uint8_t value = rw_region == nullptr
                            ? 0u
                            : (*rw_region)[static_cast<size_t>(src) * row_bytes + src_idx];
                        dest_row[idx] = value;
                    }
                }
            }
        }
    }
}

void
CimModule::NOT_COND(const uint16_t& dest,
                    const uint16_t& src,
                    const bool& always_NOT,
                    const bool& NOT_if_zero,
                    const uint8_t& byte_mask,
                    const uint64_t& bank_mask,
                    const uint64_t& column_mask,
                    const uint32_t& mat_mask,
                    const uint32_t& array_mask)
{
    checkGeometryReady();

    const uint16_t bank_sel = normalizeBankMask16(bank_mask);
    const uint32_t mat_sel = normalizeMatMask32(mat_mask);
    const uint32_t array_sel = normalizeArrayMask32(array_mask);
    const uint64_t col_sel = normalizeColumnMask(column_mask);
    validate_row_index(dest, rowBits);
    validate_row_index(src, rowBits);

    const size_t row_bytes = static_cast<size_t>(1ull << columnBits);
    const size_t rows_per_region = static_cast<size_t>(1ull << rowBits);
    const size_t chunks = row_bytes / 8u;

    const auto banks = expand_mask16(bank_sel, bankBits);
    const auto mats = expand_mask32(mat_sel, matBits);
    const auto arrays = expand_mask32(array_sel, arrayBits);

    for (uint16_t bank : banks) {
        for (uint16_t mat : mats) {
            for (uint16_t array : arrays) {
                const RegionKey key{bank, mat, array};
                auto& temp_region = region_buffer(tempStorage_, key, rows_per_region, row_bytes);
                const auto* rw_region = find_region_buffer(rwStorage_, key);
                uint8_t* dest_row = temp_region.data() + static_cast<size_t>(dest) * row_bytes;
                const uint8_t* src_row = rw_region == nullptr
                    ? nullptr
                    : rw_region->data() + static_cast<size_t>(src) * row_bytes;

                bool src_all_zero = true;
                if (src_row != nullptr) {
                    for (size_t i = 0; i < row_bytes; ++i) {
                        if (src_row[i] != 0u) {
                            src_all_zero = false;
                            break;
                        }
                    }
                }

                const bool do_not = always_NOT || (NOT_if_zero && src_all_zero);

                for (size_t chunk = 0; chunk < chunks; ++chunk) {
                    if (((col_sel >> chunk) & 1ull) == 0ull) continue;
                    for (size_t byte = 0; byte < 8u; ++byte) {
                        if (((byte_mask >> byte) & 1u) == 0u) continue;
                        const size_t idx = chunk * 8u + byte;
                        const uint8_t value = src_row == nullptr ? 0u : src_row[idx];
                        dest_row[idx] = do_not ? static_cast<uint8_t>(~value) : value;
                    }
                }
            }
        }
    }
}

void
CimModule::copy_to_cim(uint16_t bank,
                       uint16_t mat,
                       uint16_t array,
                       uint16_t row,
                       void* cpu_array,
                       size_t size_in_byte)
{
    checkGeometryReady();
    validate_row_index(row, rowBits);

    const size_t row_bytes = static_cast<size_t>(1ull << columnBits);
    const size_t rows_per_region = static_cast<size_t>(1ull << rowBits);
    if (size_in_byte != row_bytes) throw std::runtime_error("cim_api_emu: unexpected row size");

    auto& region = region_buffer(rwStorage_, RegionKey{bank, mat, array}, rows_per_region, row_bytes);
    std::memcpy(region.data() + static_cast<size_t>(row) * row_bytes, cpu_array, row_bytes);
}

void
CimModule::copy_to_cpu(void* cpu_array,
                       uint16_t bank,
                       uint16_t mat,
                       uint16_t array,
                       uint16_t row,
                       size_t size_in_byte)
{
    checkGeometryReady();
    validate_row_index(row, rowBits);

    const size_t row_bytes = static_cast<size_t>(1ull << columnBits);
    if (size_in_byte != row_bytes) throw std::runtime_error("cim_api_emu: unexpected row size");

    const auto* region = find_region_buffer(rwStorage_, RegionKey{bank, mat, array});
    if (region == nullptr) {
        std::memset(cpu_array, 0, row_bytes);
        return;
    }

    std::memcpy(cpu_array,
                region->data() + static_cast<size_t>(row) * row_bytes,
                row_bytes);
}

void
CimModule::copy_temp_to_cpu(void* cpu_array,
                            uint16_t bank,
                            uint16_t mat,
                            uint16_t array,
                            uint16_t row,
                            size_t size_in_byte)
{
    checkGeometryReady();
    validate_row_index(row, rowBits);

    const size_t row_bytes = static_cast<size_t>(1ull << columnBits);
    if (size_in_byte != row_bytes) throw std::runtime_error("cim_api_emu: unexpected temp row size");

    const auto* region = find_region_buffer(tempStorage_, RegionKey{bank, mat, array});
    if (region == nullptr) {
        std::memset(cpu_array, 0, row_bytes);
        return;
    }

    std::memcpy(cpu_array,
                region->data() + static_cast<size_t>(row) * row_bytes,
                row_bytes);
}

void
CimModule::copy_temp_block_to_cpu(void* cpu_array,
                                  uint16_t bank,
                                  uint16_t mat,
                                  uint16_t array,
                                  uint16_t start_row,
                                  size_t num_rows)
{
    checkGeometryReady();
    if (num_rows == 0u) throw std::runtime_error("cim_api_emu: num_rows must be positive");

    const size_t row_bytes = static_cast<size_t>(1ull << columnBits);
    const auto* region = find_region_buffer(tempStorage_, RegionKey{bank, mat, array});

    auto* out = static_cast<uint8_t*>(cpu_array);
    for (size_t i = 0; i < num_rows; ++i) {
        const uint16_t row = static_cast<uint16_t>(start_row + i);
        validate_row_index(row, rowBits);
        if (region == nullptr) {
            std::memset(out + i * row_bytes, 0, row_bytes);
            continue;
        }
        std::memcpy(out + i * row_bytes,
                    region->data() + static_cast<size_t>(row) * row_bytes,
                    row_bytes);
    }
}

void
CimModule::do_pseudo_push(void*, uintptr_t, size_t) const
{
}
