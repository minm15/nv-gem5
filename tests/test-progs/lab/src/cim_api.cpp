#include "cim_api.hpp"

static constexpr uint64_t kAllOnes64 = 0xffffffffffffffffull;

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

    assert(bankBits > 0);
    assert(bankBits <= 16);

    assert(matBits > 0);
    assert(matBits <= 32);

    assert(arrayBits > 0);
    assert(arrayBits <= 32);

    assert(rowBits > 0);
    assert(rowBits <= 16);

    assert(columnBits > 0);
    assert(columnBits <= 63);
    assert(columnBits >= kByteBits);
}

uint64_t
CimModule::Mask::bank(uint16_t b)
{
    if (b >= 16) return 0ull;
    return (1ull << b);
}

uint64_t
CimModule::Mask::banksAll(unsigned bank_bits)
{
    // number of banks = 2^bank_bits, mask is bank_id bits
    const uint64_t n = 1ull << bank_bits;
    if (n >= 16) return 0xffffull;                 // bank_mask16 only supports 16 bits
    return (n == 0) ? 0ull : ((1ull << n) - 1ull); // e.g. bank_bits=3 => n=8 => 0xFF
}

uint32_t
CimModule::Mask::mat(uint16_t m)
{
    if (m >= 32) return 0u;
    return (1u << m);
}

uint32_t
CimModule::Mask::matsAll(unsigned mat_bits)
{
    const uint64_t n = 1ull << mat_bits;           // mats = 2^mat_bits
    if (n >= 32) return 0xffffffffu;               // mat_mask32 only supports 32 bits
    return (n == 0) ? 0u : static_cast<uint32_t>((1ull << n) - 1ull); // mat_bits=4 => n=16 => 0xFFFF
}

uint32_t
CimModule::Mask::array(uint16_t a)
{
    if (a >= 32) return 0u;
    return (1u << a);
}

uint32_t
CimModule::Mask::arraysAll(unsigned array_bits)
{
    const uint64_t n = 1ull << array_bits;         // arrays = 2^array_bits
    if (n >= 32) return 0xffffffffu;               // array_mask32 only supports 32 bits
    return (n == 0) ? 0u : static_cast<uint32_t>((1ull << n) - 1ull); // array_bits=4 => n=16 => 0xFFFF
}

uint64_t
CimModule::Mask::colsAll()
{
    return kAllOnes64;
}

void
CimModule::checkGeometryReady() const
{
    assert(bankBits > 0);
    assert(bankBits <= 16);

    assert(matBits > 0);
    assert(matBits <= 32);

    assert(arrayBits > 0);
    assert(arrayBits <= 32);

    assert(rowBits > 0);
    assert(rowBits <= 16);

    assert(columnBits > 0);
    assert(columnBits <= 63);
    assert(columnBits >= kByteBits);
}

uint16_t
CimModule::normalizeBankMask16(uint64_t bank_mask) const
{
    const uint64_t valid = Mask::banksAll(bankBits);

    if (bank_mask == kAllOnes64) {
        return static_cast<uint16_t>(valid & 0xffffull);
    }

    assert((bank_mask & ~valid) == 0ull);
    return static_cast<uint16_t>(bank_mask & 0xffffull);
}

uint32_t
CimModule::normalizeMatMask32(uint32_t mat_mask) const
{
    const uint32_t valid = Mask::matsAll(matBits);

    if (mat_mask == 0xffffffffu) {
        return valid;
    }

    assert((mat_mask & ~valid) == 0u);
    assert(mat_mask != 0u);
    return mat_mask;
}

uint32_t
CimModule::normalizeArrayMask32(uint32_t array_mask) const
{
    const uint32_t valid = Mask::arraysAll(arrayBits);

    if (array_mask == 0xffffffffu) {
        return valid;
    }

    assert((array_mask & ~valid) == 0u);
    assert(array_mask != 0u);
    return array_mask;
}

uint64_t
CimModule::normalizeColumnMask(uint64_t column_mask) const
{
    const unsigned chunk_bits = columnBits - kByteBits;
    const uint64_t chunks = 1ull << chunk_bits; // number of 8-byte chunks per row

    uint64_t valid = 0ull;
    if (chunks >= 64) {
        valid = 0xffffffffffffffffull;
    } else {
        valid = (chunks == 0) ? 0ull : ((1ull << chunks) - 1ull); // COL_BITS=6 => chunks=8 => valid=0xFF
    }

    if (column_mask == kAllOnes64) {
        return valid;
    }

    assert((column_mask & ~valid) == 0ull);
    assert(column_mask != 0ull);
    return column_mask;
}

uintptr_t
CimModule::calcRegionOffsetBytes(uint16_t bank,
                                 uint16_t mat,
                                 uint16_t array,
                                 uint16_t row) const
{
    assert(bank < (1u << bankBits));
    assert(mat < (1u << matBits));
    assert(array < (1u << arrayBits));
    assert(row < (1u << rowBits));

    uintptr_t offs = 0;
    offs += (static_cast<uintptr_t>(bank)  << (matBits + arrayBits + rowBits + columnBits));
    offs += (static_cast<uintptr_t>(mat)   << (arrayBits + rowBits + columnBits));
    offs += (static_cast<uintptr_t>(array) << (rowBits + columnBits));
    offs += (static_cast<uintptr_t>(row)   << (columnBits));
    return offs;
}

CimModule::CommandEncode::CommandEncode(volatile uint64_t *const command_address)
    : commandAddress(command_address)
{
    operation_type = 0xff;
    operation_flag_mask = 0;
    byte_mask = 0xffu;

    bank_mask16 = 0xffffu;
    mat_mask32 = 0xffffffffu;
    array_mask32 = 0xffffffffu;

    column_mask = kAllOnes64;

    dest = 0;

    for (int i = 0; i < 4; ++i) {
        row_number[i] = 0xffffu;
    }
}

void
CimModule::generateCommand(uint8_t op_type,
                           const std::vector<uint16_t> &rows,
                           uint8_t byte_mask,
                           uint64_t bank_mask,
                           uint64_t column_mask,
                           uint16_t dest,
                           uint32_t mat_mask,
                           uint32_t array_mask)
{
    checkGeometryReady();

    assert(rows.size() >= 2);
    assert(rows.size() <= 4);
    assert(byte_mask != 0);

    CommandEncode c1(this->commandWriteAddress);

    c1.operation_type = op_type;
    c1.byte_mask = byte_mask;
    c1.dest = dest;

    c1.bank_mask16 = normalizeBankMask16(bank_mask);
    c1.mat_mask32 = normalizeMatMask32(mat_mask);
    c1.array_mask32 = normalizeArrayMask32(array_mask);
    c1.column_mask = normalizeColumnMask(column_mask);

    c1.operation_flag_mask = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        c1.row_number[i] = rows[i];
        c1.operation_flag_mask |= static_cast<uint8_t>(1u << i);
    }

    c1.issue();
}

void
CimModule::AND(const std::vector<uint16_t> &rows,
               uint8_t byte_mask,
               uint64_t bank_mask,
               uint64_t column_mask,
               uint16_t dest,
               uint32_t mat_mask,
               uint32_t array_mask)
{
    generateCommand(0, rows, byte_mask, bank_mask, column_mask, dest, mat_mask, array_mask);
}

void
CimModule::OR(const std::vector<uint16_t> &rows,
              uint8_t byte_mask,
              uint64_t bank_mask,
              uint64_t column_mask,
              uint16_t dest,
              uint32_t mat_mask,
              uint32_t array_mask)
{
    generateCommand(1, rows, byte_mask, bank_mask, column_mask, dest, mat_mask, array_mask);
}

void
CimModule::XOR(const std::vector<uint16_t> &rows,
               uint8_t byte_mask,
               uint64_t bank_mask,
               uint64_t column_mask,
               uint16_t dest,
               uint32_t mat_mask,
               uint32_t array_mask)
{
    generateCommand(2, rows, byte_mask, bank_mask, column_mask, dest, mat_mask, array_mask);
}

void
CimModule::COPY(const uint16_t &dest,
                const uint16_t &src,
                const uint8_t &rotate_left,
                const uint8_t &byte_mask,
                const uint64_t &bank_mask,
                const uint64_t &column_mask,
                const uint32_t &mat_mask,
                const uint32_t &array_mask)
{
    checkGeometryReady();

    assert(byte_mask != 0);

    CommandEncode c1(this->commandWriteAddress);

    c1.operation_type = 3;
    c1.byte_mask = byte_mask;
    c1.dest = dest;

    c1.bank_mask16 = normalizeBankMask16(bank_mask);
    c1.mat_mask32 = normalizeMatMask32(mat_mask);
    c1.array_mask32 = normalizeArrayMask32(array_mask);
    c1.column_mask = normalizeColumnMask(column_mask);

    c1.row_number[0] = src;
    c1.operation_flag_mask = rotate_left;

    c1.issue();
}

void
CimModule::NOT_COND(const uint16_t &dest,
                    const uint16_t &src,
                    const bool &always_NOT,
                    const bool &NOT_if_zero,
                    const uint8_t &byte_mask,
                    const uint64_t &bank_mask,
                    const uint64_t &column_mask,
                    const uint32_t &mat_mask,
                    const uint32_t &array_mask)
{
    checkGeometryReady();

    assert(byte_mask != 0);

    CommandEncode c1(this->commandWriteAddress);

    c1.operation_type = 4;
    c1.byte_mask = byte_mask;
    c1.dest = dest;

    c1.bank_mask16 = normalizeBankMask16(bank_mask);
    c1.mat_mask32 = normalizeMatMask32(mat_mask);
    c1.array_mask32 = normalizeArrayMask32(array_mask);
    c1.column_mask = normalizeColumnMask(column_mask);

    c1.row_number[0] = src;

    c1.operation_flag_mask = 0;
    if (always_NOT) c1.operation_flag_mask = static_cast<uint8_t>(c1.operation_flag_mask + 2);
    if (NOT_if_zero) c1.operation_flag_mask = static_cast<uint8_t>(c1.operation_flag_mask + 1);

    c1.issue();
}

void
CimModule::copy_to_cim(uint16_t bank,
                       uint16_t mat,
                       uint16_t array,
                       uint16_t row,
                       void *cpu_array,
                       size_t size_in_byte)
{
    checkGeometryReady();

    assert(cpu_array != nullptr);
    assert(readWriteAddress != nullptr);
    assert(size_in_byte == (1ull << columnBits));

    const uintptr_t base = reinterpret_cast<uintptr_t>(readWriteAddress);
    const uintptr_t offs = calcRegionOffsetBytes(bank, mat, array, row);

    uint8_t *dest = reinterpret_cast<uint8_t *>(base + offs);
    std::memcpy(dest, cpu_array, size_in_byte);
}

void
CimModule::copy_to_cpu(void *cpu_array,
                       uint16_t bank,
                       uint16_t mat,
                       uint16_t array,
                       uint16_t row,
                       size_t size_in_byte)
{
    checkGeometryReady();

    assert(cpu_array != nullptr);
    assert(readWriteAddress != nullptr);
    assert(size_in_byte == (1ull << columnBits));

    const uintptr_t base = reinterpret_cast<uintptr_t>(readWriteAddress);
    const uintptr_t offs = calcRegionOffsetBytes(bank, mat, array, row);

    const uint8_t *src = reinterpret_cast<const uint8_t *>(base + offs);
    std::memcpy(cpu_array, src, size_in_byte);
}

void
CimModule::copy_temp_to_cpu(void *cpu_array,
                            uint16_t bank,
                            uint16_t mat,
                            uint16_t array,
                            uint16_t row,
                            size_t size_in_byte)
{
    checkGeometryReady();

    assert(cpu_array != nullptr);
    assert(tempAddress != nullptr);
    assert(size_in_byte == (1ull << columnBits));

    const uintptr_t base = reinterpret_cast<uintptr_t>(tempAddress);
    const uintptr_t offs = calcRegionOffsetBytes(bank, mat, array, row);

    const uint8_t *src = reinterpret_cast<const uint8_t *>(base + offs);
    std::memcpy(cpu_array, src, size_in_byte);
}

void
CimModule::CommandEncode::print()
{
    printf("-------\n");
    printf("type: %02x, flag: %02x, byte_mask: %02x\n",
           operation_type, operation_flag_mask, byte_mask);
    printf("bank_mask16: 0x%04x, mat_mask32: 0x%08x, array_mask32: 0x%08x\n",
           bank_mask16, mat_mask32, array_mask32);
    printf("column_mask: 0x%016" PRIx64 "\n", column_mask);
    printf("dest: %04x\n", dest);
    printf("row0: %04x\n", row_number[0]);
    printf("row1: %04x\n", row_number[1]);
    printf("row2: %04x\n", row_number[2]);
    printf("row3: %04x\n", row_number[3]);
    printf("------\n");
}

void
CimModule::CommandEncode::issue()
{
    uint64_t w0 = 0;
    uint64_t w1 = 0;
    uint64_t w2 = 0;
    uint64_t w3 = 0;

    w0 |= (static_cast<uint64_t>(operation_type))      << 56;
    w0 |= (static_cast<uint64_t>(operation_flag_mask)) << 48;
    w0 |= (static_cast<uint64_t>(byte_mask))           << 40;
    w0 |= (static_cast<uint64_t>(dest))                << 16;
    w0 |= (static_cast<uint64_t>(bank_mask16))         << 0;

    w1 = column_mask;

    if ((operation_type % 0x80u) < 3) {
        w2 |= (static_cast<uint64_t>(row_number[0])) << 0;
        w2 |= (static_cast<uint64_t>(row_number[1])) << 16;
        w2 |= (static_cast<uint64_t>(row_number[2])) << 32;
        w2 |= (static_cast<uint64_t>(row_number[3])) << 48;
    } else {
        w2 |= (static_cast<uint64_t>(row_number[0])) << 0;
    }

    w3 |= (static_cast<uint64_t>(mat_mask32)) << 0;
    w3 |= (static_cast<uint64_t>(array_mask32)) << 32;

    volatile uint64_t *ca = commandAddress;
    ca[1] = w1;
    ca[2] = w2;
    ca[3] = w3;
    __sync_synchronize();
    ca[0] = w0;
}