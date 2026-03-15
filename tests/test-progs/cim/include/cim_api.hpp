#ifndef __CIM_API__HPP__
#define __CIM_API__HPP__

#include <cinttypes>
#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#define DEFAULT_READWRITE_ADDRESS ((volatile uint64_t *const)0x10000000ul)
#define DEFAULT_TEMP_ADDRESS      ((volatile uint64_t *const)0x14000000ul)
#define DEFAULT_COMMAND_ADDRESS   ((volatile uint64_t *const)0x15000000ul)

class CimModule
{
  protected:
    static constexpr uint8_t kByteBits = 3;
    static constexpr uint64_t kAllOnes64 = 0xffffffffffffffffull;

    volatile uint64_t *const readWriteAddress;
    volatile uint64_t *tempAddress;
    volatile uint64_t *const commandWriteAddress;

    unsigned bankBits;
    unsigned matBits;
    unsigned arrayBits;
    unsigned rowBits;
    unsigned columnBits;

    struct CommandEncode
    {
      private:
        volatile uint64_t *const commandAddress;

      public:
        uint8_t operation_type;
        uint8_t operation_flag_mask;
        uint8_t byte_mask;

        uint16_t bank_mask16;
        uint32_t mat_mask32;
        uint32_t array_mask32;

        uint64_t column_mask;

        uint16_t row_number[4];
        uint16_t dest;

        CommandEncode(volatile uint64_t *const command_address);

        void print();
        void issue();
    };

    void checkGeometryReady() const;

    uint16_t normalizeBankMask16(uint64_t bank_mask) const;
    uint32_t normalizeMatMask32(uint32_t mat_mask) const;
    uint32_t normalizeArrayMask32(uint32_t array_mask) const;
    uint64_t normalizeColumnMask(uint64_t column_mask) const;

    uintptr_t calcRegionOffsetBytes(
        uint16_t bank,
        uint16_t mat,
        uint16_t array,
        uint16_t row) const;

    void generateCommand(
        uint8_t op_type,
        const std::vector<uint16_t> &rows,
        uint8_t byte_mask,
        uint64_t bank_mask,
        uint64_t column_mask,
        uint16_t dest,
        uint32_t mat_mask,
        uint32_t array_mask);

  public:
    struct Mask
    {
        static uint64_t bank(uint16_t b);
        static uint64_t banksAll(unsigned bank_bits);

        static uint32_t mat(uint16_t m);
        static uint32_t matsAll(unsigned mat_bits);

        static uint32_t array(uint16_t a);
        static uint32_t arraysAll(unsigned array_bits);

        static uint64_t colsAll();
    };

    CimModule(
        volatile uint64_t *const read_write_address = DEFAULT_READWRITE_ADDRESS,
        volatile uint64_t *const temp_address = DEFAULT_TEMP_ADDRESS,
        volatile uint64_t *const command_address = DEFAULT_COMMAND_ADDRESS);

    void setGeometry(
        unsigned num_bank_bits,
        unsigned num_mat_bits,
        unsigned num_array_bits,
        unsigned num_row_bits,
        unsigned num_column_bits);

    void AND(
        const std::vector<uint16_t> &rows,
        uint8_t byte_mask = 0xffu,
        uint64_t bank_mask = kAllOnes64,
        uint64_t column_mask = kAllOnes64,
        uint16_t dest = 0x0000u,
        uint32_t mat_mask = 0xffffffffu,
        uint32_t array_mask = 0xffffffffu);

    void OR(
        const std::vector<uint16_t> &rows,
        uint8_t byte_mask = 0xffu,
        uint64_t bank_mask = kAllOnes64,
        uint64_t column_mask = kAllOnes64,
        uint16_t dest = 0x0000u,
        uint32_t mat_mask = 0xffffffffu,
        uint32_t array_mask = 0xffffffffu);

    void XOR(
        const std::vector<uint16_t> &rows,
        uint8_t byte_mask = 0xffu,
        uint64_t bank_mask = kAllOnes64,
        uint64_t column_mask = kAllOnes64,
        uint16_t dest = 0x0000u,
        uint32_t mat_mask = 0xffffffffu,
        uint32_t array_mask = 0xffffffffu);

    void COPY(
        const uint16_t &dest,
        const uint16_t &src,
        const uint8_t &rotate_left = 0,
        const uint8_t &byte_mask = 0xffu,
        const uint64_t &bank_mask = kAllOnes64,
        const uint64_t &column_mask = kAllOnes64,
        const uint32_t &mat_mask = 0xffffffffu,
        const uint32_t &array_mask = 0xffffffffu);

    void NOT_COND(
        const uint16_t &dest,
        const uint16_t &src,
        const bool &always_NOT,
        const bool &NOT_if_zero,
        const uint8_t &byte_mask = 0xffu,
        const uint64_t &bank_mask = kAllOnes64,
        const uint64_t &column_mask = kAllOnes64,
        const uint32_t &mat_mask = 0xffffffffu,
        const uint32_t &array_mask = 0xffffffffu);

    void copy_to_cim(
        uint16_t bank,
        uint16_t mat,
        uint16_t array,
        uint16_t row,
        void *cpu_array,
        size_t size_in_byte);

    void copy_to_cpu(
        void *cpu_array,
        uint16_t bank,
        uint16_t mat,
        uint16_t array,
        uint16_t row,
        size_t size_in_byte);

    void copy_temp_to_cpu(
        void *cpu_array,
        uint16_t bank,
        uint16_t mat,
        uint16_t array,
        uint16_t row,
        size_t size_in_byte);

    void copy_temp_block_to_cpu(
        void *cpu_array,
        uint16_t bank,
        uint16_t mat,
        uint16_t array,
        uint16_t start_row,
        size_t num_rows);
};

#endif // __CIM_API__HPP__
