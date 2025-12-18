#ifdef CDNCcimFlag

#include "cim_handler.hh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sim/core.hh"

namespace gem5
{
namespace memory
{

CimHandler::CimHandler(const CimHandlerParams &params)
    : SimObject(params),
      readWriteAddress(params.read_write_address),
      resultTemporaryBufferAddress(params.result_temporary_buffer_address),
      commandWriteAddress(params.command_write_address),
      numColumnBits(params.num_column_bits),
      numBankBits(params.num_bank_bits),
      numRowBits(params.num_row_bits),
      numMatBits(params.num_mat_bits),
      numArrayBits(params.num_array_bits),
      operationsInitLatency(params.operations_init_latency),
      operationsOnWordLatency(params.operations_on_word_latency),
      cimOperationHandler(params.cim_operation_handler)
{
    DPRINTF(CIMDBG, "CimHandler Constructed! this_ptr: %p\n", this);
    DPRINTF(CIMDBG, "readWrite=0x%lx temp=0x%lx cmd=0x%lx\n",
        readWriteAddress, resultTemporaryBufferAddress, commandWriteAddress);

    assert(numOperationTypes == operationsInitLatency.size());
    assert(numOperationTypes == operationsOnWordLatency.size());

    assert(numBankBits <= 16);
    assert(numMatBits <= 32);
    assert(numArrayBits <= 32);

    size_t parallel_units = (1u << numBankBits);
    unitReleaseTime = new Tick[parallel_units]();
    for (size_t i = 0; i < parallel_units; i++) {
        unitReleaseTime[i] = curTick();
    }

    opInitLat = params.operations_init_latency;
}

CimHandler::~CimHandler()
{
    if (unitReleaseTime)
        delete[] unitReleaseTime;
    unitReleaseTime = nullptr;
}

uint64_t
CimHandler::regionSizeBytes() const
{
    const uint64_t bits = static_cast<uint64_t>(numBankBits)
                        + static_cast<uint64_t>(numMatBits)
                        + static_cast<uint64_t>(numArrayBits)
                        + static_cast<uint64_t>(numRowBits)
                        + static_cast<uint64_t>(numColumnBits);

    return (1ull << bits);
}

void
CimHandler::cimFetchCommand(AbstractMemory *abstract_mem, PacketPtr pkt, uint8_t *host_addr)
{
    DPRINTF(CIMDBG, "call cimFetchCommand\n");
    DPRINTF(CIMDBG, "[%s:%s:%s] from address: 0x%lx : command: 0x%016lx \n",
        __FILE__, __func__, __LINE__, pkt->getAddr(), *(uint64_t *)host_addr);

    uint64_t *command_address = reinterpret_cast<uint64_t *>(
        abstract_mem->toHostAddr(commandWriteAddress));

    if (command_address[0] == 0ul)
        return;

    CommandDecode command;

    if (command_address[0] & (1ul << 63)) {
        panic("Short command is not supported in this protocol.\n");
    }

    if ((command_address[1] == 0ul) || (command_address[2] == 0ul) || (command_address[3] == 0ul))
        return;

    const uint64_t w0 = command_address[0];
    const uint64_t w1 = command_address[1];
    const uint64_t w2 = command_address[2];
    const uint64_t w3 = command_address[3];

    command.operation_type = (w0 >> 56) & 0xffu;
    command.operation_flag_mask = (w0 >> 48) & 0xffu;
    command.byte_mask = (w0 >> 40) & 0xffu;

    command.dest = (w0 >> 16) & 0xffffu;
    command.bank_mask16 = (w0 >> 0) & 0xffffu;
    command.bank_mask = static_cast<uint64_t>(command.bank_mask16);

    command.column_mask = w1;

    command.mat_mask32 = static_cast<uint32_t>(w3 & 0xffffffffull);
    command.array_mask32 = static_cast<uint32_t>((w3 >> 32) & 0xffffffffull);

    switch (static_cast<OperationType>(command.operation_type)) {
        case OperationType::AND:
        case OperationType::OR:
        case OperationType::XOR: {
            for (int8_t i = 0; i < 4; i++) {
                if (command.operation_flag_mask & (1u << i)) {
                    command.row_number[i] = (w2 >> (16 * i)) & 0xffffu;
                }
            }
            break;
        }
        case OperationType::COPY:
        case OperationType::NOT_COND: {
            command.row_number[0] = (w2 >> 0) & 0xffffu;
            break;
        }
        default:
            panic("\n>>> Unrecognized Command issued!! : command: %x \n",
                  command.operation_type);
            break;
    }

    command_address[0] = 0ul;
    command_address[1] = 0ul;
    command_address[2] = 0ul;
    command_address[3] = 0ul;

    DPRINTF(CIMDBG,
    "cimFetchCommand BEFORE exec @%lu unionBusyUntil=%lu bank0Ready=%lu\n",
    curTick(), unionBusyUntil, unitReleaseTime[0]);

    cimExecuteCommand(abstract_mem, command);

    DPRINTF(CIMDBG,
    "cimFetchCommand AFTER  exec @%lu unionBusyUntil=%lu bank0Ready=%lu\n",
    curTick(), unionBusyUntil, unitReleaseTime[0]);
}

void
CimHandler::cimExecuteCommand(AbstractMemory *abstract_mem, CommandDecode &command)
{
    cimOpCmdCount++;
    DPRINTF(CIMDBG, "[%s:%s:%s] cimExecuteCommand\n", __FILE__, __func__, __LINE__);
    command.print();

    const size_t numBanks = (1ull << numBankBits);
    const size_t numMats = (1ull << numMatBits);
    const size_t numArrays = (1ull << numArrayBits);
    const size_t numColumns = (1ull << (numColumnBits - byteBits));

    uint64_t row_mask = 0;
    if (numRowBits >= 16) {
        row_mask = 0xffffull;
    } else {
        row_mask = (1ull << numRowBits) - 1ull;
    }

    for (size_t bank = 0; bank < numBanks; bank++) {
        if ((command.bank_mask & (1ull << bank)) == 0ull)
            continue;

        cimUpdateLatencyTable(true, command.operation_type, bank);

        for (size_t mat = 0; mat < numMats; mat++) {
            if ((command.mat_mask32 & (1u << mat)) == 0u)
                continue;

            for (size_t array = 0; array < numArrays; array++) {
                if ((command.array_mask32 & (1u << array)) == 0u)
                    continue;

                for (size_t column = 0; column < numColumns; column++) {
                    if ((command.column_mask & (1ull << column)) == 0ull)
                        continue;

                    cimUpdateLatencyTable(false, command.operation_type, bank);

                    switch (static_cast<OperationType>(command.operation_type)) {
                        case OperationType::AND: {
                            const uint16_t dst_row = static_cast<uint16_t>(
                                static_cast<uint64_t>(command.dest) & row_mask);

                            uint8_t *dest = addressTranslator(
                                abstract_mem,
                                resultTemporaryBufferAddress,
                                dst_row,
                                static_cast<uint8_t>(bank),
                                static_cast<uint8_t>(mat),
                                static_cast<uint8_t>(array),
                                static_cast<uint8_t>(column));

                            std::vector<uint8_t *> rows;
                            for (auto &r : command.row_number) {
                                if (r == 0xffffu)
                                    continue;

                                const uint16_t src_row = static_cast<uint16_t>(
                                    static_cast<uint64_t>(r) & row_mask);

                                if (static_cast<uint64_t>(src_row) < (1ull << numRowBits)) {
                                    rows.push_back(addressTranslator(
                                        abstract_mem,
                                        readWriteAddress,
                                        src_row,
                                        static_cast<uint8_t>(bank),
                                        static_cast<uint8_t>(mat),
                                        static_cast<uint8_t>(array),
                                        static_cast<uint8_t>(column)));
                                }
                            }

                            assert(rows.size() > 1);
                            cimOperationHandler->AND(rows, dest, command.byte_mask);
                            break;
                        }

                        case OperationType::OR: {
                            const uint16_t dst_row = static_cast<uint16_t>(
                                static_cast<uint64_t>(command.dest) & row_mask);

                            uint8_t *dest = addressTranslator(
                                abstract_mem,
                                resultTemporaryBufferAddress,
                                dst_row,
                                static_cast<uint8_t>(bank),
                                static_cast<uint8_t>(mat),
                                static_cast<uint8_t>(array),
                                static_cast<uint8_t>(column));

                            std::vector<uint8_t *> rows;
                            for (auto &r : command.row_number) {
                                if (r == 0xffffu)
                                    continue;

                                const uint16_t src_row = static_cast<uint16_t>(
                                    static_cast<uint64_t>(r) & row_mask);

                                if (static_cast<uint64_t>(src_row) < (1ull << numRowBits)) {
                                    rows.push_back(addressTranslator(
                                        abstract_mem,
                                        readWriteAddress,
                                        src_row,
                                        static_cast<uint8_t>(bank),
                                        static_cast<uint8_t>(mat),
                                        static_cast<uint8_t>(array),
                                        static_cast<uint8_t>(column)));
                                }
                            }

                            assert(rows.size() > 1);
                            cimOperationHandler->OR(rows, dest, command.byte_mask);
                            break;
                        }

                        case OperationType::XOR: {
                            const uint16_t dst_row = static_cast<uint16_t>(
                                static_cast<uint64_t>(command.dest) & row_mask);

                            uint8_t *dest = addressTranslator(
                                abstract_mem,
                                resultTemporaryBufferAddress,
                                dst_row,
                                static_cast<uint8_t>(bank),
                                static_cast<uint8_t>(mat),
                                static_cast<uint8_t>(array),
                                static_cast<uint8_t>(column));

                            std::vector<uint8_t *> rows;
                            for (auto &r : command.row_number) {
                                if (r == 0xffffu)
                                    continue;

                                const uint16_t src_row = static_cast<uint16_t>(
                                    static_cast<uint64_t>(r) & row_mask);

                                if (static_cast<uint64_t>(src_row) < (1ull << numRowBits)) {
                                    rows.push_back(addressTranslator(
                                        abstract_mem,
                                        readWriteAddress,
                                        src_row,
                                        static_cast<uint8_t>(bank),
                                        static_cast<uint8_t>(mat),
                                        static_cast<uint8_t>(array),
                                        static_cast<uint8_t>(column)));
                                }
                            }

                            assert(rows.size() > 1);
                            cimOperationHandler->XOR(rows, dest, command.byte_mask);
                            break;
                        }

                        case OperationType::COPY: {
                            const bool src_is_temp = (command.row_number[0] & 0x8000u) != 0u;
                            const bool dst_is_temp = (command.dest & 0x8000u) != 0u;

                            const uint16_t src_row = static_cast<uint16_t>(
                                static_cast<uint64_t>(command.row_number[0]) & row_mask);
                            const uint16_t dst_row = static_cast<uint16_t>(
                                static_cast<uint64_t>(command.dest) & row_mask);

                            const Addr src_base = src_is_temp ? resultTemporaryBufferAddress
                                                              : readWriteAddress;
                            const Addr dst_base = dst_is_temp ? resultTemporaryBufferAddress
                                                              : readWriteAddress;

                            uint8_t *src = addressTranslator(
                                abstract_mem,
                                src_base,
                                src_row,
                                static_cast<uint8_t>(bank),
                                static_cast<uint8_t>(mat),
                                static_cast<uint8_t>(array),
                                static_cast<uint8_t>(column));

                            uint8_t *dest = addressTranslator(
                                abstract_mem,
                                dst_base,
                                dst_row,
                                static_cast<uint8_t>(bank),
                                static_cast<uint8_t>(mat),
                                static_cast<uint8_t>(array),
                                static_cast<uint8_t>(column));

                            cimOperationHandler->COPY(
                                src, dest, command.operation_flag_mask, command.byte_mask);
                            break;
                        }

                        case OperationType::NOT_COND: {
                            const bool src_is_temp = (command.row_number[0] & 0x8000u) != 0u;
                            const bool dst_is_temp = (command.dest & 0x8000u) != 0u;

                            const uint16_t src_row = static_cast<uint16_t>(
                                static_cast<uint64_t>(command.row_number[0]) & row_mask);
                            const uint16_t dst_row = static_cast<uint16_t>(
                                static_cast<uint64_t>(command.dest) & row_mask);

                            const Addr src_base = src_is_temp ? resultTemporaryBufferAddress
                                                              : readWriteAddress;
                            const Addr dst_base = dst_is_temp ? resultTemporaryBufferAddress
                                                              : readWriteAddress;

                            uint8_t *src = addressTranslator(
                                abstract_mem,
                                src_base,
                                src_row,
                                static_cast<uint8_t>(bank),
                                static_cast<uint8_t>(mat),
                                static_cast<uint8_t>(array),
                                static_cast<uint8_t>(column));

                            uint8_t *dest = addressTranslator(
                                abstract_mem,
                                dst_base,
                                dst_row,
                                static_cast<uint8_t>(bank),
                                static_cast<uint8_t>(mat),
                                static_cast<uint8_t>(array),
                                static_cast<uint8_t>(column));

                            const bool always_NOT = (command.operation_flag_mask & 0x02u) != 0u;
                            const bool NOT_if_zero = (command.operation_flag_mask & 0x01u) != 0u;

                            cimOperationHandler->NOT_COND(
                                src, dest, always_NOT, NOT_if_zero, command.byte_mask);
                            break;
                        }

                        default:
                            panic("\n>>>Unpredicted Behavior!!!\n");
                            break;
                    }
                }
            }
        }
    }
}

void
CimHandler::cimUpdateLatencyTable(bool init, uint8_t operation, size_t bank)
{
    const Tick delta = init
        ? operationsInitLatency[operation % 0x80]
        : operationsOnWordLatency[operation % 0x80];

    Tick start, end;

    if ((int64_t)unitReleaseTime[bank] - (int64_t)curTick() > 0) {
        start = unitReleaseTime[bank];
        end = unitReleaseTime[bank] + delta;
        unitReleaseTime[bank] = end;
    } else {
        start = curTick();
        end = curTick() + delta;
        unitReleaseTime[bank] = end;
    }

    cimWorkTicksSum += delta;
    if (init) cimInitChunkCount++; else cimWordChunkCount++;

    if (start >= unionBusyUntil) {
        cimWorkTicksUnion += (end - start);
        unionBusyUntil = end;
    } else if (end > unionBusyUntil) {
        cimWorkTicksUnion += (end - unionBusyUntil);
        unionBusyUntil = end;
    }
}

uint8_t *
CimHandler::addressTranslator(AbstractMemory *abstract_mem,
                             Addr startAddress,
                             uint16_t row,
                             uint8_t bank,
                             uint8_t mat,
                             uint8_t array,
                             uint8_t column)
{
    uint64_t addr = static_cast<uint64_t>(startAddress);

    addr += (static_cast<uint64_t>(bank)  << (numMatBits + numArrayBits + numRowBits + numColumnBits));
    addr += (static_cast<uint64_t>(mat)   << (numArrayBits + numRowBits + numColumnBits));
    addr += (static_cast<uint64_t>(array) << (numRowBits + numColumnBits));
    addr += (static_cast<uint64_t>(row)   << (numColumnBits));
    addr += (static_cast<uint64_t>(column) << (byteBits));

    const uint64_t end = static_cast<uint64_t>(startAddress) + regionSizeBytes();
    panic_if(addr >= end, ">>> Corruption in CimHandler::addressTranslator!!! \n");

    return abstract_mem->toHostAddr(addr);
}

Tick
CimHandler::getCimLatency(const Addr &addr)
{
    const uint64_t base = static_cast<uint64_t>(readWriteAddress);
    const uint64_t off = static_cast<uint64_t>(addr) - base;

    const uint64_t bank_shift = static_cast<uint64_t>(numMatBits)
                              + static_cast<uint64_t>(numArrayBits)
                              + static_cast<uint64_t>(numRowBits)
                              + static_cast<uint64_t>(numColumnBits);

    const uint8_t bank = static_cast<uint8_t>(
        (off >> bank_shift) & ((1ull << numBankBits) - 1ull));

    int64_t left_time =
        (static_cast<int64_t>(unitReleaseTime[bank]) - static_cast<int64_t>(curTick()));

    DPRINTF(CIMDBG, "[%s:%s:%s] left_time: %d, for address: 0x%lx\n",
        __FILE__, __func__, __LINE__, left_time, addr);
    DPRINTF(CIMDBG,
        "getCimLatency @%lu addr=0x%lx bank=%u unitRelease=%lu left=%ld\n",
        curTick(), addr, bank, unitReleaseTime[bank], left_time);

    if (left_time > 0)
        return left_time;
    return 0;
}

void
CimHandler::regStats()
{
    SimObject::regStats();
    using namespace gem5::statistics;

    cimWorkTicksSum
        .name(name() + ".cimWorkTicksSum")
        .desc("Sum of internal CIM work time (ticks) "
              "== sum of all init+on_word latencies added");

    cimWorkTicksUnion
        .name(name() + ".cimWorkTicksUnion")
        .desc("Union of CIM busy time across all banks (ticks); "
              "approx. wall-clock busy time for CIM");

    cimInitChunkCount
        .name(name() + ".cimInitChunkCount")
        .desc("# of init-latency chunks added to schedule");

    cimWordChunkCount
        .name(name() + ".cimWordChunkCount")
        .desc("# of on-word-latency chunks added to schedule");

    cimOpCmdCount
        .name(name() + ".cimOpCmdCount")
        .desc("# of CIM commands (cimExecuteCommand calls)");
}

void
CimHandler::CommandDecode::print()
{
    if (operation_type != 0x04) return;

    DPRINTFR(CIMDBG, "-------\n");
    DPRINTFR(CIMDBG, "type: %02x, flag: %02x, byte_mask: %02x\n",
        operation_type, operation_flag_mask, byte_mask);
    DPRINTFR(CIMDBG, "bank_mask16: %04x, mat_mask32: %08x, array_mask32: %08x\n",
        bank_mask16, mat_mask32, array_mask32);
    DPRINTFR(CIMDBG, "column_mask: %016lx\n", column_mask);
    DPRINTFR(CIMDBG, "row0: %04x\n", row_number[0]);
    DPRINTFR(CIMDBG, "row1: %04x\n", row_number[1]);
    DPRINTFR(CIMDBG, "row2: %04x\n", row_number[2]);
    DPRINTFR(CIMDBG, "row3: %04x\n", row_number[3]);
    DPRINTFR(CIMDBG, "dest: %04x\n", dest);
    DPRINTFR(CIMDBG, "------\n");
}

Tick
CimHandler::scheduleCmdAndGetExtraDelay(PacketPtr pkt)
{
    const uint8_t *data = pkt->getConstPtr<uint8_t>();
    uint64_t cmd0 = 0;
    std::memcpy(&cmd0, data, sizeof(uint64_t));

    const uint8_t operation_type = static_cast<uint8_t>((cmd0 >> 56) & 0xffu);

    auto opIndex = [&](uint8_t op)->int {
        switch (static_cast<OperationType>(op)) {
        case OperationType::AND:            return 0;
        case OperationType::OR:             return 1;
        case OperationType::XOR:            return 2;
        case OperationType::NOT_COND:       return 3;
        case OperationType::COPY:           return 4;
        case OperationType::short_AND:      return 0;
        case OperationType::short_OR:       return 1;
        case OperationType::short_XOR:      return 2;
        case OperationType::short_NOT_COND: return 3;
        case OperationType::short_COPY:     return 4;
        default:
            return -1;
        }
    };

    const int idx = opIndex(operation_type);

    Tick cmdLat = 0;
    if (idx >= 0 && idx < static_cast<int>(opInitLat.size())) {
        cmdLat = opInitLat[idx];
    }

    const Tick now = curTick();
    const Tick start = std::max(now, cimReadyAt);
    const Tick finish = start + cmdLat;
    cimReadyAt = finish;

    DPRINTF(CIMDBG, "scheduleCmdAndGetExtraDelay: curTick=%lu", curTick());

    return finish - now;
}

} // namespace memory
} // namespace gem5

#endif // CDNCcimFlag