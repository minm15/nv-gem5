#ifdef CDNCcimFlag

#ifndef __CIM_HANDLER_HH__
#define __CIM_HANDLER_HH__

#include "base/statistics.hh"
#include "debug/CIMDBG.hh"
#include "mem/abstract_mem.hh"
#include "mem/mem_ctrl.hh"
#include "params/CimHandler.hh"
#include "sim/cur_tick.hh"
#include "sim/sim_object.hh"

namespace gem5
{
namespace memory
{

class CimHandler : public SimObject
{
  private:
    enum class OperationType : uint8_t
    {
        AND = 0,
        OR,
        XOR,
        COPY,
        NOT_COND,
        short_AND = 0x80,
        short_OR,
        short_XOR,
        short_COPY,
        short_NOT_COND,
    };

    struct CommandDecode
    {
        uint8_t operation_type { 0xff };
        uint8_t operation_flag_mask { 0 };
        uint8_t byte_mask { 0xffu };

        uint16_t bank_mask16 { 0xffffu };
        uint32_t mat_mask32 { 0xffffffffu };
        uint32_t array_mask32 { 0xffffffffu };

        uint64_t bank_mask { 0xfffffffffffffffful };
        uint64_t column_mask { 0xfffffffffffffffful };

        uint16_t row_number[4] { 0xffffu, 0xffffu, 0xffffu, 0xffffu };
        uint16_t dest { 0 };

        void print();
    };

    const uint64_t CommandSize { 4 * 8 };
    const uint8_t numOperationTypes { 5 };
    const uint8_t byteBits { 3 };

    Addr readWriteAddress;
    Addr resultTemporaryBufferAddress;
    Addr commandWriteAddress;

    uint8_t numColumnBits;
    uint8_t numBankBits;
    uint8_t numRowBits;
    uint8_t numMatBits;
    uint8_t numArrayBits;

    std::vector<Tick> operationsInitLatency;
    std::vector<Tick> operationsOnWordLatency;
    std::vector<Tick> operationsOnHTreeLatency;
    std::vector<Tick> operationsOnPredecoderLatency;
    std::vector<Tick> operationsOnRowDecoderLatency;
    std::vector<Tick> operationsOnColumnLatency;
    

    Tick *unitReleaseTime;

    // statistics::Scalar cimWorkTicksSum;
    // statistics::Scalar cimWorkTicksUnion;
    // statistics::Scalar cimInitChunkCount;
    // statistics::Scalar cimWordChunkCount;
    // statistics::Scalar cimOpCmdCount;
    // Tick unionBusyUntil = 0;

    struct CimStats : public statistics::Group
    {
        CimStats(statistics::Group *parent);
        statistics::Scalar orOpCount;
    } stats;

    void decodeCommandWords(
        uint64_t w0,
        uint64_t w1,
        uint64_t w2,
        uint64_t w3,
        CommandDecode &command) const;
    void cimExecuteCommand(AbstractMemory *abstract_mem, CommandDecode &command);
    void cimUpdateLatencyTable(bool init, uint8_t operation, size_t bank);

    uint8_t *addressTranslator(
        AbstractMemory *abstract_mem,
        Addr startAddress,
        uint16_t row,
        uint8_t bank,
        uint8_t mat,
        uint8_t array,
        uint8_t column);

    uint64_t regionSizeBytes() const;

    std::vector<Tick> opInitLat;
    Tick cimReadyAt = 0;

  public:
    CimOperationInterface *cimOperationHandler;

    inline bool isCimAddressRenge(const Addr &addr) const
    {
        const Addr rw_end = readWriteAddress + regionSizeBytes();
        const Addr tmp_end = resultTemporaryBufferAddress + regionSizeBytes();
        const Addr cmd_end = commandWriteAddress + CommandSize;

        const bool in_rw = (addr >= readWriteAddress) && (addr < rw_end);
        const bool in_tmp = (addr >= resultTemporaryBufferAddress) && (addr < tmp_end);
        const bool in_cmd = (addr >= commandWriteAddress) && (addr < cmd_end);

        return in_rw || in_tmp || in_cmd;
    }

    inline bool isTmpAddressRange(const Addr &addr) const
    {
        const Addr tmp_end = resultTemporaryBufferAddress + regionSizeBytes();
        return (addr >= resultTemporaryBufferAddress) && (addr < tmp_end);
    }

    inline bool isCimReadWriteRegion(const Addr &addr) const
    {
        const Addr end = readWriteAddress + regionSizeBytes();
        return (addr >= readWriteAddress) && (addr < end);
    }

    inline bool isCimBufferRegion(const Addr &addr) const
    {
        const Addr end = resultTemporaryBufferAddress + regionSizeBytes();
        return (addr >= resultTemporaryBufferAddress) && (addr < end);
    }

    inline bool isCimCommandRegion(const Addr &addr) const
    {
        return (addr >= commandWriteAddress)
               && (addr < (commandWriteAddress + CommandSize));
    }

    CimHandler(const CimHandlerParams &_p);
    ~CimHandler();

    void issueCommand(
        AbstractMemory *abstract_mem,
        Addr cmd_addr,
        uint64_t w0,
        uint64_t w1,
        uint64_t w2,
        uint64_t w3);
    void cimFetchCommand(AbstractMemory *abstract_mem, PacketPtr pkt, uint8_t *host_addr);

    Tick getCimLatency(const Addr &addr);

    void regStats() override;

    Addr getReadWriteAddress() const { return readWriteAddress; }
    Addr getResultTemporaryBufferAddress() const { return resultTemporaryBufferAddress; }
    Addr getCommandWriteAddress() const { return commandWriteAddress; }
    uint64_t getRegionSizeBytes() const { return regionSizeBytes(); }
};

} // namespace memory
} // namespace gem5

#endif // __CIM_HANDLER_HH__
#endif // CDNCcimFlag
