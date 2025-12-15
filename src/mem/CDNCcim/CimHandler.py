from m5.params import *
from m5.SimObject import SimObject


class CimHandler(SimObject):
    type = "CimHandler"
    cxx_header = "mem/CDNCcim/cim_handler.hh"
    cxx_class = "gem5::memory::CimHandler"

    read_write_address = Param.Addr(
        Addr(0x10000000), "Physical address of the CIM read/write region"
    )
    result_temporary_buffer_address = Param.Addr(
        Addr(0x14000000),
        "Physical address of the CIM temporary buffer region",
    )
    command_write_address = Param.Addr(
        Addr(0x18000000), "Physical address of the CIM command region"
    )

    num_column_bits = Param.UInt8(6, "Row size in bytes = 2^num_column_bits")
    num_bank_bits = Param.UInt8(5, "Number of banks = 2^num_bank_bits")
    num_row_bits = Param.UInt8(15, "Number of rows per bank = 2^num_row_bits")
    num_mat_bits = Param.UInt8(4, "Number of mats per bank = 2^num_mat_bits")
    num_array_bits = Param.UInt8(4, "Number of arrays per mat = 2^num_array_bits")

    operations_init_latency = VectorParam.Latency(
        [
            "10ns",
            "10ns",
            "10ns",
            "10ns",
            "10ns",
        ],
        "Initial time for each operation to start performing",
    )
    operations_on_word_latency = VectorParam.Latency(
        [
            "10ns",
            "10ns",
            "10ns",
            "10ns",
            "10ns",
        ],
        "Time used for each 8-byte word to be read and processed",
    )

    cim_operation_handler = Param.CimOperationInterface(
        "Operations are defined in this interface object",
    )