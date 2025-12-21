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
    
    """
    NVSim parameter
    Timing:
    -  Read Latency = 4.432ns
    |--- H-Tree Latency = 257.820ps
    |--- Mat Latency    = 4.174ns
        |--- Predecoder Latency = 195.646ps
        |--- Subarray Latency   = 3.979ns
            |--- Row Decoder Latency = 270.473ps
            |--- Bitline Latency     = 38.481ps
            |--- Senseamp Latency    = 3.000ns
            |--- Mux Latency         = 9.758ps
            |--- Precharge Latency   = 277.810ps
            |--- Subaddon component Latency = 660.000ps
        |--- Mataddon component Latency   = 0.000ps
    """
    # H tree
    operations_on_H_tree_latency = VectorParam.Latency(
        [ "10ns", "10ns", "10ns", "10ns", "10ns", ],
        "Time used for each 8-byte word to be read and processed",
    )
    # Predecoder
    operations_on_Predecoder_latency = VectorParam.Latency(
        [ "10ns", "10ns", "10ns", "10ns", "10ns", ],
        "Time used for each 8-byte word to be read and processed",
    )
    # Row Decoder
    operations_on_Row_decoder_latency = VectorParam.Latency(
        [ "10ns", "10ns", "10ns", "10ns", "10ns", ],
        "Time used for each 8-byte word to be read and processed",
    )
    # Column: Bitline + Senseamp + Mux + Precharge + Subaddon
    operations_on_Column_latency = VectorParam.Latency(
        [ "10ns", "10ns", "10ns", "10ns", "10ns", ],
        "Time used for each 8-byte word to be read and processed",
    )
    
    
    cim_operation_handler = Param.CimOperationInterface(
        "Operations are defined in this interface object",
    )