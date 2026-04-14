import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEM5_ROOT = os.path.dirname(SCRIPT_DIR)
if GEM5_ROOT not in sys.path:
    sys.path.insert(0, GEM5_ROOT)

import m5
from m5.objects import *
from configs.common import SimpleOpts
from m5.objects import Cache
from m5.objects import DerivO3CPU
from m5.objects import StridePrefetcher


# --- System ---
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '2.2GHz'
system.clk_domain.voltage_domain = VoltageDomain()

# timing
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('1GiB')]

# CPU TimingSimpleCPU
system.cpu = DerivO3CPU()

# Bus
system.membus = SystemXBar()
system.membus.frontend_latency = 1
system.membus.forward_latency = 1
system.membus.response_latency = 1
system.membus.snoop_response_latency = 1
system.membus.width = 64 

# Connect CPU ports directly to the memory bus (no cache hierarchy).
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports

system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.mem_sched_policy = "frfcfs"
system.mem_ctrl.min_writes_per_switch = 1
system.mem_ctrl.min_reads_per_switch = 1
system.mem_ctrl.static_frontend_latency = "1ns"
system.mem_ctrl.static_backend_latency = "1ns"
system.mem_ctrl.command_window = "1ns"
system.mem_ctrl.port = system.membus.mem_side_ports

system.mem_ctrl.dram = NVM_CIM_1x64()
system.mem_ctrl.dram.range = system.mem_ranges[0]

# CIM handler 
try:
    system.mem_ctrl.dram.cim_handler_list = [CimHandler()]
    for cim in system.mem_ctrl.dram.cim_handler_list:
        cim.num_column_bits = 6
        cim.num_bank_bits   = 4
        cim.num_mat_bits    = 4
        cim.num_array_bits  = 4
        cim.num_row_bits    = 9

        cim.cim_operation_handler = CimOperationInterface()

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
        cim.operations_init_latency = [ "4.432ns", "4.432ns", "4.432ns", "4.432ns", "4.432ns", ]
        cim.operations_on_word_latency = [ "4.432ns", "4.432ns", "4.432ns", "4.432ns", "4.432ns", ]
        cim.operations_on_H_tree_latency = [ "257.82ps", "257.82ps", "257.82ps", "257.82ps", "257.82ps", ]
        cim.operations_on_Predecoder_latency = [ "195.65ps", "195.65ps", "195.65ps", "195.65ps", "195.65ps", ]
        cim.operations_on_Row_decoder_latency = [ "270ps", "270ps", "270ps", "270ps", "270ps", ]
        cim.operations_on_Column_latency = [ "3.709ns", "3.709ns", "3.709ns", "3.709ns", "3.709ns", ]
        
except Exception as e:
    print("WARNING: CIM not enabled or SimObjects not found:", e) 
# IRQs
system.cpu.createInterruptController()

# --- Binary ---
binary = "./tests/test-progs/ivf_matching/verify_bin/ivf_verify"
SimpleOpts.add_option("binary", nargs="?", default=binary)

EndAddress = 0x21000000
SimpleOpts.add_option("--EndAddress", type=str, default="0x21000000")

process = Process()
process.cmd = [binary]
system.workload = SEWorkload.init_compatible(binary)
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()

# NVM array (readWriteAddress) 64MiB: [0x10000000, 0x18000000)
process.map(vaddr=Addr(0x10000000), paddr=Addr(0x10000000),
            size=0x08000000, cacheable=False)

# temp buffer (resultTemporaryBufferAddress) 64MiB: [0x18000000, 0x18000000)
process.map(vaddr=Addr(0x18000000), paddr=Addr(0x18000000),
            size=0x08000000, cacheable=False)

# cmd/mmio (commandWriteAddress) at 0x18000000
process.map(vaddr=Addr(0x20000000), paddr=Addr(0x20000000),
            size=0x1000, cacheable=False)

print("Beginning simulation!")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
