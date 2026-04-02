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
system.l2_xbar = L2XBar()
system.membus.frontend_latency = 1
system.membus.forward_latency = 1
system.membus.response_latency = 1
system.membus.snoop_response_latency = 1
system.membus.width = 64 
system.l2_xbar.width = 64

# --- L1 Cache (single core) ---
system.cpu.icache = Cache(
    size='64kB',   # Orin A78AE
    assoc=2,
    tag_latency=5,
    data_latency=5,
    response_latency=5,
    mshrs=4,
    tgts_per_mshr=20
)

system.cpu.dcache = Cache(
    size='64kB',   # Orin A78AE
    assoc=2,
    tag_latency=5,
    data_latency=5,
    response_latency=5,
    mshrs=8,
    tgts_per_mshr=20,
    write_buffers=8
)

# --- L2 Cache (per-core) ---
system.l2_cache = Cache(
    size='256kB',  # single core L2
    assoc=8,      
    tag_latency=10,
    data_latency=10,
    response_latency=8,
    mshrs=20,
    tgts_per_mshr=12
)

# --- L3 Cache (per-cluster) ---
system.l3_cache = Cache(
    size='2MB',    # single cluster slice
    assoc=16,
    tag_latency=20,
    data_latency=20,
    response_latency=10,
    mshrs=20,
    tgts_per_mshr=12
)

# L1 -> L2XBar
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.icache.mem_side = system.l2_xbar.cpu_side_ports
system.cpu.dcache.mem_side = system.l2_xbar.cpu_side_ports

# L2XBar -> L2
system.l2_cache.cpu_side = system.l2_xbar.mem_side_ports

# L2 -> L3 (Direct)
system.l2_cache.mem_side = system.l3_cache.cpu_side

# L3 -> MemBus
system.l3_cache.mem_side = system.membus.cpu_side_ports

system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.mem_sched_policy = "frfcfs"
system.mem_ctrl.min_writes_per_switch = 1
system.mem_ctrl.min_reads_per_switch = 1
system.mem_ctrl.static_frontend_latency = "1ns"
system.mem_ctrl.static_backend_latency = "1ns"
system.mem_ctrl.command_window = "1ns"
system.mem_ctrl.port = system.membus.mem_side_ports

system.mem_ctrl.dram = NVMInterface()
system.mem_ctrl.dram.write_buffer_size = 2048
system.mem_ctrl.dram.read_buffer_size = 2048
system.mem_ctrl.dram.max_pending_writes = 64
system.mem_ctrl.dram.max_pending_reads = 64
system.mem_ctrl.dram.burst_length = 64
system.mem_ctrl.dram.tCK = "1ps"
system.mem_ctrl.dram.tREAD = "680ps"
system.mem_ctrl.dram.tWRITE = "1ps"
system.mem_ctrl.dram.tSEND = "1ps"
system.mem_ctrl.dram.tBURST = "1ps"
system.mem_ctrl.dram.tWTR = "1ps"
system.mem_ctrl.dram.tRTW = "1ps"
system.mem_ctrl.dram.tCS = "1ps"
system.mem_ctrl.dram.device_rowbuffer_size = "512B"
system.mem_ctrl.dram.device_size = "1GiB"
system.mem_ctrl.dram.device_bus_width = 64
system.mem_ctrl.dram.devices_per_rank = 16
system.mem_ctrl.dram.ranks_per_channel = 16
system.mem_ctrl.dram.banks_per_rank = 16
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
binary = "./tests/test-progs/ivf_matching/bin/ivf_sim"
SimpleOpts.add_option("binary", nargs="?", default=binary)
SimpleOpts.add_option("--max-steps", type=int, default=None)

EndAddress = 0x21000000
SimpleOpts.add_option("--EndAddress", type=str, default="0x21000000")
args = SimpleOpts.parse_args()

process = Process()
cmd = [args.binary]
if args.max_steps is not None:
    cmd.extend(["--max-steps", str(args.max_steps)])
print(cmd)
process.cmd = cmd
print(process.cmd)
system.workload = SEWorkload.init_compatible(args.binary)
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
