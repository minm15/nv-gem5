import m5
from m5.objects import *

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

# Memory (64GB LPDDR5 256-bit)
system.mem_ctrl = MemCtrl()
system.mem_ctrl.port = system.membus.mem_side_ports

from gem5.components.memory.dram_interfaces.lpddr5 import LPDDR5_6400_1x16_BG_BL32
dram_intf = LPDDR5_6400_1x16_BG_BL32()
dram_intf.device_bus_width = 64
dram_intf.devices_per_rank = 1
dram_intf.device_size = "32GiB"
dram_intf.ranks_per_channel = 1
dram_intf.banks_per_rank = 16
system.mem_ctrl.dram = dram_intf

system.mem_ctrl.dram.range = system.mem_ranges[0]

system.cpu.createInterruptController()
# ----------------------------------------------------------------
# 9. Workload
# ----------------------------------------------------------------
binary = "./tests/test-progs/simulation_kdtree/bin/pf_kernel_roi"
map_path = "./tests/test-progs/simulation_kdtree/pf_export/map.bin"
frames_path = "./tests/test-progs/simulation_kdtree/pf_export/frames.bin"  
mcl_path = "/home/kaiii/tmp/pfivf_roi.bin"
hello_binary = "/home/kaiii/NVMSimulation/simulator/gem5/tests/test-progs/hello/bin/arm/linux/hello"


process = Process()
process.cmd = [binary, mcl_path]
system.cpu.workload = process
system.cpu.createThreads()
system.workload = SEWorkload.init_compatible(binary)

# ----------------------------------------------------------------
# 10. Run
# ----------------------------------------------------------------
root = Root(full_system=False, system=system)
m5.instantiate()

print(f"Beginning simulation with 64GB LPDDR5 (256-bit) and fixed L3 Cache!")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")