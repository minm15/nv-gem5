import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEM5_ROOT = os.path.dirname(SCRIPT_DIR)
if GEM5_ROOT not in sys.path:
    sys.path.insert(0, GEM5_ROOT)

import m5
from m5.objects import *
from configs.common import SimpleOpts

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
system.l3_xbar = L2XBar()
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
)

system.cpu.dcache = Cache(
    size='64kB',   # Orin A78AE
    assoc=2,
    tag_latency=5,
    data_latency=5,
    response_latency=5,
)

# --- L2 Cache (per-core) ---
system.l2_cache = Cache(
    size='256kB',  # single core L2
    assoc=4,      
    tag_latency=15,
    data_latency=15,
    response_latency=15,
)

# --- L3 Cache (per-cluster) ---
system.l3_cache = Cache(
    size='2MB',    # single cluster slice
    assoc=8,
    tag_latency=50,
    data_latency=50,
    response_latency=50,
)


# L1 -> L2XBar
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.icache.mem_side = system.l2_xbar.cpu_side_ports
system.cpu.dcache.mem_side = system.l2_xbar.cpu_side_ports

# L2XBar -> L2
system.l2_cache.cpu_side = system.l2_xbar.mem_side_ports

# L2 -> L3XBar -> L3
system.l2_cache.mem_side = system.l3_xbar.cpu_side_ports
system.l3_cache.cpu_side = system.l3_xbar.mem_side_ports

# L3 -> MemBus
system.l3_cache.mem_side = system.membus.cpu_side_ports
system.system_port = system.membus.cpu_side_ports

# Memory
system.mem_ctrl = MemCtrl()
system.mem_ctrl.port = system.membus.mem_side_ports

from gem5.components.memory.dram_interfaces.lpddr5 import LPDDR5_5500_1x16_8B_BL32
dram_intf = LPDDR5_5500_1x16_8B_BL32()
system.mem_ctrl.dram = dram_intf
system.mem_ctrl.dram.range = system.mem_ranges[0]

system.cpu.createInterruptController()
# ----------------------------------------------------------------
# 9. Workload
# ----------------------------------------------------------------
ivf_bin = "./tests/test-progs/ivf_nocim_matching/bin/pf_kernel_roi"
ivf_map_path = "./tests/test-progs/export_gem5/2013-01-10/map"
ivf_query_path = "./tests/test-progs/export_gem5/2013-01-10/query"
SimpleOpts.add_option("--max-steps", type=int, default=None)
args = SimpleOpts.parse_args()

process = Process()
cmd = [ivf_bin, ivf_map_path, ivf_query_path]
if args.max_steps is not None:
    cmd.extend(["--max-steps", str(args.max_steps)])
process.cmd = cmd
system.cpu.workload = process
system.cpu.createThreads()
system.workload = SEWorkload.init_compatible(ivf_bin)

# ----------------------------------------------------------------
# 10. Run
# ----------------------------------------------------------------
root = Root(full_system=False, system=system)
m5.instantiate()

print(f"Beginning simulation with 64GB LPDDR5 (256-bit) and fixed L3 Cache!")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
