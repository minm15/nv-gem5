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
system.cpu = TimingSimpleCPU()

# Bus
system.membus = SystemXBar()
system.membus.frontend_latency = 10
system.membus.forward_latency = 10
system.membus.response_latency = 10
system.membus.snoop_response_latency = 10

class L1ICache(Cache):
    size = "32kB"
    assoc = 2
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 4
    tgts_per_mshr = 20

class L1DCache(Cache):
    size = "32kB"
    assoc = 2
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 8
    tgts_per_mshr = 20
    write_buffers = 8

# L1 caches
system.cpu.icache = L1ICache()
system.cpu.dcache = L1DCache()

# CPU <-> L1
system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side

# L1 <-> membus
system.cpu.icache.mem_side = system.membus.cpu_side_ports
system.cpu.dcache.mem_side = system.membus.cpu_side_ports

system.system_port = system.membus.cpu_side_ports

# ----------------------------
# Memory controller + LPDDR5
# ----------------------------
system.mem_ctrl = MemCtrl()
system.mem_ctrl.port = system.membus.mem_side_ports

# Prefer gem5 stdlib LPDDR5 timing interface (avoids manual timing/proxy pitfalls).
# gem5 provides LPDDR5 interfaces like LPDDR5_6400_1x16_BG_BL32 / _8B_BL32 etc. 
try:
    from gem5.components.memory.dram_interfaces.lpddr5 import LPDDR5_6400_1x16_BG_BL32
    dram_if = LPDDR5_6400_1x16_BG_BL32()
    # 256-bit channel = 16 devices * x16 = 256-bit
    dram_if.device_bus_width = 16
    dram_if.devices_per_rank = 16
    # 64GiB total. The LPDDR5 8Gbit die implies ~1GiB per device; 16 devices/rank => ~16GiB/rank.
    # Use 4 ranks to reach ~64GiB.
    dram_if.device_size = "1GiB"
    dram_if.ranks_per_channel = 4
    system.mem_ctrl.dram = dram_if
except Exception as e:
    print("[WARN] LPDDR5 stdlib interface not available, fallback to generic DRAMInterface:", e)
    system.mem_ctrl.dram = DRAMInterface()
    d = system.mem_ctrl.dram
    # Minimal capacity/bandwidth shape (timing left mostly default; you can refine later)
    d.device_bus_width = 16
    d.devices_per_rank = 16
    d.device_size = "1GiB"
    d.ranks_per_channel = 4

system.mem_ctrl.dram.range = system.mem_ranges[0]

# ----------------------------
# IRQs
# ----------------------------
system.cpu.createInterruptController()

# ----------------------------
# Workload
# ----------------------------
binary = "./tests/test-progs/lab/bin/hello64-static"
SimpleOpts.add_option("binary", nargs="?", default=binary)

process = Process()
process.cmd = [binary]
system.workload = SEWorkload.init_compatible(binary)
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()

# ----------------------------
# Your memory map (keep same addresses)
# ----------------------------
# NVM array 64MiB: [0x10000000, 0x14000000)
process.map(vaddr=Addr(0x10000000), paddr=Addr(0x10000000),
            size=0x04000000, cacheable=True)

# temp buffer 64MiB: [0x14000000, 0x18000000)
# NOTE: cacheable=False => bypass caches (often the reason ROI still slow)
process.map(vaddr=Addr(0x14000000), paddr=Addr(0x14000000),
            size=0x04000000, cacheable=False)

# cmd/mmio 0x18000000
process.map(vaddr=Addr(0x18000000), paddr=Addr(0x18000000),
            size=0x1000, cacheable=False)

print("Beginning simulation!")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")