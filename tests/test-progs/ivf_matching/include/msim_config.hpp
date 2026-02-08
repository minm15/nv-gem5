#pragma once
#include <cstdint>
#include <cstddef>

namespace msim {

// static constexpr const char* kMapDir   = "/home/kaiii/NVMSimulation/simulator/gem5/tests/test-progs/ivf_matching/export_gem5/2013-01-10/map";
// static constexpr const char* kQueryDir = "/home/kaiii/NVMSimulation/simulator/gem5/tests/test-progs/ivf_matching/export_gem5/2013-01-10/query";

// static constexpr const char* kMapDir   = "/home/kaiii/NVMSimulation/simulator/gem5/tests/test-progs/ivf_matching/test/map";
// static constexpr const char* kQueryDir = "/home/kaiii/NVMSimulation/simulator/gem5/tests/test-progs/ivf_matching/test/query";

static constexpr const char* kMapDir   = "/home/kaiii/nas/homes/kaiii_data/export_gem5_dir/export_gem5_512_8/2013-01-10/map";
static constexpr const char* kQueryDir = "/home/kaiii/nas/homes/kaiii_data/export_gem5_dir/export_gem5_512_8/2013-01-10/query";

// CIM geometry (512x512 bits per array => 512 rows, 512 cols => rowBytes=64).
static constexpr uint32_t kRowBits   = 9;   // 2^9  = 512 rows
static constexpr uint32_t kColBits   = 6;   // 2^6  = 64 bytes per row (512 bits)
static constexpr uint32_t kArrayBits = 4;   // 16 arrays per mat
static constexpr uint32_t kMatBits   = 4;   // 16 mats per bank
static constexpr uint32_t kBankBits  = 4;   // 16 banks total (you said you'll open 16 banks)

// Layout constants.
static constexpr size_t kRowBytes      = (1u << kColBits); // 64
static constexpr size_t kLanesPerGroup = 512;

// Descriptor: 64 dims, q4, store true+inv bitplanes => 64 * 4 * 2 = 512 rows per group.
static constexpr int kDescDims   = 64;
static constexpr int kDescBits   = 4;
static constexpr int kDescRowsPerGroup = kDescDims * kDescBits * 2; // 512

// Geometry: 8-bit x and 8-bit y, each bit has true+inv.
// To respect "max 4 rows open", we OR 4 selected rows per nibble.
// Storage per group: x(16 rows) + y(16 rows) = 32 rows.
static constexpr int kGeoRowsPerGroup = 32;
static constexpr int kGeoGroupsPerArray = 512 / kGeoRowsPerGroup; // 16
static constexpr uint8_t kGeoMaxDefault = 90;

// Probing: 10 checks (x sweep 5 + y sweep 5), center duplicated (still counts as 10 ops).
static constexpr int kGeoProbeRadius = 2;

} // namespace msim