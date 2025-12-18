#include <gem5/m5ops.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// ===== problem size (same as your CIM test) =====
static constexpr int NUM_MAP_LOCAL = 512;   // per array
static constexpr int DIMS          = 64;    // 64 dims
static constexpr int VALUE_BITS    = 4;     // 4-bit

static constexpr unsigned ARRAY_BITS = 4;   // 16 arrays
static constexpr uint16_t TEST_BANK = 0;
static constexpr uint16_t TEST_MAT  = 0;

static constexpr uint16_t ARRAYS = (1u << ARRAY_BITS); // 16
static constexpr uint32_t TOTAL_MAPS = uint32_t(ARRAYS) * uint32_t(NUM_MAP_LOCAL); // 8192
static constexpr uint32_t PACKED_BYTES_PER_MAP = uint32_t(DIMS / 2); // 64 nibbles -> 32 bytes

// deterministic 4-bit map value generator (same as before)
static inline uint8_t map_val(uint16_t bank, uint16_t mat, uint16_t array,
                              int local_map, int d)
{
    uint32_t x = 0x9E3779B9u;
    x ^= (static_cast<uint32_t>(bank)  * 0xA341316Cu);
    x ^= (static_cast<uint32_t>(mat)   * 0xC8013EA4u);
    x ^= (static_cast<uint32_t>(array) * 0xAD90777Du);
    x ^= (static_cast<uint32_t>(local_map) * 0x7E95761Eu);
    x ^= (static_cast<uint32_t>(d)     * 0xF39CC060u);

    x ^= (x >> 16);
    x *= 0x85EBCA6Bu;
    x ^= (x >> 13);
    x *= 0xC2B2AE35u;
    x ^= (x >> 16);

    return static_cast<uint8_t>(x & 0xFu);
}

// ---- 4-bit packed layout helpers ----
// packed_maps size = TOTAL_MAPS * 32 bytes
// map_id = array*512 + local_map
// dim d stored in byte (d/2), low nibble if even d, high nibble if odd d
static inline void set_nibble(uint8_t* base32B, int d, uint8_t v4)
{
    const int idx = d >> 1;                 // 0..31
    const bool hi = (d & 1) != 0;
    const uint8_t v = static_cast<uint8_t>(v4 & 0xFu);

    uint8_t b = base32B[idx];
    if (!hi) {
        b = static_cast<uint8_t>((b & 0xF0u) | v);
    } else {
        b = static_cast<uint8_t>((b & 0x0Fu) | (v << 4));
    }
    base32B[idx] = b;
}

static inline uint8_t get_nibble(const uint8_t* base32B, int d)
{
    const int idx = d >> 1;
    const bool hi = (d & 1) != 0;
    const uint8_t b = base32B[idx];
    return static_cast<uint8_t>(hi ? ((b >> 4) & 0xFu) : (b & 0xFu));
}

static inline uint8_t golden_score(uint16_t bank, uint16_t mat, uint16_t array,
                                   int local_map,
                                   const std::array<uint8_t, DIMS> &q)
{
    uint32_t s = 0;
    for (int d = 0; d < DIMS; ++d) {
        const uint8_t mv = map_val(bank, mat, array, local_map, d);
        s += ((mv & 0xFu) != (q[d] & 0xFu));
    }
    return static_cast<uint8_t>(s);
}

int main()
{
    // =========================
    // ROI OUT: build maps in CPU memory
    // =========================
    std::vector<uint8_t> packed_maps;
    packed_maps.resize(size_t(TOTAL_MAPS) * size_t(PACKED_BYTES_PER_MAP), 0);

    std::cout << "Building packed maps in CPU memory (ROI OUT) ..." << std::endl;

    for (uint16_t array = 0; array < ARRAYS; ++array) {
        for (int m = 0; m < NUM_MAP_LOCAL; ++m) {
            if (m%32 == 0) std::cout << "Building packed maps in array: " << array << " map: " << m << std::endl;
            const uint32_t map_id = uint32_t(array) * uint32_t(NUM_MAP_LOCAL) + uint32_t(m);
            uint8_t* dst32B = &packed_maps[size_t(map_id) * size_t(PACKED_BYTES_PER_MAP)];

            // fill 64 dims (4-bit each) into 32B
            for (int d = 0; d < DIMS; ++d) {
                const uint8_t v = map_val(TEST_BANK, TEST_MAT, array, m, d);
                set_nibble(dst32B, d, v);
            }
        }
    }

    // =========================
    // ROI OUT: build 4 queries
    // =========================
    std::array<std::array<uint8_t, DIMS>, 4> queries{};

    // q0 = map(bank0,mat0,array0,local0)
    for (int d = 0; d < DIMS; ++d) {
        queries[0][d] = map_val(TEST_BANK, TEST_MAT, 0, 0, d);
    }
    // q1 = q0 but flip dim0
    queries[1] = queries[0];
    queries[1][0] = static_cast<uint8_t>((queries[1][0] + 1u) & 0xFu);

    // q2/q3 random fixed seed
    std::mt19937 rng(20251216);
    std::uniform_int_distribution<int> dist(0, 15);
    for (int d = 0; d < DIMS; ++d) {
        queries[2][d] = static_cast<uint8_t>(dist(rng));
        queries[3][d] = static_cast<uint8_t>(dist(rng));
    }

    // store ROI result: 4 queries * 8192 maps
    std::vector<uint8_t> scores;
    scores.resize(size_t(4) * size_t(TOTAL_MAPS), 0);

    // =========================
    // ROI: matching only
    // =========================
    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);

    for (int qi = 0; qi < 4; ++qi) {
        std::cout << "processing on query: " << qi << std::endl;
        const size_t out_base = size_t(qi) * size_t(TOTAL_MAPS);

        for (uint32_t map_id = 0; map_id < TOTAL_MAPS; ++map_id) {
            const uint8_t* map32B = &packed_maps[size_t(map_id) * size_t(PACKED_BYTES_PER_MAP)];

            uint32_t s = 0;
            for (int d = 0; d < DIMS; ++d) {
                const uint8_t mv = get_nibble(map32B, d);
                s += ((mv & 0xFu) != (queries[qi][d] & 0xFu));
            }
            scores[out_base + size_t(map_id)] = static_cast<uint8_t>(s);
        }
    }

    m5_work_end(0, 0);
    m5_dump_stats(0, 0);

    // =========================
    // ROI OUT: golden compare (exclude verification)
    // =========================
    // std::cout << "Golden compare (ROI OUT) ..." << std::endl;

    // for (int qi = 0; qi < 4; ++qi) {
    //     const size_t out_base = size_t(qi) * size_t(TOTAL_MAPS);

    //     for (uint16_t array = 0; array < ARRAYS; ++array) {
    //         for (int m = 0; m < NUM_MAP_LOCAL; ++m) {
    //             const uint32_t map_id = uint32_t(array) * uint32_t(NUM_MAP_LOCAL) + uint32_t(m);

    //             const uint8_t got = scores[out_base + size_t(map_id)];
    //             const uint8_t exp = golden_score(TEST_BANK, TEST_MAT, array, m, queries[qi]);

    //             if (got != exp) {
    //                 std::cout << "[FAIL]\n"
    //                           << "  qi=" << qi
    //                           << " bank=" << TEST_BANK
    //                           << " mat=" << TEST_MAT
    //                           << " array=" << array
    //                           << " local_map=" << m
    //                           << " map_id=" << map_id
    //                           << " got_score=" << int(got)
    //                           << " exp_score=" << int(exp)
    //                           << "\n";
    //                 return 1;
    //             }
    //         }
    //     }
    // }

    // std::cout << "\nALL PASS ✅\n"
    //           << "comparisons = " << (uint64_t(TOTAL_MAPS) * 4ull) << "\n";
    return 0;
}