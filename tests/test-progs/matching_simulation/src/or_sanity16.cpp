#include "cim_api.hpp"
#include <gem5/m5ops.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <iomanip>

// ====== Memory map (must match the Python config) ======
static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x18000000;
static constexpr uintptr_t CMD_BASE  = 0x20000000;

// ====== Geometry (must match the CimHandler parameters) ======
static constexpr unsigned BANK_BITS  = 4;
static constexpr unsigned MAT_BITS   = 4;
static constexpr unsigned ARRAY_BITS = 4;
static constexpr unsigned ROW_BITS   = 9;  // 512 rows
static constexpr unsigned COL_BITS   = 6;  // 64B per row

static constexpr size_t ROW_BYTES = (1ull << COL_BITS);          // 64 Bytes
static constexpr size_t NUM_ROWS_PER_ARRAY = (1ull << ROW_BITS); // 512 Rows

// Test target
static constexpr uint16_t TARGET_BANK  = 0;
static constexpr uint16_t TARGET_MAT   = 0;
static constexpr uint16_t TARGET_ARRAY = 0;

// ====== Helper: verify data ======
bool verify_result(const std::string& strategy_name, 
                   const std::vector<uint8_t>& actual, 
                   const std::vector<uint8_t>& expected, 
                   size_t num_rows) 
{
    std::cout << "   -> Verifying " << strategy_name << "... ";
    if (std::memcmp(actual.data(), expected.data(), actual.size()) == 0) {
        std::cout << "[PASS]\n";
        return true;
    } else {
        std::cout << "[FAIL]\n";
        // Simple error dump (print only the first mismatch)
        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < ROW_BYTES; ++c) {
                size_t idx = r * ROW_BYTES + c;
                if (actual[idx] != expected[idx]) {
                    std::cout << "      Mismatch at Row " << r << " Byte " << c 
                              << ": Expected 0x" << std::hex << (int)expected[idx]
                              << ", Got 0x" << (int)actual[idx] << std::dec << "\n";
                    return false; // Found first error, return
                }
            }
        }
        return false;
    }
}

int main() {
    // 1. Set memory pointers
    auto* rw  = reinterpret_cast<volatile uint64_t*>(DATA_BASE);
    auto* tmp = reinterpret_cast<volatile uint64_t*>(TEMP_BASE);
    auto* cmd = reinterpret_cast<volatile uint64_t*>(CMD_BASE);

    // Initialize the CIM module
    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, MAT_BITS, ARRAY_BITS, ROW_BITS, COL_BITS);

    // Prepare buffers
    std::vector<uint8_t> pattern_buf(ROW_BYTES);

    std::cout << "=== Phase 1: Initialization (Fill 512 Rows) ===\n";
    // Fill all 512 rows in Bank 0 / Mat 0 / Array 0
    for (int r = 0; r < NUM_ROWS_PER_ARRAY; ++r) {
        // Simple pattern: fill each byte with the row index
        std::memset(pattern_buf.data(), r % 256, ROW_BYTES);
        
        cim.copy_to_cim(TARGET_BANK, TARGET_MAT, TARGET_ARRAY, 
                        static_cast<uint16_t>(r), 
                        pattern_buf.data(), 
                        ROW_BYTES);
    }
    std::cout << "   -> Initialization Done.\n";


    std::cout << "=== Phase 2: Compute OR(2i, 2i+1) -> Temp(i) ===\n";
    // OR rows {2i, 2i+1} and store the result in Temp row i
    // Produce 256 results total (512 / 2)
    int num_results = NUM_ROWS_PER_ARRAY / 2; // 256
    
    for (int i = 0; i < num_results; ++i) {
        std::vector<uint16_t> src_rows = {
            static_cast<uint16_t>(2 * i), 
            static_cast<uint16_t>(2 * i + 1)
        };
        
        // Call the OR API
        cim.OR(src_rows, 0xFF, 
               CimModule::Mask::bank(TARGET_BANK), 
               CimModule::Mask::colsAll(), 
               static_cast<uint16_t>(i), // Dest: Temp Row i
               CimModule::Mask::mat(TARGET_MAT), 
               CimModule::Mask::array(TARGET_ARRAY));
    }
    std::cout << "   -> Computation Commands Sent.\n";


    // ==========================================
    // Phase 2.5: compute golden data on the CPU
    // ==========================================
    std::cout << "=== Phase 2.5: Calculating CPU Golden Data ===\n";
    std::vector<uint8_t> expected_buf(num_results * ROW_BYTES);
    
    for (int i = 0; i < num_results; ++i) {
        // Recreate the original init pattern: Data[Row R] = memset(R % 256)
        uint8_t val_src1 = (2 * i) % 256;
        uint8_t val_src2 = (2 * i + 1) % 256;
        
        // Expected result = Src1 OR Src2
        uint8_t val_expected = val_src1 | val_src2;
        
        // Fill the corresponding row in the expected buffer
        uint8_t* ptr = expected_buf.data() + (i * ROW_BYTES);
        std::memset(ptr, val_expected, ROW_BYTES);
    }
    std::cout << "   -> Golden Data Ready.\n";


    // Prepare CPU-side receive buffers
    std::vector<uint8_t> result_buf_A(num_results * ROW_BYTES);
    std::vector<uint8_t> result_buf_B(num_results * ROW_BYTES);


    // ==========================================
    // 3a. Strategy A: row-by-row reads
    // ==========================================
    std::cout << "=== Phase 3a: Benchmarking Strategy A (Row-by-Row) ===\n";
    
    // Reset stats: clear counters from initialization and compute phases
    m5_reset_stats(0, 0);
    // Work begin: mark work item 0 (row read)
    m5_work_begin(0, 0);

    for (int i = 0; i < num_results; ++i) {
        uint8_t* ptr = result_buf_A.data() + (i * ROW_BYTES);
        
        // Call the original API to read one row at a time
        cim.copy_temp_to_cpu(ptr, 
                             TARGET_BANK, TARGET_MAT, TARGET_ARRAY, 
                             static_cast<uint16_t>(i), 
                             ROW_BYTES);
    }

    // Work end: finish work item 0
    m5_work_end(0, 0);
    // Dump stats: write data for this phase to stats.txt
    m5_dump_stats(0, 0);
    
    std::cout << "   -> Strategy A Done. Stats dumped.\n";
    
    // Verify Strategy A
    if (!verify_result("Strategy A", result_buf_A, expected_buf, num_results)) {
        return 1; // Exit early on error
    }


    // ==========================================
    // 3b. Strategy B: block reads
    // ==========================================
    std::cout << "=== Phase 3b: Benchmarking Strategy B (Block Read) ===\n";
    
    // Reset stats: clear cache effects from Strategy A
    m5_reset_stats(0, 0);
    // Work begin: mark work item 1 (block read)
    m5_work_begin(1, 0);

    // Call the block API to read num_results (256) rows at once
    cim.copy_temp_block_to_cpu(result_buf_B.data(), 
                               TARGET_BANK, TARGET_MAT, TARGET_ARRAY, 
                               0,              // Start Row
                               num_results);   // Num Rows

    // Work end: finish work item 1
    m5_work_end(1, 0);
    // Dump stats: write data for this phase to stats.txt
    m5_dump_stats(0, 0);

    std::cout << "   -> Strategy B Done. Stats dumped.\n";
    
    // Verify Strategy B
    if (!verify_result("Strategy B", result_buf_B, expected_buf, num_results)) {
        return 1;
    }
    
    // Double-check that A and B match
    if (std::memcmp(result_buf_A.data(), result_buf_B.data(), result_buf_A.size()) == 0) {
        std::cout << "\n[SUCCESS] Both strategies returned correct and identical data.\n";
    }

    return 0;
}