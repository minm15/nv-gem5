#include "cim_api.hpp"
#include <gem5/m5ops.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <iomanip>

// ====== 記憶體映射 (需與 Python 設定一致) ======
static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x18000000;
static constexpr uintptr_t CMD_BASE  = 0x20000000;

// ====== Geometry (需與 CimHandler 參數一致) ======
static constexpr unsigned BANK_BITS  = 4;
static constexpr unsigned MAT_BITS   = 4;
static constexpr unsigned ARRAY_BITS = 4;
static constexpr unsigned ROW_BITS   = 9;  // 512 rows
static constexpr unsigned COL_BITS   = 6;  // 64B per row

static constexpr size_t ROW_BYTES = (1ull << COL_BITS);          // 64 Bytes
static constexpr size_t NUM_ROWS_PER_ARRAY = (1ull << ROW_BITS); // 512 Rows

// 測試目標
static constexpr uint16_t TARGET_BANK  = 0;
static constexpr uint16_t TARGET_MAT   = 0;
static constexpr uint16_t TARGET_ARRAY = 0;

// ====== 輔助函式：驗證資料 ======
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
        // 簡單的錯誤傾印 (只印出第一個錯誤)
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
    // 1. 設定記憶體指標
    auto* rw  = reinterpret_cast<volatile uint64_t*>(DATA_BASE);
    auto* tmp = reinterpret_cast<volatile uint64_t*>(TEMP_BASE);
    auto* cmd = reinterpret_cast<volatile uint64_t*>(CMD_BASE);

    // 初始化 CIM Module
    CimModule cim(rw, tmp, cmd);
    cim.setGeometry(BANK_BITS, MAT_BITS, ARRAY_BITS, ROW_BITS, COL_BITS);

    // 準備 Buffer
    std::vector<uint8_t> pattern_buf(ROW_BYTES);

    std::cout << "=== Phase 1: Initialization (Fill 512 Rows) ===\n";
    // 對 Bank 0 / Mat 0 / Array 0 寫入資料塞滿 512 個 row
    for (int r = 0; r < NUM_ROWS_PER_ARRAY; ++r) {
        // 簡單填入資料: 每個 byte 都填入 row index
        std::memset(pattern_buf.data(), r % 256, ROW_BYTES);
        
        cim.copy_to_cim(TARGET_BANK, TARGET_MAT, TARGET_ARRAY, 
                        static_cast<uint16_t>(r), 
                        pattern_buf.data(), 
                        ROW_BYTES);
    }
    std::cout << "   -> Initialization Done.\n";


    std::cout << "=== Phase 2: Compute OR(2i, 2i+1) -> Temp(i) ===\n";
    // 對 {2i, 2i+1} 的 row 做 OR，結果存在 Temp 的 row i
    // 總共產生 256 個結果 (512 / 2)
    int num_results = NUM_ROWS_PER_ARRAY / 2; // 256
    
    for (int i = 0; i < num_results; ++i) {
        std::vector<uint16_t> src_rows = {
            static_cast<uint16_t>(2 * i), 
            static_cast<uint16_t>(2 * i + 1)
        };
        
        // 呼叫 OR API
        cim.OR(src_rows, 0xFF, 
               CimModule::Mask::bank(TARGET_BANK), 
               CimModule::Mask::colsAll(), 
               static_cast<uint16_t>(i), // Dest: Temp Row i
               CimModule::Mask::mat(TARGET_MAT), 
               CimModule::Mask::array(TARGET_ARRAY));
    }
    std::cout << "   -> Computation Commands Sent.\n";


    // ==========================================
    // Phase 2.5: 在 CPU 端計算 Golden Data (預期結果)
    // ==========================================
    std::cout << "=== Phase 2.5: Calculating CPU Golden Data ===\n";
    std::vector<uint8_t> expected_buf(num_results * ROW_BYTES);
    
    for (int i = 0; i < num_results; ++i) {
        // 模擬原本的 Init Pattern: Data[Row R] = memset(R % 256)
        uint8_t val_src1 = (2 * i) % 256;
        uint8_t val_src2 = (2 * i + 1) % 256;
        
        // 預期結果 = Src1 OR Src2
        uint8_t val_expected = val_src1 | val_src2;
        
        // 填入 Expected Buffer 對應的 Row
        uint8_t* ptr = expected_buf.data() + (i * ROW_BYTES);
        std::memset(ptr, val_expected, ROW_BYTES);
    }
    std::cout << "   -> Golden Data Ready.\n";


    // 準備 CPU 端的接收 Buffer
    std::vector<uint8_t> result_buf_A(num_results * ROW_BYTES);
    std::vector<uint8_t> result_buf_B(num_results * ROW_BYTES);


    // ==========================================
    // 3a. 策略 A: 逐行讀取 (Row-by-Row)
    // ==========================================
    std::cout << "=== Phase 3a: Benchmarking Strategy A (Row-by-Row) ===\n";
    
    // Reset stats: 清除初始化和運算階段的 counters
    m5_reset_stats(0, 0);
    // Work begin: 標記工作項目 0 (Row Read)
    m5_work_begin(0, 0);

    for (int i = 0; i < num_results; ++i) {
        uint8_t* ptr = result_buf_A.data() + (i * ROW_BYTES);
        
        // 呼叫原本的 API，一次讀一個 Row
        cim.copy_temp_to_cpu(ptr, 
                             TARGET_BANK, TARGET_MAT, TARGET_ARRAY, 
                             static_cast<uint16_t>(i), 
                             ROW_BYTES);
    }

    // Work end: 結束工作項目 0
    m5_work_end(0, 0);
    // Dump stats: 將這段期間的數據寫入 stats.txt
    m5_dump_stats(0, 0);
    
    std::cout << "   -> Strategy A Done. Stats dumped.\n";
    
    // 驗證 A 的結果
    if (!verify_result("Strategy A", result_buf_A, expected_buf, num_results)) {
        return 1; // 發生錯誤提早結束
    }


    // ==========================================
    // 3b. 策略 B: 區塊讀取 (Block Read)
    // ==========================================
    std::cout << "=== Phase 3b: Benchmarking Strategy B (Block Read) ===\n";
    
    // Reset stats: 清除策略 A 造成的 Cache 狀態影響
    m5_reset_stats(0, 0);
    // Work begin: 標記工作項目 1 (Block Read)
    m5_work_begin(1, 0);

    // 呼叫新增的 Block API，一次讀取 num_results (256) 個 Rows
    cim.copy_temp_block_to_cpu(result_buf_B.data(), 
                               TARGET_BANK, TARGET_MAT, TARGET_ARRAY, 
                               0,              // Start Row
                               num_results);   // Num Rows

    // Work end: 結束工作項目 1
    m5_work_end(1, 0);
    // Dump stats: 將這段期間的數據寫入 stats.txt
    m5_dump_stats(0, 0);

    std::cout << "   -> Strategy B Done. Stats dumped.\n";
    
    // 驗證 B 的結果
    if (!verify_result("Strategy B", result_buf_B, expected_buf, num_results)) {
        return 1;
    }
    
    // 雙重確認 A 與 B 是否一致 (理論上通過 Golden Check 就一定一致)
    if (std::memcmp(result_buf_A.data(), result_buf_B.data(), result_buf_A.size()) == 0) {
        std::cout << "\n[SUCCESS] Both strategies returned correct and identical data.\n";
    }

    return 0;
}