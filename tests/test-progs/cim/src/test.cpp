#include "cim_api.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <gem5/m5ops.h> 

static constexpr uintptr_t DATA_BASE = 0x10000000;
static constexpr uintptr_t TEMP_BASE = 0x18000000;
static constexpr uintptr_t CMD_BASE  = 0x20000000;

int main() {
    std::cout << "--- CIM ROI Measurement: Full Array OR ---" << std::endl;

    auto* rw  = reinterpret_cast<volatile uint64_t*>(DATA_BASE);
    auto* tmp = reinterpret_cast<volatile uint64_t*>(TEMP_BASE);
    auto* cmd = reinterpret_cast<volatile uint64_t*>(CMD_BASE);
    CimModule cim(rw, tmp, cmd);

    cim.setGeometry(4, 4, 4, 10, 6); 
    const size_t row_size = 64; 
    const int num_rows = 512;

    std::cout << "Step 1: Preparing data (Outside ROI)..." << std::endl;
    std::vector<uint8_t> initial_row(row_size, 0x00);
    cim.copy_to_cim(0, 0, 0, 0, initial_row.data(), row_size);

    for (int i = 1; i < num_rows; ++i) {
        uint8_t pattern = static_cast<uint8_t>(i % 256);
        if (pattern == 0) pattern = 0xFF;
        std::vector<uint8_t> data_row(row_size, pattern);
        cim.copy_to_cim(0, 0, 0, static_cast<uint16_t>(i), data_row.data(), row_size);
    }

    std::cout << ">>> ROI BEGIN: 511 OR operations <<<" << std::endl;
    
    std::vector<uint16_t> rows_to_or = {0, 0}; 

    m5_reset_stats(0, 0); 

    for (int i = 1; i < num_rows; ++i) {
        rows_to_or[1] = static_cast<uint16_t>(i); 
        
        cim.OR(
            rows_to_or,
            0xFF,
            CimModule::Mask::bank(0),
            CimModule::Mask::colsAll(),
            0,
            CimModule::Mask::mat(0),
            CimModule::Mask::array(0)
        );
    }

    m5_dump_stats(0, 0); 
    std::cout << ">>> ROI END <<<" << std::endl;

    std::vector<uint8_t> final_result(row_size, 0x00);
    cim.copy_temp_to_cpu(final_result.data(), 0, 0, 0, 0, row_size);

    std::cout << "Final Result Check: " << std::hex << (int)final_result[0] << std::dec << std::endl;
    std::cout << "DONE" << std::endl;

    return 0;
}