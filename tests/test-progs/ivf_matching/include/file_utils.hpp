#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace msim {

std::vector<uint8_t> read_file_u8(const std::string& path);
std::vector<uint32_t> read_file_u32(const std::string& path);
std::vector<uint16_t> read_file_u16(const std::string& path);

size_t file_size_bytes(const std::string& path);

} // namespace msim