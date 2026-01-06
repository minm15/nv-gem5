#include "file_utils.hpp"
#include <fstream>
#include <stdexcept>

namespace msim {

size_t file_size_bytes(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    const std::streamsize n = f.tellg();
    if (n < 0) throw std::runtime_error("tellg failed: " + path);
    return static_cast<size_t>(n);
}

static std::vector<uint8_t> read_file_raw(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    f.seekg(0, std::ios::end);
    const std::streamsize n = f.tellg();
    if (n < 0) throw std::runtime_error("tellg failed: " + path);
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> buf(static_cast<size_t>(n));
    if (!buf.empty()) {
        f.read(reinterpret_cast<char*>(buf.data()), n);
        if (!f) throw std::runtime_error("Short read: " + path);
    }
    return buf;
}

std::vector<uint8_t> read_file_u8(const std::string& path)
{
    return read_file_raw(path);
}

std::vector<uint32_t> read_file_u32(const std::string& path)
{
    const std::vector<uint8_t> raw = read_file_raw(path);
    if (raw.size() % 4 != 0) throw std::runtime_error("u32 file size not multiple of 4: " + path);

    const size_t n = raw.size() / 4;
    std::vector<uint32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = 0;
        v |= static_cast<uint32_t>(raw[i * 4 + 0]) << 0;
        v |= static_cast<uint32_t>(raw[i * 4 + 1]) << 8;
        v |= static_cast<uint32_t>(raw[i * 4 + 2]) << 16;
        v |= static_cast<uint32_t>(raw[i * 4 + 3]) << 24;
        out[i] = v;
    }
    return out;
}

std::vector<uint16_t> read_file_u16(const std::string& path)
{
    const std::vector<uint8_t> raw = read_file_raw(path);
    if (raw.size() % 2 != 0) throw std::runtime_error("u16 file size not multiple of 2: " + path);

    const size_t n = raw.size() / 2;
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint16_t v = 0;
        v |= static_cast<uint16_t>(raw[i * 2 + 0]) << 0;
        v |= static_cast<uint16_t>(raw[i * 2 + 1]) << 8;
        out[i] = v;
    }
    return out;
}

} // namespace msim