#pragma once
#include <fstream>
#include <stdexcept>

namespace msim {

template <typename T>
std::vector<T> read_bin_as(const std::string& path, size_t count) {
    std::vector<T> out(count);
    read_bin_into(path, out.data(), count);
    return out;
}

template <typename T>
void read_bin_into(const std::string& path, T* dst, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open bin file: " + path);
    const size_t bytes = sizeof(T) * count;
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("Failed to read expected bytes from: " + path);
}

} // namespace msim