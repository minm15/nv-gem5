#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "msim_json.hpp"

namespace msim {

struct BinMeta {
    std::string file;
    std::string dtype;
    std::vector<size_t> shape;
    bool c_order = true;
    size_t bytes = 0;
};

std::string read_text_file(const std::string& path);
size_t file_size_bytes(const std::string& path);

JsonValue load_json_file(const std::string& path);

BinMeta read_bin_meta(const JsonValue& root, const std::string& bin_key);

std::vector<size_t> shape_from_json_array(const JsonValue& arr);

size_t numel_from_shape(const std::vector<size_t>& shape);

size_t dtype_size_bytes(const std::string& dtype);

template <typename T>
std::vector<T> read_bin_as(const std::string& path, size_t count);

template <typename T>
void read_bin_into(const std::string& path, T* dst, size_t count);

std::vector<uint8_t> read_bin_bytes(const std::string& path, size_t bytes);

} // namespace msim

#include "bin_reader.tpp"