#include "bin_reader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace msim {

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to open text file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

JsonValue load_json_file(const std::string& path) {
    const std::string s = read_text_file(path);
    return JsonParser::parse(s);
}

std::vector<size_t> shape_from_json_array(const JsonValue& arr) {
    if (!arr.is_array()) throw std::runtime_error("shape must be a JSON array");
    const auto& a = arr.as_array();
    std::vector<size_t> out;
    out.reserve(a.size());
    for (const auto& v : a) {
        if (!v.is_number()) throw std::runtime_error("shape elements must be numbers");
        double d = v.get_number();
        if (d < 0) throw std::runtime_error("shape element must be non-negative");
        out.push_back(static_cast<size_t>(d));
    }
    return out;
}

size_t numel_from_shape(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

size_t dtype_size_bytes(const std::string& dtype) {
    if (dtype == "uint8") return 1;
    if (dtype == "int8") return 1;
    if (dtype == "uint16") return 2;
    if (dtype == "int16") return 2;
    if (dtype == "uint32") return 4;
    if (dtype == "int32") return 4;
    if (dtype == "float32") return 4;
    if (dtype == "float64") return 8;
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

BinMeta read_bin_meta(const JsonValue& root, const std::string& bin_key) {
    const JsonValue& bins = root.at("bins");
    const JsonValue& entry = bins.at(bin_key);

    if (entry.is_null()) {
        throw std::runtime_error("bins entry is null for key: " + bin_key);
    }
    if (!entry.is_object()) {
        throw std::runtime_error("bins entry must be an object for key: " + bin_key);
    }

    BinMeta m;
    m.file = entry.at("file").get_string();
    m.dtype = entry.at("dtype").get_string();
    m.shape = shape_from_json_array(entry.at("shape"));

    if (const JsonValue* co = entry.try_get("c_order")) {
        if (!co->is_bool()) throw std::runtime_error("c_order must be bool");
        m.c_order = co->get_bool();
    } else {
        m.c_order = true;
    }

    if (const JsonValue* b = entry.try_get("bytes")) {
        if (!b->is_number()) throw std::runtime_error("bytes must be number");
        double d = b->get_number();
        if (d < 0) throw std::runtime_error("bytes must be non-negative");
        m.bytes = static_cast<size_t>(d);
    } else {
        const size_t n = numel_from_shape(m.shape);
        m.bytes = n * dtype_size_bytes(m.dtype);
    }

    return m;
}

std::vector<uint8_t> read_bin_bytes(const std::string& path, size_t bytes) {
    std::vector<uint8_t> out(bytes);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open bin file: " + path);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("Failed to read expected bytes from: " + path);
    return out;
}

} // namespace msim