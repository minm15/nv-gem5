#include "msim_json.hpp"

namespace msim {

static std::runtime_error type_error(const char* want) {
    return std::runtime_error(std::string("JSON type error: expected ") + want);
}

const bool& JsonValue::as_bool() const {
    if (!is_bool()) throw type_error("bool");
    return std::get<bool>(v);
}

const double& JsonValue::as_number() const {
    if (!is_number()) throw type_error("number");
    return std::get<double>(v);
}

const std::string& JsonValue::as_string() const {
    if (!is_string()) throw type_error("string");
    return std::get<std::string>(v);
}

const JsonArray& JsonValue::as_array() const {
    if (!is_array()) throw type_error("array");
    return std::get<JsonArray>(v);
}

const JsonObject& JsonValue::as_object() const {
    if (!is_object()) throw type_error("object");
    return std::get<JsonObject>(v);
}

bool JsonValue::get_bool() const {
    return as_bool();
}

double JsonValue::get_number() const {
    return as_number();
}

std::string JsonValue::get_string() const {
    return as_string();
}

const JsonValue& JsonValue::at(const std::string& key) const {
    const auto& obj = as_object();
    auto it = obj.find(key);
    if (it == obj.end()) throw std::runtime_error("JSON missing key: " + key);
    return it->second;
}

const JsonValue& JsonValue::at(size_t idx) const {
    const auto& arr = as_array();
    if (idx >= arr.size()) throw std::runtime_error("JSON array index out of range");
    return arr[idx];
}

bool JsonValue::has(const std::string& key) const {
    if (!is_object()) return false;
    const auto& obj = std::get<JsonObject>(v);
    return obj.find(key) != obj.end();
}

const JsonValue* JsonValue::try_get(const std::string& key) const {
    if (!is_object()) return nullptr;
    const auto& obj = std::get<JsonObject>(v);
    auto it = obj.find(key);
    return (it == obj.end()) ? nullptr : &it->second;
}

const JsonValue* JsonValue::try_get(size_t idx) const {
    if (!is_array()) return nullptr;
    const auto& arr = std::get<JsonArray>(v);
    if (idx >= arr.size()) return nullptr;
    return &arr[idx];
}

JsonParser::JsonParser(const std::string& s) : src_(s), i_(0) {}

JsonValue JsonParser::parse(const std::string& s) {
    JsonParser p(s);
    p.skip_ws();
    JsonValue root = p.parse_value();
    p.skip_ws();
    if (p.i_ != p.src_.size()) {
        throw std::runtime_error("JSON parse error: trailing characters");
    }
    return root;
}

void JsonParser::skip_ws() {
    while (i_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[i_]))) {
        ++i_;
    }
}

char JsonParser::peek() const {
    if (i_ >= src_.size()) return '\0';
    return src_[i_];
}

char JsonParser::get() {
    if (i_ >= src_.size()) throw std::runtime_error("JSON parse error: unexpected end");
    return src_[i_++];
}

void JsonParser::expect(char c) {
    char got = get();
    if (got != c) {
        throw std::runtime_error(std::string("JSON parse error: expected '") + c + "', got '" + got + "'");
    }
}

JsonValue JsonParser::parse_value() {
    skip_ws();
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string();
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    if (c == 't') return parse_true();
    if (c == 'f') return parse_false();
    if (c == 'n') return parse_null();
    throw std::runtime_error(std::string("JSON parse error: unexpected char '") + c + "'");
}

JsonValue JsonParser::parse_object() {
    expect('{');
    skip_ws();
    JsonObject obj;

    if (peek() == '}') {
        get();
        return JsonValue(std::move(obj));
    }

    while (true) {
        skip_ws();
        if (peek() != '"') throw std::runtime_error("JSON parse error: object key must be a string");
        std::string key = parse_string().as_string();

        skip_ws();
        expect(':');
        skip_ws();

        JsonValue val = parse_value();
        obj.emplace(std::move(key), std::move(val));

        skip_ws();
        char c = get();
        if (c == '}') break;
        if (c != ',') throw std::runtime_error("JSON parse error: expected ',' or '}' in object");
    }
    return JsonValue(std::move(obj));
}

JsonValue JsonParser::parse_array() {
    expect('[');
    skip_ws();
    JsonArray arr;

    if (peek() == ']') {
        get();
        return JsonValue(std::move(arr));
    }

    while (true) {
        skip_ws();
        arr.push_back(parse_value());
        skip_ws();
        char c = get();
        if (c == ']') break;
        if (c != ',') throw std::runtime_error("JSON parse error: expected ',' or ']' in array");
    }
    return JsonValue(std::move(arr));
}

JsonValue JsonParser::parse_string() {
    expect('"');
    std::string out;
    while (true) {
        char c = get();
        if (c == '"') break;
        if (c == '\\') {
            char e = get();
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    // Minimal \uXXXX handling: parse and keep ASCII when possible.
                    // For non-ASCII, we replace with '?' to keep parser small.
                    unsigned code = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = get();
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(10 + (h - 'a'));
                        else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(10 + (h - 'A'));
                        else throw std::runtime_error("JSON parse error: invalid \\u escape");
                    }
                    if (code <= 0x7F) out.push_back(static_cast<char>(code));
                    else out.push_back('?');
                    break;
                }
                default:
                    throw std::runtime_error("JSON parse error: invalid escape sequence");
            }
        } else {
            out.push_back(c);
        }
    }
    return JsonValue(std::move(out));
}

JsonValue JsonParser::parse_number() {
    size_t start = i_;
    if (peek() == '-') get();
    if (peek() == '0') {
        get();
    } else {
        if (!(peek() >= '1' && peek() <= '9')) throw std::runtime_error("JSON parse error: invalid number");
        while (peek() >= '0' && peek() <= '9') get();
    }
    if (peek() == '.') {
        get();
        if (!(peek() >= '0' && peek() <= '9')) throw std::runtime_error("JSON parse error: invalid fractional part");
        while (peek() >= '0' && peek() <= '9') get();
    }
    if (peek() == 'e' || peek() == 'E') {
        get();
        if (peek() == '+' || peek() == '-') get();
        if (!(peek() >= '0' && peek() <= '9')) throw std::runtime_error("JSON parse error: invalid exponent");
        while (peek() >= '0' && peek() <= '9') get();
    }

    const std::string num = src_.substr(start, i_ - start);
    char* endp = nullptr;
    double d = std::strtod(num.c_str(), &endp);
    if (endp == nullptr || *endp != '\0') throw std::runtime_error("JSON parse error: strtod failed");
    return JsonValue(d);
}

JsonValue JsonParser::parse_true() {
    if (src_.compare(i_, 4, "true") != 0) throw std::runtime_error("JSON parse error: expected true");
    i_ += 4;
    return JsonValue(true);
}

JsonValue JsonParser::parse_false() {
    if (src_.compare(i_, 5, "false") != 0) throw std::runtime_error("JSON parse error: expected false");
    i_ += 5;
    return JsonValue(false);
}

JsonValue JsonParser::parse_null() {
    if (src_.compare(i_, 4, "null") != 0) throw std::runtime_error("JSON parse error: expected null");
    i_ += 4;
    return JsonValue(nullptr);
}

} // namespace msim