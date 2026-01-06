#pragma once
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace msim {

struct JsonValue;

using JsonObject = std::unordered_map<std::string, JsonValue>;
using JsonArray  = std::vector<JsonValue>;

struct JsonValue {
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>;
    Storage v;

    JsonValue() : v(nullptr) {}
    JsonValue(std::nullptr_t) : v(nullptr) {}
    JsonValue(bool b) : v(b) {}
    JsonValue(double d) : v(d) {}
    JsonValue(std::string s) : v(std::move(s)) {}
    JsonValue(JsonArray a) : v(std::move(a)) {}
    JsonValue(JsonObject o) : v(std::move(o)) {}

    bool is_null()   const { return std::holds_alternative<std::nullptr_t>(v); }
    bool is_bool()   const { return std::holds_alternative<bool>(v); }
    bool is_number() const { return std::holds_alternative<double>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_array()  const { return std::holds_alternative<JsonArray>(v); }
    bool is_object() const { return std::holds_alternative<JsonObject>(v); }

    const bool&        as_bool()   const;
    const double&      as_number() const;
    const std::string& as_string() const;
    const JsonArray&   as_array()  const;
    const JsonObject&  as_object() const;

    bool        get_bool()   const;
    double      get_number() const;
    std::string get_string() const;

    const JsonValue& at(const std::string& key) const;
    const JsonValue& at(size_t idx) const;

    bool has(const std::string& key) const;

    const JsonValue* try_get(const std::string& key) const;
    const JsonValue* try_get(size_t idx) const;
};

class JsonParser {
public:
    static JsonValue parse(const std::string& s);

private:
    explicit JsonParser(const std::string& s);

    JsonValue parse_value();
    JsonValue parse_object();
    JsonValue parse_array();
    JsonValue parse_string();
    JsonValue parse_number();
    JsonValue parse_true();
    JsonValue parse_false();
    JsonValue parse_null();

    void skip_ws();
    char peek() const;
    char get();
    void expect(char c);

    const std::string& src_;
    size_t i_;
};

} // namespace msim