#include "query_loader.hpp"

#include <stdexcept>

namespace msim {

std::string QueryLoader::join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

template <typename T>
void QueryLoader::validate_file_size_or_throw(const std::string& path, size_t expect_bytes) {
    const size_t fs = file_size_bytes(path);
    if (fs != expect_bytes) {
        throw std::runtime_error("File size mismatch: expected " + std::to_string(expect_bytes) +
                                 ", got " + std::to_string(fs) + " for " + path);
    }
}

bool QueryLoader::load_from_folder(const std::string& query_dir) {
    query_dir_ = query_dir;
    query_json_path_ = join_path(query_dir_, "query.json");
    root_ = load_json_file(query_json_path_);

    if (!root_.has("bins")) throw std::runtime_error("query.json missing 'bins'");

    // query_desc_q4
    const BinMeta qd = read_bin_meta(root_, "query_desc_q4");
    if (qd.dtype != "uint8") throw std::runtime_error("query_desc_q4 dtype must be uint8");
    if (qd.shape.size() != 2 || qd.shape[1] != 64) throw std::runtime_error("query_desc_q4 shape must be (Q,64)");
    total_rows_ = qd.shape[0];

    const std::string qd_path = join_path(query_dir_, qd.file);
    validate_file_size_or_throw<uint8_t>(qd_path, qd.bytes);
    query_desc_q4_ = read_bin_as<uint8_t>(qd_path, numel_from_shape(qd.shape));

    // pose grid
    const BinMeta pg = read_bin_meta(root_, "query_step_pose_grid");
    if (pg.dtype != "uint8") throw std::runtime_error("query_step_pose_grid dtype must be uint8");
    if (pg.shape.size() != 2 || pg.shape[1] != 2) throw std::runtime_error("query_step_pose_grid shape must be (S,2)");
    steps_ = pg.shape[0];

    const std::string pg_path = join_path(query_dir_, pg.file);
    validate_file_size_or_throw<uint8_t>(pg_path, pg.bytes);
    step_pose_grid_ = read_bin_as<uint8_t>(pg_path, numel_from_shape(pg.shape));

    // offsets
    const BinMeta off = read_bin_meta(root_, "query_step_offsets");
    if (off.dtype != "uint32") throw std::runtime_error("query_step_offsets dtype must be uint32");
    if (off.shape.size() != 1 || off.shape[0] != steps_ + 1) throw std::runtime_error("query_step_offsets shape must be (S+1)");

    const std::string off_path = join_path(query_dir_, off.file);
    validate_file_size_or_throw<uint32_t>(off_path, off.bytes);
    step_offsets_ = read_bin_as<uint32_t>(off_path, numel_from_shape(off.shape));

    // counts
    const BinMeta cnt = read_bin_meta(root_, "query_step_counts");
    if (cnt.dtype != "uint16") throw std::runtime_error("query_step_counts dtype must be uint16");
    if (cnt.shape.size() != 1 || cnt.shape[0] != steps_) throw std::runtime_error("query_step_counts shape must be (S)");

    const std::string cnt_path = join_path(query_dir_, cnt.file);
    validate_file_size_or_throw<uint16_t>(cnt_path, cnt.bytes);
    step_counts_ = read_bin_as<uint16_t>(cnt_path, numel_from_shape(cnt.shape));

    // Optional: relodo_i
    has_relodo_i_ = false;
    if (const JsonValue* bins = root_.try_get("bins")) {
        if (bins->is_object()) {
            if (bins->as_object().find("query_step_relodo_i") != bins->as_object().end()) {
                const JsonValue& e = bins->at("query_step_relodo_i");
                if (!e.is_null()) {
                    const BinMeta rid = read_bin_meta(root_, "query_step_relodo_i");
                    if (rid.dtype != "int32") throw std::runtime_error("query_step_relodo_i dtype must be int32");
                    if (rid.shape.size() != 1 || rid.shape[0] != steps_) throw std::runtime_error("query_step_relodo_i shape must be (S)");
                    const std::string rid_path = join_path(query_dir_, rid.file);
                    validate_file_size_or_throw<int32_t>(rid_path, rid.bytes);
                    step_relodo_i_ = read_bin_as<int32_t>(rid_path, numel_from_shape(rid.shape));
                    has_relodo_i_ = true;
                }
            }
        }
    }

    // Optional: t_now
    has_t_now_ = false;
    if (const JsonValue* bins = root_.try_get("bins")) {
        if (bins->is_object()) {
            if (bins->as_object().find("query_step_t_now") != bins->as_object().end()) {
                const JsonValue& e = bins->at("query_step_t_now");
                if (!e.is_null()) {
                    const BinMeta tn = read_bin_meta(root_, "query_step_t_now");
                    if (tn.dtype != "float64") throw std::runtime_error("query_step_t_now dtype must be float64");
                    if (tn.shape.size() != 1 || tn.shape[0] != steps_) throw std::runtime_error("query_step_t_now shape must be (S)");
                    const std::string tn_path = join_path(query_dir_, tn.file);
                    validate_file_size_or_throw<double>(tn_path, tn.bytes);
                    step_t_now_ = read_bin_as<double>(tn_path, numel_from_shape(tn.shape));
                    has_t_now_ = true;
                }
            }
        }
    }

    // Basic consistency checks
    if (step_offsets_.front() != 0) throw std::runtime_error("query_step_offsets[0] must be 0");
    if (step_offsets_.back() != static_cast<uint32_t>(total_rows_)) {
        throw std::runtime_error("query_step_offsets[-1] must equal total query rows");
    }
    for (size_t s = 0; s < steps_; ++s) {
        const uint32_t a = step_offsets_[s];
        const uint32_t b = step_offsets_[s + 1];
        if (b < a) throw std::runtime_error("query_step_offsets must be non-decreasing");
        const uint32_t m = b - a;
        if (m != static_cast<uint32_t>(step_counts_[s])) {
            throw std::runtime_error("query_step_counts mismatch at step " + std::to_string(s));
        }
    }

    return true;
}

QueryStepView QueryLoader::step(size_t s) const {
    if (s >= steps_) throw std::runtime_error("Query step out of range");

    QueryStepView v;
    v.gx = step_pose_grid_[2 * s + 0];
    v.gy = step_pose_grid_[2 * s + 1];

    const uint32_t a = step_offsets_[s];
    const uint32_t b = step_offsets_[s + 1];
    const size_t m = static_cast<size_t>(b - a);

    v.m = m;
    v.desc_ptr = query_desc_q4_.data() + static_cast<size_t>(a) * 64;

    v.has_optional_meta = false;
    if (has_relodo_i_) {
        v.relodo_i = step_relodo_i_[s];
        v.has_optional_meta = true;
    }
    if (has_t_now_) {
        v.t_now = step_t_now_[s];
        v.has_optional_meta = true;
    }
    return v;
}

} // namespace msim