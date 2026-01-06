#include "query_loader.hpp"

#include "file_utils.hpp"
#include "msim_json.hpp"

#include <stdexcept>

namespace msim {

static bool has_key_obj(const JsonObject& obj, const std::string& k)
{
    return obj.find(k) != obj.end();
}

std::string QueryLoader::join_path(const std::string& a, const std::string& b)
{
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

template <typename T>
void QueryLoader::validate_file_size_or_throw(const std::string& path, size_t expect_bytes)
{
    const uint64_t got = file_size_bytes(path);
    if (got != static_cast<uint64_t>(expect_bytes)) {
        throw std::runtime_error("file bytes mismatch: " + path);
    }
}

bool QueryLoader::load_from_folder(const std::string& query_dir)
{
    query_dir_ = query_dir;
    query_json_path_ = join_path(query_dir, "query.json");

    if (file_size_bytes(query_json_path_) == 0) {
        throw std::runtime_error("query.json not found: " + query_json_path_);
    }

    root_ = load_json_file(query_json_path_);
    if (!root_.is_object()) {
        throw std::runtime_error("query.json root must be object");
    }
    const JsonObject& obj = root_.as_object();

    // bins
    const BinMeta m_desc   = read_bin_meta(root_, "query_desc_q4");
    const BinMeta m_pose   = read_bin_meta(root_, "query_step_pose_grid");
    const BinMeta m_off    = read_bin_meta(root_, "query_step_offsets");
    const BinMeta m_cnt    = read_bin_meta(root_, "query_step_counts");

    if (m_desc.dtype != "uint8" || m_desc.shape.size() != 2 || m_desc.shape[1] != 64) {
        throw std::runtime_error("query_desc_q4 must be (Q,64) uint8");
    }
    if (m_pose.dtype != "uint8" || m_pose.shape.size() != 2 || m_pose.shape[1] != 2) {
        throw std::runtime_error("query_step_pose_grid must be (S,2) uint8");
    }
    if (m_off.dtype != "uint32" || m_off.shape.size() != 1) {
        throw std::runtime_error("query_step_offsets must be (S+1) uint32");
    }
    if (m_cnt.dtype != "uint16" || m_cnt.shape.size() != 1) {
        throw std::runtime_error("query_step_counts must be (S) uint16");
    }

    total_rows_ = m_desc.shape[0];
    steps_      = m_pose.shape[0];

    // optional bins
    has_relodo_i_ = false;
    has_t_now_ = false;

    BinMeta m_rel{};
    BinMeta m_tnow{};
    const JsonValue* bins_v = root_.try_get("bins");
    if (bins_v && bins_v->is_object()) {
        const JsonObject& bins = bins_v->as_object();
        if (bins.find("query_step_relodo_i") != bins.end()) {
            m_rel = read_bin_meta(root_, "query_step_relodo_i");
            has_relodo_i_ = true;
        }
        if (bins.find("query_step_t_now") != bins.end()) {
            m_tnow = read_bin_meta(root_, "query_step_t_now");
            has_t_now_ = true;
        }
    }

    // load files
    const std::string p_desc = join_path(query_dir, m_desc.file);
    const std::string p_pose = join_path(query_dir, m_pose.file);
    const std::string p_off  = join_path(query_dir, m_off.file);
    const std::string p_cnt  = join_path(query_dir, m_cnt.file);

    validate_file_size_or_throw<uint8_t>(p_desc, static_cast<size_t>(m_desc.bytes));
    validate_file_size_or_throw<uint8_t>(p_pose, static_cast<size_t>(m_pose.bytes));
    validate_file_size_or_throw<uint32_t>(p_off,  static_cast<size_t>(m_off.bytes));
    validate_file_size_or_throw<uint16_t>(p_cnt,  static_cast<size_t>(m_cnt.bytes));

    query_desc_q4_  = read_bin_as<uint8_t>(p_desc, numel_from_shape(m_desc.shape));
    step_pose_grid_ = read_bin_as<uint8_t>(p_pose, numel_from_shape(m_pose.shape));
    step_offsets_   = read_bin_as<uint32_t>(p_off,  numel_from_shape(m_off.shape));
    step_counts_    = read_bin_as<uint16_t>(p_cnt,  numel_from_shape(m_cnt.shape));

    if (step_offsets_.size() != steps_ + 1) {
        throw std::runtime_error("query_step_offsets size mismatch");
    }

    if (has_relodo_i_) {
        if (m_rel.dtype != "int32" || m_rel.shape.size() != 1 || m_rel.shape[0] != steps_) {
            throw std::runtime_error("query_step_relodo_i must be (S) int32");
        }
        const std::string p_rel = join_path(query_dir, m_rel.file);
        validate_file_size_or_throw<int32_t>(p_rel, static_cast<size_t>(m_rel.bytes));
        step_relodo_i_ = read_bin_as<int32_t>(p_rel, numel_from_shape(m_rel.shape));
    } else {
        step_relodo_i_.clear();
    }

    if (has_t_now_) {
        if (m_tnow.dtype != "float64" || m_tnow.shape.size() != 1 || m_tnow.shape[0] != steps_) {
            throw std::runtime_error("query_step_t_now must be (S) float64");
        }
        const std::string p_t = join_path(query_dir, m_tnow.file);
        validate_file_size_or_throw<double>(p_t, static_cast<size_t>(m_tnow.bytes));
        step_t_now_ = read_bin_as<double>(p_t, numel_from_shape(m_tnow.shape));
    } else {
        step_t_now_.clear();
    }

    return true;
}

QueryStepView QueryLoader::step(size_t s) const
{
    if (s >= steps_) throw std::runtime_error("QueryLoader::step out of range");

    QueryStepView v;
    v.gx = step_pose_grid_.at(s * 2 + 0);
    v.gy = step_pose_grid_.at(s * 2 + 1);

    const uint32_t b = step_offsets_.at(s);
    const uint32_t e = step_offsets_.at(s + 1);
    if (e < b) throw std::runtime_error("query_step_offsets invalid");

    v.m = static_cast<size_t>(e - b);
    v.desc_ptr = query_desc_q4_.data() + static_cast<size_t>(b) * 64;

    if (has_relodo_i_) v.relodo_i = step_relodo_i_.at(s);
    if (has_t_now_)    v.t_now    = step_t_now_.at(s);

    v.has_optional_meta = has_relodo_i_ || has_t_now_;
    return v;
}

} // namespace msim