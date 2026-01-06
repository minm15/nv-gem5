#include "map_loader.hpp"

#include "file_utils.hpp"
#include "msim_json.hpp"

#include <algorithm>
#include <stdexcept>

namespace msim {

static bool has_key_obj(const JsonObject& obj, const std::string& k)
{
    return obj.find(k) != obj.end();
}

void MapLoader::validate_or_throw(bool ok, const std::string& msg)
{
    if (!ok) throw std::runtime_error(msg);
}

uint64_t MapLoader::get_u64(const std::string& ctx, const JsonValue& j)
{
    validate_or_throw(j.is_number(), ctx + ": expected number");
    const double d = j.get_number();
    validate_or_throw(d >= 0.0, ctx + ": expected non-negative");
    return static_cast<uint64_t>(d);
}

double MapLoader::get_f64(const std::string& ctx, const JsonValue& j)
{
    validate_or_throw(j.is_number(), ctx + ": expected number");
    return j.get_number();
}

std::string MapLoader::get_str(const std::string& ctx, const JsonValue& j)
{
    validate_or_throw(j.is_string(), ctx + ": expected string");
    return j.get_string();
}

std::string MapLoader::join_path(const std::string& a, const std::string& b)
{
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

const float* MapLoader::IvfData::centroid_ptr(uint32_t lid) const
{
    if (!enabled) return nullptr;
    if (lid >= nlist) return nullptr;
    const size_t off = static_cast<size_t>(lid) * static_cast<size_t>(dim);
    if (off + dim > centroids_f32.size()) return nullptr;
    return &centroids_f32[off];
}

uint32_t MapLoader::IvfData::postings_begin(uint32_t lid) const
{
    if (!enabled) return 0;
    if (lid >= nlist) return 0;
    if (postings_offsets_u32.size() != static_cast<size_t>(nlist) + 1u) return 0;
    return postings_offsets_u32[lid];
}

uint32_t MapLoader::IvfData::postings_end(uint32_t lid) const
{
    if (!enabled) return 0;
    if (lid >= nlist) return 0;
    if (postings_offsets_u32.size() != static_cast<size_t>(nlist) + 1u) return 0;
    return postings_offsets_u32[lid + 1u];
}

uint32_t MapLoader::IvfData::postings_size(uint32_t lid) const
{
    const uint32_t b = postings_begin(lid);
    const uint32_t e = postings_end(lid);
    return (e >= b) ? (e - b) : 0;
}

bool MapLoader::load_from_folder(const std::string& map_dir)
{
    // Reset state
    n_ = 0;
    desc_bits_ = 0;
    geo_ = GeoGridParams{};
    desc_q4_u8_.clear();
    geo_grid_u8_.clear();
    ivf_ = IvfData{};

    const std::string map_json_path = join_path(map_dir, "map.json");
    validate_or_throw(file_size_bytes(map_json_path) > 0, "map.json not found: " + map_json_path);

    const JsonValue root = load_json_file(map_json_path);
    validate_or_throw(root.is_object(), "map.json root must be object");

    validate_or_throw(load_dense_map_bins_(map_dir), "failed to load dense map bins");
    (void)load_ivf_bins_if_present_(map_dir);

    return true;
}

bool MapLoader::load_dense_map_bins_(const std::string& dir)
{
    const std::string map_json_path = join_path(dir, "map.json");
    const JsonValue root = load_json_file(map_json_path);
    validate_or_throw(root.is_object(), "map.json root must be object");
    const JsonObject& obj = root.as_object();

    validate_or_throw(has_key_obj(obj, "desc_bits"), "map.json missing desc_bits");
    desc_bits_ = static_cast<uint8_t>(get_u64("desc_bits", obj.at("desc_bits")));
    validate_or_throw(desc_bits_ == 4, "map.json desc_bits must be 4 for current pipeline");

    validate_or_throw(has_key_obj(obj, "geo_grid"), "map.json missing geo_grid");
    const JsonValue& geo_v = obj.at("geo_grid");
    validate_or_throw(geo_v.is_object(), "map.json geo_grid must be object");
    const JsonObject& geo_obj = geo_v.as_object();

    validate_or_throw(has_key_obj(geo_obj, "max"), "map.json geo_grid missing max");
    geo_.max = static_cast<uint8_t>(get_u64("geo_grid.max", geo_obj.at("max")));

    validate_or_throw(has_key_obj(geo_obj, "qparams"), "map.json geo_grid missing qparams");
    const JsonValue& qp_v = geo_obj.at("qparams");
    validate_or_throw(qp_v.is_array(), "geo_grid.qparams must be array");
    const JsonArray& qparams = qp_v.as_array();
    validate_or_throw(qparams.size() == 3, "geo_grid.qparams must be [min_x, min_y, R]");

    geo_.min_x = static_cast<float>(get_f64("geo_grid.qparams[0]", qparams[0]));
    geo_.min_y = static_cast<float>(get_f64("geo_grid.qparams[1]", qparams[1]));
    geo_.R     = static_cast<float>(get_f64("geo_grid.qparams[2]", qparams[2]));

    const BinMeta m_desc = read_bin_meta(root, "map_desc_q4");
    const BinMeta m_geo  = read_bin_meta(root, "map_geo_grid");

    validate_or_throw(m_desc.dtype == "uint8", "map_desc_q4 dtype must be uint8");
    validate_or_throw(m_geo.dtype  == "uint8", "map_geo_grid dtype must be uint8");

    validate_or_throw(m_desc.shape.size() == 2 && m_desc.shape[1] == 64, "map_desc_q4 shape must be (N,64)");
    validate_or_throw(m_geo.shape.size()  == 2 && m_geo.shape[1]  == 2,  "map_geo_grid shape must be (N,2)");

    const size_t N_desc = m_desc.shape[0];
    const size_t N_geo  = m_geo.shape[0];
    validate_or_throw(N_desc == N_geo, "map_desc_q4 N != map_geo_grid N");

    n_ = N_desc;

    const std::string p_desc = join_path(dir, m_desc.file);
    const std::string p_geo  = join_path(dir, m_geo.file);

    validate_or_throw(file_size_bytes(p_desc) == m_desc.bytes, "map_desc_q4 bytes mismatch");
    validate_or_throw(file_size_bytes(p_geo)  == m_geo.bytes,  "map_geo_grid bytes mismatch");

    desc_q4_u8_  = read_bin_as<uint8_t>(p_desc, numel_from_shape(m_desc.shape));
    geo_grid_u8_ = read_bin_as<uint8_t>(p_geo,  numel_from_shape(m_geo.shape));

    validate_or_throw(desc_q4_u8_.size()  == n_ * 64, "map_desc_q4 loaded size mismatch");
    validate_or_throw(geo_grid_u8_.size() == n_ * 2,  "map_geo_grid loaded size mismatch");

    return true;
}

bool MapLoader::load_ivf_bins_if_present_(const std::string& dir)
{
    const std::string map_json_path = join_path(dir, "map.json");
    const JsonValue root = load_json_file(map_json_path);
    validate_or_throw(root.is_object(), "map.json root must be object");
    const JsonObject& obj = root.as_object();

    if (!has_key_obj(obj, "ivf")) {
        ivf_.enabled = false;
        return false;
    }
    const JsonValue& ivf_v = obj.at("ivf");
    if (!ivf_v.is_object()) {
        ivf_.enabled = false;
        return false;
    }
    const JsonObject& ivf_obj = ivf_v.as_object();
    if (!has_key_obj(ivf_obj, "nlist")) {
        ivf_.enabled = false;
        return false;
    }

    ivf_.enabled = true;
    ivf_.nlist = static_cast<uint32_t>(get_u64("ivf.nlist", ivf_obj.at("nlist")));

    const BinMeta m_cent = read_bin_meta(root, "ivf_centroids");
    const BinMeta m_off  = read_bin_meta(root, "ivf_postings_offsets");
    const BinMeta m_ids  = read_bin_meta(root, "ivf_postings_map_ids");

    validate_or_throw(m_cent.dtype == "float32", "ivf_centroids dtype must be float32");
    validate_or_throw(m_off.dtype  == "uint32",  "ivf_postings_offsets dtype must be uint32");
    validate_or_throw(m_ids.dtype  == "int32",   "ivf_postings_map_ids dtype must be int32");

    validate_or_throw(m_cent.shape.size() == 2, "ivf_centroids shape must be (K,D)");
    validate_or_throw(m_cent.shape[0] == static_cast<size_t>(ivf_.nlist), "ivf_centroids K mismatch with ivf.nlist");

    ivf_.dim = static_cast<uint32_t>(m_cent.shape[1]);
    validate_or_throw(ivf_.dim > 0, "ivf_centroids D must be > 0");

    validate_or_throw(m_off.shape.size() == 1, "ivf_postings_offsets shape must be (K+1,)");
    validate_or_throw(m_off.shape[0] == static_cast<size_t>(ivf_.nlist) + 1u, "ivf_postings_offsets must be (K+1,)");

    validate_or_throw(m_ids.shape.size() == 1, "ivf_postings_map_ids shape must be (total,)");

    const std::string p_cent = join_path(dir, m_cent.file);
    const std::string p_off  = join_path(dir, m_off.file);
    const std::string p_ids  = join_path(dir, m_ids.file);

    validate_or_throw(file_size_bytes(p_cent) == m_cent.bytes, "ivf_centroids bytes mismatch");
    validate_or_throw(file_size_bytes(p_off)  == m_off.bytes,  "ivf_postings_offsets bytes mismatch");
    validate_or_throw(file_size_bytes(p_ids)  == m_ids.bytes,  "ivf_postings_map_ids bytes mismatch");

    ivf_.centroids_f32        = read_bin_as<float>(p_cent, numel_from_shape(m_cent.shape));
    ivf_.postings_offsets_u32 = read_bin_as<uint32_t>(p_off, numel_from_shape(m_off.shape));
    ivf_.postings_map_ids_i32 = read_bin_as<int32_t>(p_ids, numel_from_shape(m_ids.shape));

    validate_or_throw(ivf_.postings_offsets_u32.size() == static_cast<size_t>(ivf_.nlist) + 1u,
                      "ivf_postings_offsets loaded size mismatch");

    const uint32_t total = ivf_.postings_offsets_u32.back();
    validate_or_throw(ivf_.postings_map_ids_i32.size() == static_cast<size_t>(total),
                      "ivf_postings_map_ids size must equal offsets.back()");

    // optional ivf.extra
    if (has_key_obj(ivf_obj, "extra")) {
        const JsonValue& ex_v = ivf_obj.at("extra");
        if (ex_v.is_object()) {
            const JsonObject& ex = ex_v.as_object();

            auto load_opt_f32_vec = [&](const std::string& key, std::vector<float>& out, size_t expect_numel) {
                auto it = ex.find(key);
                if (it == ex.end()) return;
                if (!it->second.is_object()) return;

                const JsonObject& vo = it->second.as_object();
                auto itref = vo.find("ref");
                if (itref == vo.end()) return;
                if (!itref->second.is_string()) return;

                const std::string ref_key = itref->second.get_string();
                const BinMeta m = read_bin_meta(root, ref_key);
                validate_or_throw(m.dtype == "float32", ref_key + " dtype must be float32");
                const size_t numel = numel_from_shape(m.shape);
                validate_or_throw(numel == expect_numel, ref_key + " numel mismatch");
                const std::string p = join_path(dir, m.file);
                validate_or_throw(file_size_bytes(p) == m.bytes, ref_key + " bytes mismatch");
                out = read_bin_as<float>(p, numel);
            };

            load_opt_f32_vec("val_scale", ivf_.val_scale_f32, 64);
            load_opt_f32_vec("idf_w",     ivf_.idf_w_f32,     64);

            if (has_key_obj(ex, "alpha_value") && ex.at("alpha_value").is_number()) {
                ivf_.alpha_value = static_cast<float>(ex.at("alpha_value").get_number());
            }
            if (has_key_obj(ex, "beta_presence") && ex.at("beta_presence").is_number()) {
                ivf_.beta_presence = static_cast<float>(ex.at("beta_presence").get_number());
            }
            if (has_key_obj(ex, "use_idf") && ex.at("use_idf").is_number()) {
                ivf_.use_idf = (ex.at("use_idf").get_number() != 0.0);
            }
        }
    }

    return true;
}

} // namespace msim