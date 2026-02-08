#include "ivf_layout.hpp"
#include "layout.hpp"
#include <algorithm>

namespace msim {

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

static inline void pad_to_next_mat_if_needed(uint32_t& cur_arrays,
                                             uint32_t need_arrays,
                                             uint32_t arrays_per_mat)
{
    if (need_arrays == 0) return;
    const uint32_t rem = cur_arrays % arrays_per_mat;
    if (rem == 0) return;

    const uint32_t left = arrays_per_mat - rem;
    if (need_arrays > left) {
        cur_arrays += left;
    }
}

void decode_linear_array(uint32_t linear_array,
                         uint16_t bank0,
                         uint16_t& bank, uint16_t& mat, uint16_t& array)
{
    const uint32_t arrays_per_bank = (1u << kMatBits) * (1u << kArrayBits); // 256
    const uint32_t arrays_per_mat  = (1u << kArrayBits);                    // 16

    const uint32_t bank_sel = linear_array / arrays_per_bank;
    const uint32_t within   = linear_array % arrays_per_bank;

    bank  = static_cast<uint16_t>(bank0 + bank_sel);
    mat   = static_cast<uint16_t>(within / arrays_per_mat);
    array = static_cast<uint16_t>(within % arrays_per_mat);
}

IvfPlacement compute_ivf_placement(const IvfMapBins& map)
{
    const uint32_t K = map.postings.nlist;
    std::vector<uint32_t> identity;
    identity.reserve(K);
    for (uint32_t i = 0; i < K; ++i) identity.push_back(i);
    return compute_ivf_placement(map, identity);
}

IvfPlacement compute_ivf_placement(const IvfMapBins& map,
                                   const std::vector<uint32_t>& order)
{
    IvfPlacement p;
    p.total_banks = static_cast<uint16_t>(1u << kBankBits);

    const uint32_t K = map.postings.nlist;
    if (K == 0 || map.postings.offsets.size() != static_cast<size_t>(K + 1)) {
        throw std::runtime_error("compute_ivf_placement: invalid postings offsets");
    }

    if (order.size() != static_cast<size_t>(K)) {
        throw std::runtime_error("compute_ivf_placement(order): order size != K");
    }

    // validate permutation [0..K-1]
    {
        std::vector<uint8_t> seen(K, 0);
        for (uint32_t v : order) {
            if (v >= K) throw std::runtime_error("compute_ivf_placement(order): order has out-of-range id");
            if (seen[v]) throw std::runtime_error("compute_ivf_placement(order): order has duplicates");
            seen[v] = 1;
        }
    }

    p.bucket.resize(K);

    // --- first pass: compute group counts (跟你原本一樣) ---
    for (uint32_t b = 0; b < K; ++b) {
        const uint32_t off0 = map.postings.offsets[b];
        const uint32_t off1 = map.postings.offsets[b + 1];
        if (off1 < off0) throw std::runtime_error("compute_ivf_placement: offsets not monotonic");

        IvfBucketPlacement bp;
        bp.map_count   = off1 - off0;
        bp.posting_off = off0;

        bp.desc_groups = ceil_div_u32(bp.map_count, static_cast<uint32_t>(kLanesPerGroup));
        bp.geo_groups  = bp.desc_groups;

        p.bucket[b] = bp;
    }

    // --- second pass: pack arrays bucket-by-bucket, but follow `order` ---
    uint32_t desc_arrays = 0;
    uint32_t geo_arrays  = 0;

    const uint32_t arrays_per_mat  = (1u << kArrayBits);                    // 16
    const uint32_t arrays_per_bank = (1u << kMatBits) * (1u << kArrayBits); // 256

    for (uint32_t idx = 0; idx < K; ++idx) {
        const uint32_t b = order[idx];
        auto& bp = p.bucket[b];

        // how many arrays this bucket needs
        const uint32_t need_desc_arrays = bp.desc_groups; // 1 desc-group per array
        const uint32_t need_geo_arrays  =
            ceil_div_u32(bp.geo_groups, static_cast<uint32_t>(kGeoGroupsPerArray)); // 16 geo-groups per array

        // === mat padding (desc) ===
        // 如果 bucket 需要 arrays <= 1 mat 才做「避免跨 mat」padding
        // （若 need > arrays_per_mat，本來就一定跨 mat，但你也可以讓它對齊邊界；我這裡仍然對齊，通常也比較好看）
        pad_to_next_mat_if_needed(desc_arrays, need_desc_arrays, arrays_per_mat);

        // === mat padding (geo) ===
        pad_to_next_mat_if_needed(geo_arrays, need_geo_arrays, arrays_per_mat);

        bp.desc_array_base = desc_arrays;
        bp.geo_array_base  = geo_arrays;

        desc_arrays += need_desc_arrays;
        geo_arrays  += need_geo_arrays;
    }

    p.total_desc_arrays = desc_arrays;
    p.total_geo_arrays  = geo_arrays;

    p.desc_bank0 = 0;
    p.desc_bank_count = static_cast<uint16_t>(ceil_div_u32(p.total_desc_arrays, arrays_per_bank));

    p.geo_bank0 = static_cast<uint16_t>(p.desc_bank0 + p.desc_bank_count);
    p.geo_bank_count = static_cast<uint16_t>(ceil_div_u32(p.total_geo_arrays, arrays_per_bank));

    if (static_cast<uint32_t>(p.geo_bank0) + static_cast<uint32_t>(p.geo_bank_count) > static_cast<uint32_t>(p.total_banks)) {
        throw std::runtime_error("compute_ivf_placement: banks exceeded (need more banks or smaller mapping)");
    }

    return p;
}

void map_desc_bucket_group_to_region(const IvfPlacement& place,
                                    uint32_t bucket_id,
                                    uint32_t group_in_bucket,
                                    uint16_t& bank, uint16_t& mat, uint16_t& array)
{
    if (bucket_id >= place.bucket.size()) throw std::runtime_error("desc bucket_id out of range");

    const auto& bp = place.bucket[bucket_id];
    if (group_in_bucket >= bp.desc_groups) throw std::runtime_error("desc group_in_bucket out of range");

    const uint32_t linear_array = bp.desc_array_base + group_in_bucket;
    decode_linear_array(linear_array, place.desc_bank0, bank, mat, array);
}

void map_geo_bucket_group_to_region(const IvfPlacement& place,
                                   uint32_t bucket_id,
                                   uint32_t group_in_bucket,
                                   uint16_t& bank, uint16_t& mat, uint16_t& array,
                                   uint16_t& row_base)
{
    if (bucket_id >= place.bucket.size()) throw std::runtime_error("geo bucket_id out of range");

    const auto& bp = place.bucket[bucket_id];
    if (group_in_bucket >= bp.geo_groups) throw std::runtime_error("geo group_in_bucket out of range");

    const uint32_t geo_array_off = group_in_bucket / static_cast<uint32_t>(kGeoGroupsPerArray);
    const uint32_t group_in_array = group_in_bucket % static_cast<uint32_t>(kGeoGroupsPerArray);

    const uint32_t linear_array = bp.geo_array_base + geo_array_off;
    decode_linear_array(linear_array, place.geo_bank0, bank, mat, array);

    row_base = geo_row_base_for_group_in_array(static_cast<uint16_t>(group_in_array));
}

} // namespace msim