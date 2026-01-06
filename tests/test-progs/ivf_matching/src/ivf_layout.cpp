#include "ivf_layout.hpp"
#include "layout.hpp"
#include <algorithm>

namespace msim {

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
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
    IvfPlacement p;
    p.total_banks = static_cast<uint16_t>(1u << kBankBits);

    const uint32_t K = map.postings.nlist;
    if (K == 0 || map.postings.offsets.size() != static_cast<size_t>(K + 1)) {
        throw std::runtime_error("compute_ivf_placement: invalid postings offsets");
    }

    p.bucket.resize(K);

    // --- first pass: compute group counts ---
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

    // --- pack arrays bucket-by-bucket (simple rule, good baseline) ---
    uint32_t desc_arrays = 0;
    uint32_t geo_arrays  = 0;

    for (uint32_t b = 0; b < K; ++b) {
        p.bucket[b].desc_array_base = desc_arrays;
        p.bucket[b].geo_array_base  = geo_arrays;

        desc_arrays += p.bucket[b].desc_groups; // 1 desc-group per array
        geo_arrays  += ceil_div_u32(p.bucket[b].geo_groups, static_cast<uint32_t>(kGeoGroupsPerArray)); // 16 geo-groups per array
    }

    p.total_desc_arrays = desc_arrays;
    p.total_geo_arrays  = geo_arrays;

    const uint32_t arrays_per_bank = (1u << kMatBits) * (1u << kArrayBits); // 256

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