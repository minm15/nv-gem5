#pragma once
#include "msim_config.hpp"
#include "ivf_bins.hpp"
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace msim {

// Per IVF bucket placement info (bucket == list id)
struct IvfBucketPlacement {
    uint32_t map_count = 0;         // number of map ids in this bucket
    uint32_t posting_off = 0;       // postings.offsets[b]
    uint32_t desc_groups = 0;       // ceil(map_count / kLanesPerGroup)
    uint32_t geo_groups  = 0;       // same as desc_groups

    uint32_t desc_array_base = 0;   // linear array index inside desc region (0..)
    uint32_t geo_array_base  = 0;   // linear array index inside geo region (0..)
};

struct IvfPlacement {
    // bank ranges
    uint16_t total_banks = 0;

    uint16_t desc_bank0 = 0;
    uint16_t desc_bank_count = 0;

    uint16_t geo_bank0 = 0;
    uint16_t geo_bank_count = 0;

    // totals in arrays (linear index)
    uint32_t total_desc_arrays = 0;
    uint32_t total_geo_arrays  = 0;

    std::vector<IvfBucketPlacement> bucket; // size K
};

IvfPlacement compute_ivf_placement(const IvfMapBins& map);

// Helpers: decode linear array index -> (bank,mat,array)
void decode_linear_array(uint32_t linear_array,
                         uint16_t bank0,
                         uint16_t& bank, uint16_t& mat, uint16_t& array);

// Map (bucket, group_in_bucket) -> desc region (bank/mat/array)
void map_desc_bucket_group_to_region(const IvfPlacement& place,
                                    uint32_t bucket_id,
                                    uint32_t group_in_bucket,
                                    uint16_t& bank, uint16_t& mat, uint16_t& array);

// Map (bucket, group_in_bucket) -> geo region (bank/mat/array + row_base)
void map_geo_bucket_group_to_region(const IvfPlacement& place,
                                   uint32_t bucket_id,
                                   uint32_t group_in_bucket,
                                   uint16_t& bank, uint16_t& mat, uint16_t& array,
                                   uint16_t& row_base);

} // namespace msim