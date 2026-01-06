#pragma once
#include "cim_api.hpp"
#include "ivf_bins.hpp"
#include "ivf_layout.hpp"
#include "msim_config.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace msim {

struct MatchResult {
    int32_t best_map_id = -1;
    uint8_t best_mismatch = 0xFFu;
};

class IvfMatcher {
public:
    using MaskRow = std::array<uint8_t, kRowBytes>;

    struct Planes {
        // enough bits to count mismatches up to 64 (6 bits). keep 8 planes for safety.
        std::array<MaskRow, 8> b{};
        void clear();
        void add_mask(const MaskRow& x);
        uint8_t lane_value(int lane) const;
    };

    IvfMatcher(CimModule& cim, const IvfMapBins& map, const IvfPlacement& place);

    // nprobe: how many IVF buckets to scan (0 => scan all lists)
    MatchResult match_one_query_desc(uint8_t qgx, uint8_t qgy, const uint8_t* qdesc64, uint32_t nprobe = 10, uint32_t max_groups_per_list = 0);

private:
    // list selection (CPU)
    std::vector<uint32_t> select_lists_cpu(const uint8_t* qdesc64, uint32_t nprobe) const;

    // geo / desc compute on a single list
    void geo_eq_masks_xy(uint32_t bucket_id, uint32_t group_in_bucket, uint8_t qgx, uint8_t qgy, MaskRow& out_mask_xy);
    MaskRow geo_eq_mask_axis(uint32_t bucket_id, uint32_t group_in_bucket, bool is_x, uint8_t v);

    void desc_mismatch_planes(uint32_t bucket_id, uint32_t group_in_bucket, const uint8_t* qdesc64, Planes& out_planes);

    // helpers
    static void mask_and_inplace(MaskRow& a, const MaskRow& b);
    static void mask_or_inplace (MaskRow& a, const MaskRow& b);
    static void mask_not_inplace(MaskRow& a);
    static bool mask_any(const MaskRow& a);

private:
    CimModule& cim_;
    const IvfMapBins& map_;
    const IvfPlacement& place_;
};

} // namespace msim