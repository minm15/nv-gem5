#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <iomanip>
#include <iostream>
#include "msim_bins.hpp"
#include "layout.hpp"
#include "cim_api.hpp"

namespace msim {

struct MatchResult {
    int32_t best_map_id = -1;
    uint32_t best_mismatch = 0xFFFFFFFFu;
};

class NoIvfMatcher {
public:
    NoIvfMatcher(CimModule& cim, const Placement& place, uint8_t geo_max);

    MatchResult match_one_query_desc(uint8_t qgx, uint8_t qgy, const uint8_t* qdesc64, uint32_t map_N, uint32_t max_groups);

private:
    CimModule& cim_;
    Placement place_;
    uint8_t geo_max_;

    using MaskRow = std::array<uint8_t, msim::kRowBytes>;

    struct Planes {
        std::array<MaskRow, 7> b;
        void clear();
        void add_mask(const MaskRow& x);
        uint8_t lane_value(int lane) const;
    };

    void map_desc_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array) const;
    void map_geo_group_to_region(uint32_t group, uint16_t& bank, uint16_t& mat, uint16_t& array, uint16_t& row_base) const;

    MaskRow geo_eq_mask_axis(uint32_t group, bool is_x, uint8_t v);
    void geo_eq_masks_xy(uint32_t group, uint8_t qgx, uint8_t qgy, MaskRow& out_mask_xy);

    void desc_mismatch_planes(uint32_t group, const uint8_t* qdesc64, Planes& out_planes);

    static inline void mask_and_inplace(MaskRow& a, const MaskRow& b);
    static inline void mask_or_inplace(MaskRow& a, const MaskRow& b);
    static inline void mask_not_inplace(MaskRow& a);
    static inline bool mask_any(const MaskRow& a);

    static inline uint8_t clamp_u8(int v, uint8_t lo, uint8_t hi)
    {
        if (v < static_cast<int>(lo)) return lo;
        if (v > static_cast<int>(hi)) return hi;
        return static_cast<uint8_t>(v);
    }

    static void dump_hex_prefix(const MaskRow& a, size_t n = 64) {
        n = std::min(n, a.size());
        std::ios old_state(nullptr);
        old_state.copyfmt(std::cout);

        std::cout << "tmp[0.." << (n ? n - 1 : 0) << "] = ";
        for (size_t i = 0; i < n; ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(a[i]) << (i + 1 == n ? "" : " ");
        }
        std::cout << std::dec << "\n";

        std::cout.copyfmt(old_state);
    }
};

} // namespace msim