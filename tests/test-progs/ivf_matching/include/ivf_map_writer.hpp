#pragma once
#include "cim_api.hpp"
#include "ivf_bins.hpp"
#include "ivf_layout.hpp"

namespace msim {

class IvfMapWriter {
public:
    IvfMapWriter(CimModule& cim, const IvfPlacement& place);

    void write_all(const IvfMapBins& map);

private:
    void write_desc_bucket_group(const IvfMapBins& map, uint32_t bucket_id, uint32_t group_in_bucket);
    void write_geo_bucket_group(const IvfMapBins& map, uint32_t bucket_id, uint32_t group_in_bucket);

private:
    CimModule& cim_;
    const IvfPlacement& place_;
};

} // namespace msim