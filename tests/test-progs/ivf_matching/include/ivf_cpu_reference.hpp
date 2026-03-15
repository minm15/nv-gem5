#pragma once

#include "ivf_bins.hpp"
#include "ivf_matcher.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace msim {

MatchResult match_one_query_desc_cpu_reference(const IvfMapBins& map,
                                               uint8_t qgx,
                                               uint8_t qgy,
                                               const uint8_t* qdesc64,
                                               uint32_t nprobe,
                                               uint32_t max_groups_per_list);

std::vector<MatchResult> match_one_step_cpu_reference(const IvfMapBins& map,
                                                      uint8_t qgx,
                                                      uint8_t qgy,
                                                      const uint8_t* desc_ptr,
                                                      size_t m,
                                                      uint32_t nprobe,
                                                      uint32_t max_groups_per_list);

} // namespace msim
