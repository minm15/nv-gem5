#pragma once
#include <cstdint>
#include <vector>

namespace msim {

// IVF posting: offsets (K+1), ids (total)
struct IvfPostings {
    uint32_t nlist = 0;                 // K
    std::vector<uint32_t> offsets;      // (K+1)
    std::vector<int32_t>  map_ids;      // (total)
};

// IVF centroids + optional projection extras (exported by your python)
struct IvfModel {
    uint32_t nlist = 0;                 // K
    uint32_t dim = 0;                   // usually 128 (aug128)
    std::vector<float> centroids;       // (K*dim) row-major

    // optional
    std::vector<float> val_scale;       // (64) or empty
    std::vector<float> idf_w;           // (64) or empty
    float alpha_value = 1.0f;
    float beta_presence = 5.0f;
    int   use_idf = 1;
};

struct IvfMapBins {
    uint32_t N = 0;                     // total map descriptors
    uint8_t  geo_max = 0;

    // same as dense bins
    std::vector<uint8_t> desc_q4;       // (N*64) uint8
    std::vector<uint8_t> geo_grid;      // (N*2)  uint8 (gx,gy)

    // IVF
    IvfModel model;
    IvfPostings postings;
};

} // namespace msim