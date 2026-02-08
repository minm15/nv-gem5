#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <limits>

#include "msim_config.hpp"
#include "map_loader.hpp"
#include "query_loader.hpp"

#include "ivf_bins.hpp"
#include "ivf_layout.hpp"

// ---------------------------
// helpers
// ---------------------------
static uint32_t parse_u32_or(const char* s, uint32_t defv)
{
    if (!s) return defv;
    char* endp = nullptr;
    unsigned long v = std::strtoul(s, &endp, 10);
    if (!endp || *endp != '\0') return defv;
    return static_cast<uint32_t>(v);
}

static void print_usage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " [step_idx] [nprobe]\n"
        << "\n"
        << "Args:\n"
        << "  step_idx   : which query step to inspect (default: ALL)\n"
        << "  nprobe     : how many IVF lists to probe (default: 32)\n"
        << "\n"
        << "Note: This tool now runs Comparison Mode (Base vs NN vs Gap-Filling) automatically.\n";
}

// Helper: L2 Distance Squared
static float dist_l2_sq(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// ---------------------------
// Layout Access Helpers
// ---------------------------
static std::vector<std::tuple<uint16_t,uint16_t,uint16_t>>
bucket_all_desc_regions(const msim::IvfPlacement& place, uint32_t bucket_id)
{
    std::set<std::tuple<uint16_t,uint16_t,uint16_t>> uniq;
    if (bucket_id >= place.bucket.size()) return {};
    const auto& bp = place.bucket[bucket_id];
    for (uint32_t g = 0; g < bp.desc_groups; ++g) {
        uint16_t b=0, m=0, a=0;
        msim::map_desc_bucket_group_to_region(place, bucket_id, g, b, m, a);
        uniq.insert({b,m,a});
    }
    return std::vector<std::tuple<uint16_t,uint16_t,uint16_t>>(uniq.begin(), uniq.end());
}

static std::vector<std::tuple<uint16_t,uint16_t>>
bucket_all_desc_mats(const msim::IvfPlacement& place, uint32_t bucket_id)
{
    std::set<std::tuple<uint16_t,uint16_t>> uniq;
    if (bucket_id >= place.bucket.size()) return {};
    const auto& bp = place.bucket[bucket_id];
    for (uint32_t g = 0; g < bp.desc_groups; ++g) {
        uint16_t b=0, m=0, a=0;
        msim::map_desc_bucket_group_to_region(place, bucket_id, g, b, m, a);
        uniq.insert({b, m});
    }
    return std::vector<std::tuple<uint16_t,uint16_t>>(uniq.begin(), uniq.end());
}

// per list: unique (bank,mat)
static std::vector<std::vector<std::tuple<uint16_t,uint16_t>>>
compute_mats_per_list(const msim::IvfPlacement& place)
{
    std::vector<std::vector<std::tuple<uint16_t,uint16_t>>> mats(place.bucket.size());
    for (uint32_t lid = 0; lid < place.bucket.size(); ++lid) {
        mats[lid] = bucket_all_desc_mats(place, lid);
    }
    return mats;
}

// per list: unique (bank,mat,array)
static std::vector<std::vector<std::tuple<uint16_t,uint16_t,uint16_t>>>
compute_arrays_regions_per_list(const msim::IvfPlacement& place)
{
    std::vector<std::vector<std::tuple<uint16_t,uint16_t,uint16_t>>> regs(place.bucket.size());
    for (uint32_t lid = 0; lid < place.bucket.size(); ++lid) {
        regs[lid] = bucket_all_desc_regions(place, lid);
    }
    return regs;
}

// ---------------------------
// ORDER STRATEGY 1: Centroid NN (Path / TSP-like)
// ---------------------------
static std::vector<uint32_t> build_order_centroid_nn(const msim::IvfMapBins& map, uint32_t start = 0)
{
    const uint32_t K   = map.model.nlist;
    const uint32_t dim = map.model.dim;
    if (K == 0) return {};
    start = (K ? (start % K) : 0);

    std::vector<uint32_t> order;
    order.reserve(K);
    std::vector<uint8_t> used(K, 0);

    uint32_t cur = start;
    used[cur] = 1;
    order.push_back(cur);

    for (uint32_t t = 1; t < K; ++t) {
        uint32_t best = UINT32_MAX;
        float best_d  = std::numeric_limits<float>::max();
        const float* c_cur = &map.model.centroids[static_cast<size_t>(cur) * dim];

        for (uint32_t cand = 0; cand < K; ++cand) {
            if (used[cand]) continue;
            const float* c_cand = &map.model.centroids[static_cast<size_t>(cand) * dim];
            float d = dist_l2_sq(c_cur, c_cand, dim);
            if (d < best_d) {
                best = cand;
                best_d = d;
            }
        }
        if (best == UINT32_MAX) break;
        used[best] = 1;
        order.push_back(best);
        cur = best;
    }

    for (uint32_t i = 0; i < K; ++i) if (!used[i]) order.push_back(i);
    return order;
}

// ---------------------------
// ORDER STRATEGY 2: Gap-Filling Spatial (Knapsack)
// ---------------------------
static std::vector<uint32_t> build_order_gap_filling(const msim::IvfMapBins& map)
{
    const uint32_t K   = map.model.nlist;
    const uint32_t dim = map.model.dim;
    if (K == 0) return {};

    std::cout << "[order] Building Gap-Filling Spatial Order...\n";

    // 1. Calculate weights
    std::vector<uint32_t> weights(K);
    const uint32_t lane_limit = 512; 
    
    if (map.postings.offsets.size() < K + 1) {
        std::fill(weights.begin(), weights.end(), 1);
    } else {
        for (uint32_t i = 0; i < K; ++i) {
            uint32_t count = map.postings.offsets[i+1] - map.postings.offsets[i];
            weights[i] = (count + lane_limit - 1) / lane_limit;
            if (weights[i] == 0) weights[i] = 1;
            if (count == 0) weights[i] = 0;
        }
    }

    std::vector<uint32_t> order;
    order.reserve(K);
    std::vector<uint8_t> used(K, 0);

    // Initial seed
    uint32_t current_id = 0;
    used[current_id] = 1;
    order.push_back(current_id);

    const int MAT_CAP = 16;
    int rem_space = MAT_CAP - (weights[current_id] % MAT_CAP);
    if (rem_space == MAT_CAP) rem_space = 0; 

    const size_t SEARCH_WINDOW = 128; // Look at nearest 128 neighbors

    for (uint32_t t = 1; t < K; ++t) {
        const float* c_curr = &map.model.centroids[static_cast<size_t>(current_id) * dim];

        // Gather nearest neighbors
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(K - t);

        // Simple scan (can be optimized)
        for (uint32_t cand = 0; cand < K; ++cand) {
            if (used[cand]) continue;
            float d = dist_l2_sq(c_curr, &map.model.centroids[static_cast<size_t>(cand) * dim], dim);
            candidates.push_back({d, cand});
        }

        // Sort by distance
        size_t limit = std::min(candidates.size(), SEARCH_WINDOW);
        std::partial_sort(candidates.begin(), candidates.begin() + limit, candidates.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });

        uint32_t best_choice = UINT32_MAX;
        
        // Priority 1: Find best FIT within the window
        for (size_t i = 0; i < limit; ++i) {
            uint32_t cand = candidates[i].second;
            uint32_t w = weights[cand];
            
            // "Knapsack" logic: prefer fitting in the gap
            if (w <= static_cast<uint32_t>(rem_space) && w > 0) {
                best_choice = cand;
                break; 
            }
        }

        // Priority 2: If no fit, take closest (Spatial logic)
        if (best_choice == UINT32_MAX) {
            if (!candidates.empty()) {
                best_choice = candidates[0].second;
            }
        }

        if (best_choice != UINT32_MAX) {
            used[best_choice] = 1;
            order.push_back(best_choice);
            
            // Update state
            uint32_t w = weights[best_choice];
            int needed = static_cast<int>(w);
            
            if (needed <= rem_space) {
                rem_space -= needed;
            } else {
                needed -= rem_space;
                needed %= MAT_CAP;
                rem_space = MAT_CAP - needed;
            }
            if (rem_space == MAT_CAP) rem_space = 0;

            current_id = best_choice;
        } else {
            break; 
        }
    }

    for (uint32_t i = 0; i < K; ++i) if (!used[i]) order.push_back(i);
    return order;
}

// ---------------------------
// Query Selection
// ---------------------------
static std::vector<uint32_t> select_lists_cpu_debug(const msim::IvfMapBins& map,
                                                    const uint8_t* qdesc64,
                                                    uint32_t nprobe)
{
    const uint32_t K = map.model.nlist;
    if (K == 0) return {};
    if (nprobe == 0 || nprobe > K) nprobe = K;
    const uint32_t dim = map.model.dim;

    std::vector<float> q(dim, 0.0f);
    const float alpha   = map.model.alpha_value;
    const float beta    = map.model.beta_presence;
    const bool  use_idf = (map.model.use_idf != 0);

    auto vs = [&](int d) -> float {
        return map.model.val_scale.empty() ? 1.0f : map.model.val_scale.at(static_cast<size_t>(d));
    };
    auto iw = [&](int d) -> float {
        if (!use_idf) return 1.0f;
        return map.model.idf_w.empty() ? 1.0f : map.model.idf_w.at(static_cast<size_t>(d));
    };

    if (dim >= 128) {
        for (int d = 0; d < 64; ++d) {
            const uint8_t qv = static_cast<uint8_t>(qdesc64[d] & 0x0Fu);
            const float idf  = iw(d);
            const float val  = alpha * idf * vs(d) * static_cast<float>(qv);
            const float pres = beta  * idf * (qv != 0 ? 1.0f : 0.0f);
            q[static_cast<size_t>(d)]      = val;
            q[static_cast<size_t>(64 + d)] = pres;
        }
    } else {
        const uint32_t D = std::min<uint32_t>(dim, 64);
        for (uint32_t d = 0; d < D; ++d) q[d] = static_cast<float>(qdesc64[d] & 0x0Fu);
    }

    struct Pair { float dist; uint32_t id; };
    std::vector<Pair> best;
    best.reserve(nprobe);

    for (uint32_t k = 0; k < K; ++k) {
        const float* c = &map.model.centroids[static_cast<size_t>(k) * dim];
        float s = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            float diff = q[i] - c[i];
            s += diff * diff;
        }

        if (best.size() < nprobe) {
            best.push_back({s, k});
            if (best.size() == nprobe) {
                std::nth_element(best.begin(), best.end() - 1, best.end(),
                                 [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
            }
        } else if (s < best.back().dist) {
            best.back() = {s, k};
            std::nth_element(best.begin(), best.end() - 1, best.end(),
                             [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
        }
    }
    std::sort(best.begin(), best.end(), [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
    std::vector<uint32_t> out;
    out.reserve(best.size());
    for (const auto& p : best) out.push_back(p.id);
    return out;
}

// ---------------------------
// Struct to hold per-frame results
// ---------------------------
struct FrameResult {
    size_t step_idx;
    int gx, gy;
    size_t m;
    uint64_t base_or;
    uint64_t nn_or;  // Centroid-NN
    uint64_t gap_or; // Gap-Filling
    double speed_nn;
    double speed_gap;
};

// ---------------------------
// main
// ---------------------------
int main(int argc, char** argv)
{
    uint32_t step_idx_arg = (argc >= 2) ? parse_u32_or(argv[1], UINT32_MAX) : UINT32_MAX;
    uint32_t nprobe       = (argc >= 3) ? parse_u32_or(argv[2], 32) : 32;

    if (argc >= 2 && std::string(argv[1]) == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    const std::string map_dir   = msim::kMapDir;
    const std::string query_dir = msim::kQueryDir;

    std::cout << "========================================================\n";
    std::cout << " IVF Offline Stats - Strategy Comparison\n";
    std::cout << "========================================================\n";
    std::cout << "[cfg] nprobe=" << nprobe << "\n";
    std::cout << "[cfg] Comparing: Baseline vs. Centroid-NN vs. Gap-Filling\n";

    // ---- load map/query ----
    msim::MapLoader map_loader;
    map_loader.load_from_folder(map_dir);

    msim::QueryLoader query_loader;
    query_loader.load_from_folder(query_dir);

    // ---- build IvfMapBins ----
    msim::IvfMapBins map;
    map.N        = static_cast<uint32_t>(map_loader.n());
    map.geo_max  = map_loader.geo().max;
    map.desc_q4  = map_loader.desc_q4_u8();
    map.geo_grid = map_loader.geo_grid_u8();

    if (!map_loader.ivf().enabled) {
        std::cerr << "[fatal] map.ivf is not enabled in map.json\n";
        return 1;
    }

    map.postings.nlist   = map_loader.ivf().nlist;
    map.postings.offsets = map_loader.ivf().postings_offsets_u32;
    map.postings.map_ids = map_loader.ivf().postings_map_ids_i32;

    map.model.nlist      = map_loader.ivf().nlist;
    map.model.dim        = map_loader.ivf().dim;
    map.model.centroids  = map_loader.ivf().centroids_f32;
    map.model.alpha_value   = map_loader.ivf().alpha_value;
    map.model.beta_presence = map_loader.ivf().beta_presence;
    map.model.use_idf       = map_loader.ivf().use_idf ? 1 : 0;
    map.model.val_scale     = map_loader.ivf().val_scale_f32;
    map.model.idf_w         = map_loader.ivf().idf_w_f32;

    std::cout << "[info] N=" << map.N << " K=" << map.postings.nlist << "\n";

    const size_t S = query_loader.steps();
    if (S == 0) return 1;

    // ==========================================
    // 1. Build Layouts for ALL Strategies
    // ==========================================
    
    // A. Baseline
    const msim::IvfPlacement place_base = msim::compute_ivf_placement(map);
    auto mats_base = compute_mats_per_list(place_base);

    // B. Centroid NN (TSP-like)
    std::cout << "[setup] Generating Centroid-NN Order...\n";
    auto order_nn = build_order_centroid_nn(map, 0);
    const msim::IvfPlacement place_nn = msim::compute_ivf_placement(map, order_nn);
    auto mats_nn = compute_mats_per_list(place_nn);

    // C. Gap-Filling (Knapsack)
    auto order_gap = build_order_gap_filling(map);
    const msim::IvfPlacement place_gap = msim::compute_ivf_placement(map, order_gap);
    auto mats_gap = compute_mats_per_list(place_gap);

    // ==========================================
    // 2. Query Loop
    // ==========================================
    size_t s0 = 0, s1 = S;
    if (step_idx_arg != UINT32_MAX) {
        s0 = static_cast<size_t>(step_idx_arg % static_cast<uint32_t>(S));
        s1 = s0 + 1;
    }

    constexpr uint64_t ORS_PER_MAT = 64;
    std::vector<FrameResult> results;
    results.reserve(s1 - s0);

    double total_speed_nn = 0.0;
    double total_speed_gap = 0.0;
    
    // [MODIFIED] Added accumulators for raw OR counts
    double total_base_or_sum = 0.0;
    double total_nn_or_sum   = 0.0;
    double total_gap_or_sum  = 0.0;

    int count = 0;

    for (size_t step_idx = s0; step_idx < s1; ++step_idx) {
        const msim::QueryStepView st = query_loader.step(step_idx);

        uint64_t matops_base = 0;
        uint64_t matops_nn   = 0;
        uint64_t matops_gap  = 0;

        for (size_t qi = 0; qi < st.m; ++qi) {
            const uint8_t* qdesc = st.desc_ptr + qi * 64u;
            std::vector<uint32_t> lists = select_lists_cpu_debug(map, qdesc, nprobe);

            // Calc Base
            {
                std::set<std::tuple<uint16_t,uint16_t>> uniq;
                for (uint32_t lid : lists) {
                    if (lid >= mats_base.size()) continue;
                    for (const auto& bm : mats_base[lid]) uniq.insert(bm);
                }
                matops_base += uniq.size();
            }

            // Calc NN
            {
                std::set<std::tuple<uint16_t,uint16_t>> uniq;
                for (uint32_t lid : lists) {
                    if (lid >= mats_nn.size()) continue;
                    for (const auto& bm : mats_nn[lid]) uniq.insert(bm);
                }
                matops_nn += uniq.size();
            }

            // Calc Gap
            {
                std::set<std::tuple<uint16_t,uint16_t>> uniq;
                for (uint32_t lid : lists) {
                    if (lid >= mats_gap.size()) continue;
                    for (const auto& bm : mats_gap[lid]) uniq.insert(bm);
                }
                matops_gap += uniq.size();
            }
        }

        const uint64_t base_or = matops_base * ORS_PER_MAT;
        const uint64_t nn_or   = matops_nn   * ORS_PER_MAT;
        const uint64_t gap_or  = matops_gap  * ORS_PER_MAT;

        double sp_nn  = (nn_or > 0) ? (double)base_or / nn_or : 0.0;
        double sp_gap = (gap_or > 0) ? (double)base_or / gap_or : 0.0;

        total_speed_nn += sp_nn;
        total_speed_gap += sp_gap;
        
        // [MODIFIED] Accumulate total ORs
        total_base_or_sum += (double)base_or;
        total_nn_or_sum   += (double)nn_or;
        total_gap_or_sum  += (double)gap_or;
        
        count++;

        results.push_back({
            step_idx, (int)st.gx, (int)st.gy, st.m,
            base_or, nn_or, gap_or, sp_nn, sp_gap
        });
    }

    // ==========================================
    // 3. Print Results
    // ==========================================
    if (count > 0) {
        std::cout << "\n[SUMMARY] Comparison Results (Mean OR & Speedup)\n";
        std::cout << "-----------------------------------------------------\n";
        
        // [MODIFIED] Print Mean ORs alongside Speedups
        double mean_base = total_base_or_sum / count;
        double mean_nn   = total_nn_or_sum / count;
        double mean_gap  = total_gap_or_sum / count;
        double mean_sp_nn = total_speed_nn / count;
        double mean_sp_gap = total_speed_gap / count;

        std::cout << " Baseline                      : " 
                  << std::fixed << std::setprecision(1) << mean_base << " OR (avg)\n";
        
        std::cout << " Strategy 1 (Centroid-NN Path) : " 
                  << std::fixed << std::setprecision(1) << mean_nn << " OR (avg) -> "
                  << std::fixed << std::setprecision(3) << mean_sp_nn << "x Speedup\n";
                  
        std::cout << " Strategy 2 (Gap-Filling)      : " 
                  << std::fixed << std::setprecision(1) << mean_gap << " OR (avg) -> "
                  << std::fixed << std::setprecision(3) << mean_sp_gap << "x Speedup\n";
                  
        std::cout << "-----------------------------------------------------\n\n";
    }

    std::cout << " Step |    m | BASE OR |   NN OR (Speed) |  GAP OR (Speed)\n";
    std::cout << "------+------+---------+-----------------+----------------\n";
    
    for (const auto& r : results) {
        std::cout << std::setw(5) << r.step_idx << " | "
                  << std::setw(4) << r.m << " | "
                  << std::setw(7) << r.base_or << " | "
                  << std::setw(7) << r.nn_or 
                  << " (" << std::fixed << std::setprecision(2) << r.speed_nn << "x) | "
                  << std::setw(7) << r.gap_or 
                  << " (" << std::fixed << std::setprecision(2) << r.speed_gap << "x)\n";
    }

    return 0;
}