#include "../include/triplet_finder.cuh"

__global__ void tid() {
    printf("TID %d \n", blockIdx.x * blockDim.x * blockDim.y
+ threadIdx.y * blockDim.x + threadIdx.x);
}

__device__ bool doublet_selection(
    Hit hit_a,
    Hit hit_b) {

    return true;
}

__device__ bool triplet_selection(
    Hit hit_a,
    Hit hit_b,
    Hit hit_c) {

    return true;
}



__global__ void doublet_finding(
    unsigned n_mods,
    unsigned* module_pairs,
    unsigned* module_pairing_offsets,
    Hit* hits,
    unsigned* hits_offsets,
    unsigned* hits_a_idxs,
    unsigned* hits_b_idxs,
    unsigned* hit_doublets_atomics,
    const unsigned n_max_doublets) {
    for (unsigned mod_i = blockIdx.x; mod_i < n_mods - 1; mod_i += gridDim.x) {
        unsigned mod_i_offset = module_pairing_offsets[mod_i];
        unsigned n_mod_i_pairs = module_pairing_offsets[mod_i + 1] - mod_i_offset;
        for (unsigned mod_j_ind = blockIdx.y; mod_j_ind < n_mod_i_pairs; mod_j_ind += gridDim.y) {
            unsigned mod_j = module_pairs[mod_i_offset + mod_j_ind];

            if (mod_i + 1 > n_mods || mod_j + 1 > n_mods) {
                break;
            }
            
            unsigned hits_i_offset = hits_offsets[mod_i];
            unsigned hits_j_offset = hits_offsets[mod_j];

            unsigned n_hits_i = hits_offsets[mod_i + 1] - hits_i_offset;
            unsigned n_hits_j = hits_offsets[mod_j + 1] - hits_j_offset;

            if (n_hits_i > 1000 || n_hits_j > 1000) {
                break;
            }

            unsigned* hit_doublets_atomic = &hit_doublets_atomics[mod_i_offset + mod_j_ind];
            unsigned* hits_a_idx = &hits_a_idxs[(mod_i_offset + mod_j_ind) * n_max_doublets];
            unsigned* hits_b_idx = &hits_b_idxs[(mod_i_offset + mod_j_ind) * n_max_doublets];

            for (unsigned hit_a_ind = threadIdx.x; hit_a_ind < n_hits_i; hit_a_ind += blockDim.x) {
                for (unsigned hit_b_ind = threadIdx.y; hit_b_ind < n_hits_j; hit_b_ind += blockDim.y) {
                    // Perform some selection on the hit pair
                    if (doublet_selection(
                        hits[hits_i_offset + hit_a_ind],
                        hits[hits_j_offset + hit_b_ind]
                    )) {
                        unsigned pair_id = atomicAdd(hit_doublets_atomic, 1);
                        hits_a_idx[pair_id] = hit_a_ind;
                        hits_b_idx[pair_id] = hit_b_ind;
                    }
                }
            }
        }
    }
}

__global__ void triplet_finding(
    unsigned n_mod_triplets,
    unsigned* module_triplets,
    unsigned* module_pairs,
    unsigned* module_pairing_offsets,
    Hit* hits,
    unsigned* hits_offsets,
    unsigned* hits_a_idxs,
    unsigned* hits_b_idxs,
    unsigned* hit_doublets_atomics_cumsum,
    bool* doublet_acceptance,
    bool* hit_acceptance,
    const unsigned n_max_doublets) {

    for (unsigned mod_triplet = blockIdx.x; mod_triplet < n_mod_triplets; mod_triplet += gridDim.x) {
        unsigned* modules = &module_triplets[mod_triplet * 3];
        unsigned mod_i = modules[0];
        unsigned mod_j = modules[1];
        unsigned mod_k = modules[2];

        unsigned mod_i_offset = module_pairing_offsets[mod_i];
        unsigned n_mod_i_pairs = module_pairing_offsets[mod_i + 1] - mod_i_offset;

        unsigned mod_j_offset = module_pairing_offsets[mod_j];
        unsigned n_mod_j_pairs = module_pairing_offsets[mod_j + 1] - mod_j_offset;
        
        auto find_index = [&](const unsigned mod, const unsigned offset, const unsigned n){
            unsigned* pairs = &module_pairs[offset];
            for (unsigned i = 0; i < n; i++) {
                if (mod == pairs[i]) {
                    return i;
                }
            }
            return (unsigned)0; // Need some better way for checking error
        };

        unsigned mod_j_ind = find_index(mod_j, mod_i_offset, n_mod_i_pairs);
    
        unsigned mod_k_ind = find_index(mod_k, mod_j_offset, n_mod_j_pairs);
        
        unsigned hits_i_offset = hits_offsets[mod_i];
        unsigned hits_j_offset = hits_offsets[mod_j];
        unsigned hits_k_offset = hits_offsets[mod_k];

        unsigned n_a_doublets = hit_doublets_atomics_cumsum[mod_i_offset + mod_j_ind + 1] - hit_doublets_atomics_cumsum[mod_i_offset + mod_j_ind];
        unsigned n_b_doublets = hit_doublets_atomics_cumsum[mod_j_offset + mod_k_ind + 1] - hit_doublets_atomics_cumsum[mod_j_offset + mod_k_ind];
        unsigned* hits_a_idx = &hits_a_idxs[(mod_i_offset + mod_j_ind) * n_max_doublets];
        unsigned* hits_ij_b_idx = &hits_b_idxs[(mod_i_offset + mod_j_ind) * n_max_doublets];
        unsigned* hits_jk_a_idx = &hits_a_idxs[(mod_j_offset + mod_k_ind) * n_max_doublets];
        unsigned* hits_c_idx = &hits_b_idxs[(mod_j_offset + mod_k_ind) * n_max_doublets];
        if (n_a_doublets > 10000 || n_b_doublets > 10000) {
            // printf("Too many doublets: %d %d \n", n_a_doublets, n_b_doublets);
            break;
        }
        
        for (unsigned doublet_a = threadIdx.x; doublet_a < n_a_doublets; doublet_a += blockDim.x) {
            unsigned hit_a_idx = hits_a_idx[doublet_a];
            unsigned hit_ij_b_idx = hits_ij_b_idx[doublet_a];
            
            for (unsigned doublet_b = threadIdx.y; doublet_b < n_b_doublets; doublet_b += blockDim.y) {
                unsigned hit_jk_a_idx = hits_jk_a_idx[doublet_b];
                if (hit_ij_b_idx != hit_jk_a_idx) {
                    continue;
                }
                unsigned hit_c_idx = hits_c_idx[doublet_b];

                if (triplet_selection(
                    hits[hits_i_offset + hit_a_idx],
                    hits[hits_j_offset + hit_ij_b_idx],
                    hits[hits_k_offset + hit_c_idx]
                )) {
                    continue;
                    // Create edge
                    //atomicOr(&&(doublet_acceptance[mod_i_offset + mod_j_ind]), true);
                    //atomicOr(&&(doublet_acceptance[mod_j_offset + mod_k_ind]), true);
                    // Create node
                    //atomicOr(&&((bool)hit_acceptance[hits_i_offset + hit_a_idx]), true);
                    //atomicOr(&&((bool)hit_acceptance[hits_j_offset + hit_ij_b_idx]), true);
                    //atomicOr(&&((bool)hit_acceptance[hits_k_offset + hit_c_idx]), true);
                }
            }
        
        }
    }

}