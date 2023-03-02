#include "../include/triplet_finder.cuh"

__global__ void tid() {
    printf("TID %d \n", blockIdx.x * blockDim.x * blockDim.y
+ threadIdx.y * blockDim.x + threadIdx.x);
}

__device__ bool doublet_selection(
    Hit hit_a,
    Hit hit_b) {

    float x_diff = hit_a.x - hit_b.x;
    return true;
}

__device__ bool triplet_selection(
    Hit hit_a,
    Hit hit_b,
    Hit hit_c) {

    return true;
}

#define N_HITS_PER_BLOCK 32

__global__ void doublet_finding(
    unsigned n_mods,
    unsigned n_mod_pairs,
    unsigned* module_pairs_indices,
    unsigned* module_pairs,
    Hit* hits,
    unsigned* hits_offsets,
    bool* hits_pairs_acc,
    unsigned* hits_pairs_acc_offsets) {
    for (unsigned i = blockIdx.x; i < n_mod_pairs - 1; i += gridDim.x) {
        unsigned mod_i = module_pairs_indices[i];
        unsigned mod_j = module_pairs[i];

        unsigned hits_i_offset = hits_offsets[mod_i];
        unsigned hits_j_offset = hits_offsets[mod_j];

        unsigned n_hits_i = hits_offsets[mod_i + 1] - hits_i_offset;
        unsigned n_hits_j = hits_offsets[mod_j + 1] - hits_j_offset;
        
        unsigned hits_pairs_acc_offset = hits_pairs_acc_offsets[i];
        bool* is_hits_pair = &hits_pairs_acc[hits_pairs_acc_offset];

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        // Fix bounday conditions when n_hits is not a multiple of N_HITS_PER_BLOCK
        for (int i = 0; i < n_hits_i/N_HITS_PER_BLOCK; i++) {
            for (int j = 0; j < n_hits_j/N_HITS_PER_BLOCK; j++) {
                int blockId = i + j * N_HITS_PER_BLOCK;
                int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
                if (doublet_selection(hits[hits_i_offset + i * N_HITS_PER_BLOCK + x],
                                      hits[hits_j_offset + j * N_HITS_PER_BLOCK + y])) {
                    is_hits_pair[threadId] = true;
                } else {
                    is_hits_pair[threadId] = false;
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

/*
            unsigned hits_i_offset = hits_offsets[mod_i];
            unsigned hits_j_offset = hits_offsets[mod_j];

            unsigned n_hits_i = hits_offsets[mod_i + 1] - hits_i_offset;
            unsigned n_hits_j = hits_offsets[mod_j + 1] - hits_j_offset;

            unsigned x_loops = n_hits_i / N_HITS_PER_BLOCK;
            unsigned y_loops = n_hits_j / N_HITS_PER_BLOCK;

            __shared__ unsigned hit_doublets_atomic;

            for (unsigned x = 0; x < x_loops; x++) {
                for (unsigned hit_a_ind = threadIdx.x + (x * N_HITS_PER_BLOCK); hit_a_ind < N_HITS_PER_BLOCK + (x * N_HITS_PER_BLOCK); hit_a_ind += blockDim.x) {
                    if (hit_a_ind < n_hits_i) {
                        hits_a[hit_a_ind] = hits[hits_i_offset + hit_a_ind];
                    }   
                }
                for (unsigned y = 0; y < y_loops; y++) {
                    // Copy hits to shared memory
                    for (unsigned hit_b_ind = threadIdx.x + (x * N_HITS_PER_BLOCK); hit_b_ind < N_HITS_PER_BLOCK + (x * N_HITS_PER_BLOCK); hit_b_ind += blockDim.x) {
                        if (hit_b_ind < n_hits_j) {
                            hits_b[hit_b_ind] = hits[hits_j_offset + hit_b_ind];
                        }

                    }
                    __syncthreads();

                    // unsigned* hit_doublets_atomic = &hit_doublets_atomics[mod_i_offset + mod_j_ind];
                    // unsigned* hits_a_idx = &hits_a_idxs[(mod_i_offset + mod_j_ind) * n_max_doublets];
                    // unsigned* hits_b_idx = &hits_b_idxs[(mod_i_offset + mod_j_ind) * n_max_doublets];
                    
                    for (unsigned hit_a_ind = threadIdx.x; hit_a_ind < N_HITS_PER_BLOCK; hit_a_ind += blockDim.x) {
                        for (unsigned hit_b_ind = threadIdx.y; hit_b_ind < N_HITS_PER_BLOCK; hit_b_ind += blockDim.y) {
                            if (hit_a_ind >= n_hits_i || hit_b_ind >= n_hits_j) {
                                break;
                            }
                            // Perform some selection on the hit pair
                            if (doublet_selection(
                                hits_a[hit_a_ind],
                                hits_b[hit_b_ind]
                            )) {
                                unsigned pair_id = atomicAdd(&hit_doublets_atomic, 1);
                                //hits_a_idx[pair_id] = hit_a_ind;
                                //hits_b_idx[pair_id] = hit_b_ind;
                            }
                        }
                    }
                    __syncthreads();
                    
                }
            }
            hit_doublets_atomics[mod_i_offset + mod_j_ind] = hit_doublets_atomic;
            */