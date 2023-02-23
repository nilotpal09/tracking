#ifndef __TRIPLET_FINDING__
#define __TRIPLET_FINDING__

#include "../include/datatypes.cuh"

__host__ unsigned* PrefixSum(unsigned* arr, unsigned size);

__device__ bool doublet_selection(Hit hit_a, Hit hit_b);

__device__ bool triplet_selection(Hit hit_a, Hit hit_b, Hit hit_c);

__global__ void doublet_finding(
    unsigned n_mods,
    unsigned* module_pairs,
    unsigned* module_pairing_offsets,
    Hit* hits,
    unsigned* hits_offsets,
    unsigned* hits_a_idxs,
    unsigned* hits_b_idxs,
    unsigned* hit_doublets_atomics,
    const unsigned n_max_doublets);

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
    const unsigned n_max_doublets);

#endif