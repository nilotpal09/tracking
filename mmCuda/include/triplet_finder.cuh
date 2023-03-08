#ifndef __TRIPLET_FINDING__
#define __TRIPLET_FINDING__

#include "../include/datatypes.cuh"
#include <cub/cub.cuh>

#include <assert.h>

#define WARP_SIZE 32

__global__ void tid();

__host__ unsigned *PrefixSum(unsigned *arr, unsigned size);

__device__ bool doublet_selection(Hit hit_a, Hit hit_b);

__device__ bool triplet_selection(Hit hit_a, Hit hit_b, Hit hit_c);

__global__ void
doublet_finding(unsigned n_mod_pairs, ModuleDoublet *module_pairs, Hit *hits,
                unsigned *hits_offsets, unsigned *hits_pairs_acc,
                unsigned *hits_pairs_acc_offsets, unsigned *hits_sum,
                unsigned *hits_sum_offsets, unsigned *hits_module_sum);

__global__ void triplet_finding(unsigned n_mod_triplets,
                                ModuleTriplet *module_triplets, Hit *hits,
                                unsigned *hits_offsets,
                                unsigned *hits_pairs_acc, unsigned *hits_sum,
                                unsigned *hits_sum_offsets, unsigned *hits_a,
                                unsigned *hits_b, unsigned *hit_module_offsets);

#endif