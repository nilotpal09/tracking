#ifndef __TRIPLET_FINDER__
#define __TRIPLET_FINDER__

#include "common/include/module_map.cuh"
/*
template <typename... ARGS>
__global__ void triplet_finding(
    float* x, float* y, float *z, float *r, float *phi, float *theta,
    ARGS... args) {
    HitContainer hc{x, y, z, r, phi, theta};
    __syncthreads();
    triplet_finding_kernal(hc, args...);
}

__device__ void triplet_finding_kernal(
    HitContainer& hit_container,
    unsigned n_mod_triplets, ModuleTriplet *module_triplets,
    unsigned *hits_offsets, unsigned *hits_triplets_acc,
    unsigned *hits_triplets_acc_offsets, unsigned *hits_sum,
    unsigned *hits_sum_offsets, unsigned *hits_module_sum
);
*/
#endif