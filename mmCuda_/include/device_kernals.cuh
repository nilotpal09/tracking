
#include "../include/device_definitions.cuh"

template <typename... ARGS>
__global__ void doublet_finding(float *x, float *y, float *z, float *r,
                                float *phi, float *theta, ARGS... args) {
  HitContainer hc{x, y, z, r, phi, theta};
  __syncthreads();
  doublet_finding_kernal(hc, args...);
}

__device__
doublet_finding_kernal(unsigned n_mod_pairs, ModuleDoublet *module_pairs,
                       Hit *hits, unsigned *hits_offsets,
                       unsigned *hits_pairs_acc,
                       unsigned *hits_pairs_acc_offsets, unsigned *hits_sum,
                       unsigned *hits_sum_offsets, unsigned *hits_module_sum);

template <typename... ARGS>
__global__ void triplet_finding(float *x, float *y, float *z, float *r,
                                float *phi, float *theta, ARGS... args) {
  HitContainer hc{x, y, z, r, phi, theta};
  __syncthreads();
  triplet_finding_kernal(hc, args...);
}

__device__
triplet_finding_kernal(unsigned n_mod_triplets, ModuleTriplet *module_triplets,
                       Hit *hits, unsigned *hits_offsets,
                       unsigned *hits_triplets_acc,
                       unsigned *hits_triplets_acc_offsets, unsigned *hits_sum,
                       unsigned *hits_sum_offsets, unsigned *hits_module_sum);