#ifndef __TRIPLET_FINDING__
#define __TRIPLET_FINDING__

#include "../include/datatypes.cuh"
#include <cub/cub.cuh>
#include <math_constants.h>

#include <assert.h>

#define WARP_SIZE 32

__device__ bool doublet_selection(Hit &hit_a, Hit &hit_b,
                                  ModuleDoublet &mod_dbl);

__device__ bool triplet_selection(Hit &hit_a, Hit &hit_b, Hit &hit_c,
                                  ModuleTriplet &mod_trip);

__device__ bool triplet_selection_baseline(Hit &hit_a, Hit &hit_b, Hit &hit_c,
                                           ModuleTriplet &mod_trip);

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

__global__ void triplet_finding_baseline(unsigned n_mod_triplets,
                                         ModuleTriplet *module_triplets,
                                         Hit *hits, unsigned *hits_offsets,
                                         unsigned *hits_pairs_acc,
                                         unsigned *hits_pairs_acc_offsets);

// Edge features
__device__ float Dphi(const float &phi1, const float &phi2);
__device__ float Dr(const float &r1, const float &r2);
__device__ float PhiSlope(const float &phi1, const float &r1, const float &phi2,
                          const float &r2);
__device__ float Dz(const float &z1, const float &z2);
__device__ float Z0(const float &r1, const float &r2, const float &z1,
                    const float &z2);
__device__ float Deta(const float &eta1, const float &eta2);
__device__ float DiffDzDr(const float &r1, const float &r2, const float &r3,
                          const float &z1, const float &z2, const float &z3);
__device__ float DiffDzDr(const float &dz_12, const float &dz_23,
                          const float &dr_12, const float &dr_23);
__device__ float DiffDyDx(const float &x1, const float &x2, const float &x3,
                          const float &y1, const float &y2, const float &y3);
__device__ float DiffDyDx(const float &dy_12, const float &dy_23,
                          const float &dx_12, const float &dx_23);

#endif