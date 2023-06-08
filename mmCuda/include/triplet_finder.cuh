#ifndef __TRIPLET_FINDING__
#define __TRIPLET_FINDING__

#include "../include/datatypes.cuh"
#include <cub/cub.cuh>
#include <math_constants.h>

#include <assert.h>

#define WARP_SIZE 32

__global__ void print_counter();
__global__ void print_counter2();
__global__ void print_counter3();
__global__ void print_counter4();

__global__ void print_all(unsigned n_mod_triplets,
                          ModuleTriplet *module_triplets,
                          Hit *hits, unsigned *hits_offsets,
                          unsigned *hits_pairs_acc,
                          unsigned *hits_pairs_acc_offsets);

__global__ void
doublet_finding(unsigned n_mod_pairs, ModuleDoublet *module_pairs, Hit *hits,
                unsigned *hits_offsets, unsigned *hits_pairs_acc,
                unsigned *hits_pairs_acc_offsets, unsigned *hits_sum,
                unsigned *hits_sum_offsets, unsigned *hits_module_sum);

__global__ void triplet_finding(unsigned n_mod_triplets,
                                ModuleTriplet *module_triplets, Hit *hits,
                                unsigned *hits_offsets, unsigned *hits_pairs,
                                unsigned *hits_pairs_offsets,
                                unsigned *hits_a, unsigned *hits_b,
                                unsigned *hits_module_offsets,
                                unsigned *hits_cumsum, unsigned *hits_sum_offsets);

__global__ void triplet_finding_baseline(unsigned n_mod_triplets,
                                         ModuleTriplet *module_triplets,
                                         Hit *hits, unsigned *hits_offsets,
                                         unsigned *hits_pairs_acc,
                                         unsigned *hits_pairs_acc_offsets);

__global__ void count_reduced_triplets(unsigned n_mod_triplets,
                                ModuleTriplet *module_triplets,
                                ModuleDoublet *module_doublets,
                                Hit *hits,
                                unsigned *hits_offsets,
                                unsigned *hits_a, unsigned *hits_b,
                                unsigned *hits_module_offsets,
                                unsigned *hits_cumsum, unsigned *hits_sum_offsets);


__device__ bool doublet_selection(Hit &hit_a, Hit &hit_b,
                                  ModuleDoublet &mod_dbl);

__device__ bool doublet_selection_baseline(Hit& hit_a, Hit& hit_b, ModuleDoublet& mod_dbl);

__device__ bool triplet_selection(Hit &hit_a, Hit &hit_b, Hit &hit_c,
                                  ModuleTriplet &mod_trip);

__device__ bool triplet_selection_baseline(Hit &hit_a, Hit &hit_b, Hit &hit_c,
                                           ModuleTriplet &mod_trip);

// Edge features
__device__ float Phi(const float &x, const float &y);
__device__ float R(const float &x, const float &y);
__device__ float Eta(const float &x, const float &y, const float &z);
__device__ float Dphi(const float &c1, const float &y1, const float& x2, const float& y2);
__device__ float Dr(const float& x1, const float& y1, const float& x2, const float& y2);
__device__ float PhiSlope(const float& x1, const float& y1, const float& x2, const float& y2);
__device__ float Dz(const float &z1, const float &z2);
__device__ float Z0(const float& x1, const float& y1, const float& z1,
          const float& x2, const float& y2, const float& z2);
__device__ float Deta(const float& x1, const float& y1, const float& z1,
            const float& x2, const float& y2, const float& z2);
__device__ float DiffDzDr(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3,
            const float& z1, const float& z2, const float& z3);
__device__ float DiffDzDr(const float& dz_12, const float& dz_23,
            const float& dr_12, const float& dr_23);
__device__ float DiffDyDx(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3);
__device__ float DiffDyDx(const float& dy_12, const float& dy_23,
            const float& dx_12, const float& dx_23);

#endif