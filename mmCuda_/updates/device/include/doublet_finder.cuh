#ifndef __DOUBLET_FINDER__
#define __DOUBLET_FINDER__

#include <cub/cub.cuh>
#include <math_constants.h>
#include <stdio.h>

#include "common/include/event.cuh"
#include "common/include/module_map.cuh"
#include "common/include/coordinate_converters.cuh"
#include "device/include/hit_containers.cuh"
#include "device/include/selection.cuh"

__global__ void print_dev_counter(unsigned n);

__host__ void print_counter(unsigned n);

__host__ void doublet_finder(
    Event event,
    ModuleMap module_map
);

__global__ void dev_doublet_finder(
    unsigned n_mod_pairs,
    ModuleDoublet* mod_pairs,
    float* x, float* y, float* z,
    float* r, float* phi, float* eta,
    unsigned* hit_offsets,
    unsigned* hit_pairs, unsigned* hit_pairs_offsets,
    unsigned* hits_a, unsigned* hits_b,
    unsigned* hit_sum, unsigned* hit_sum_offsets,
    unsigned* hit_module_sum
);


__device__ void doublet_finder_kernal(
    unsigned n_mod_pairs,
    ModuleDoublet* mod_pairs,
    HitContainer& hit_container,
    unsigned* hit_offsets,
    unsigned* hit_pairs, unsigned* hit_pairs_offsets,
    unsigned* hits_a, unsigned* hits_b,
    unsigned* hit_sum, unsigned* hit_sum_offsets,
    unsigned* hit_module_sum
);

__device__ void load(
  Hit& hit,
  const HitContainer& hit_container,
  const unsigned offset,
  const unsigned max
);

__device__ bool doublet_selection(
    Hit& hit_bottom,
    Hit& hit_top,
    ModuleDoublet& mod_doub
);

__device__ bool triplet_selection(
    Hit& hit_bottom,
    Hit& hit_top,
    ModuleTriplet& mod_trip
);

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