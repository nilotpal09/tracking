#include "device/include/doublet_finder.cuh"

#define WARP_SIZE 32

__device__ unsigned long long counter[5];

__host__ void print_counter(unsigned n) {
  print_dev_counter<<<1, 1>>>(n);
}

__global__ void print_dev_counter(unsigned n) {
  printf("Counter %d: %llu\n", n, counter[n]);
}

__host__ void doublet_finder(
    Event event,
    ModuleMap module_map
) {
  dim3 grid_dim(65535);
  dim3 block_dim(32, 32);
  dev_doublet_finder<<<grid_dim, block_dim>>>(
    ModuleMap::num_doublets(),
    *module_map.doublets(),
    event.x, event.y, event.z,
    event.r, event.phi, event.eta,
    event.hit_offsets,
    event.hit_pairs, event.hit_pairs_offsets,
    event.hits_a, event.hits_b,
    event.hit_sum, event.hit_sum_offsets,
    event.hit_module_sum
  );
}

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
) {
  HitContainer hit_container{x, y, z, r, phi, eta};
  doublet_finder_kernal(
    n_mod_pairs,
    mod_pairs,
    hit_container,
    hit_offsets,
    hit_pairs, hit_pairs_offsets,
    hits_a, hits_b,
    hit_sum, hit_sum_offsets,
    hit_module_sum
  );
}

__device__ void doublet_finder_kernal(
    unsigned n_mod_pairs,
    ModuleDoublet* mod_pairs,
    HitContainer& hit_container,
    unsigned* hit_offsets,
    unsigned* hit_pairs, unsigned* hit_pairs_offsets,
    unsigned* hits_a, unsigned* hits_b,
    unsigned* hit_sum, unsigned* hit_sum_offsets,
    unsigned* hit_module_sum
) {
  unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned warp_id = tid / WARP_SIZE;

  // Warp reduce
  typedef cub::WarpReduce<unsigned> WarpReduce;
  typedef cub::BlockScan<unsigned, 32, cub::BLOCK_SCAN_RAKING, 32> BlockScan;

  __shared__ union {
    typename WarpReduce::TempStorage warp[32];
    typename BlockScan::TempStorage block;
  } temp_storage;


  auto warp_reduce = WarpReduce(temp_storage.warp[warp_id]);
  auto block_scan = BlockScan(temp_storage.block);

  __shared__ unsigned temp_hit_sum[32][WARP_SIZE];


  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_pairs; mod_idx += gridDim.x) {
    ModuleDoublet mod_dbl = mod_pairs[mod_idx];
    unsigned mod_i = mod_dbl.module_i;
    unsigned mod_j = mod_dbl.module_j;

    unsigned hits_i_offset = hit_offsets[mod_i];
    unsigned hits_j_offset = hit_offsets[mod_j];

    unsigned n_hits_i = hit_offsets[mod_i + 1] - hits_i_offset;
    unsigned n_hits_j = hit_offsets[mod_j + 1] - hits_j_offset;

    unsigned hits_pairs_offset = hit_pairs_offsets[mod_idx];
    unsigned *is_hits_pair = &hit_pairs[hits_pairs_offset];

    unsigned hits_sum_offset = hit_sum_offsets[mod_idx];
    unsigned *h_sum = hit_sum + hits_sum_offset + 1;
    
    Hit hit_i, hit_j;
    for (unsigned i = threadIdx.x; i <= WARP_SIZE; i += blockDim.x) {
      for (unsigned j = threadIdx.y; j <= WARP_SIZE; j += blockDim.y) {
        atomicAdd(&counter[0], 1);
      }
    }
    // Loop over all the hit pairs in the module pair and check if they are
    // pass the selection
    //-------------------------------------------------------------------------
    if (n_hits_i <= WARP_SIZE && n_hits_j <= WARP_SIZE) {
      unsigned *row_hits_pairs = is_hits_pair + threadIdx.y * n_hits_j;
      load(hit_i, hit_container, hits_i_offset, n_hits_i);
      load(hit_j, hit_container, hits_j_offset, n_hits_j);
    
      bool accept = doublet_selection(hit_i, hit_j, mod_dbl);
      atomicAdd(&counter[1], accept);
      atomicAdd(&counter[2], 1);
      row_hits_pairs[threadIdx.x] = accept;
      temp_hit_sum[threadIdx.y][threadIdx.x] = accept;
    }
    __syncthreads();
  }
  
}


__device__ void load(
  Hit& hit,
  const HitContainer& hit_container,
  const unsigned offset,
  const unsigned max
) {
  unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (tid < max) {
    hit.x = hit_container.x[offset + tid];
    hit.y = hit_container.y[offset + tid];
    hit.z = hit_container.z[offset + tid];
    hit.r = hit_container.r[offset + tid];
    hit.phi = hit_container.phi[offset + tid];
    hit.eta = hit_container.eta[offset + tid];
  }
}

__device__ bool doublet_selection(
    Hit& hit_a,
    Hit& hit_b,
    ModuleDoublet& mod_dbl
) {
  float dphi = Dphi(hit_a.x, hit_a.y, hit_b.x, hit_b.y);

  float z0 = Z0(hit_a.x, hit_a.y, hit_a.z, hit_b.x, hit_b.y, hit_b.z);

  float deta = Deta(hit_a.x, hit_a.y, hit_a.z, hit_b.x, hit_b.y, hit_b.z);

  float phi_slope = PhiSlope(hit_a.x, hit_a.y, hit_b.x, hit_b.y);

  bool accept = (dphi > mod_dbl.dphi_min) && (dphi < mod_dbl.dphi_max) &&
                (z0 > mod_dbl.z0_min) && (z0 < mod_dbl.z0_max) &&
                (deta > mod_dbl.deta_min) && (deta < mod_dbl.deta_max) &&
                (phi_slope > mod_dbl.phi_slope_min) &&
                (phi_slope < mod_dbl.phi_slope_max);
  return accept;
}

__device__ bool triplet_selection(
    Hit& hit_bottom,
    Hit& hit_top,
    ModuleTriplet& mod_trip
) {
    return true;
}

__device__ float Dphi(const float &x1, const float &y1, const float& x2, const float& y2) {
  float phi1 = Phi(x1,y1);
  float phi2 = Phi(x2,y2);
  float dphi = phi2 - phi1;

    if (dphi>CUDART_PI_F) dphi-=2*CUDART_PI_F;
    if (dphi<-CUDART_PI_F) dphi+=2*CUDART_PI_F;

    return dphi;
}

__device__ float Dr(const float& x1, const float& y1, const float& x2, const float& y2) {
  float r1 = R(x1,y1);
  float r2 = R(x2,y2);
  return r2 - r1;
}

__device__ float PhiSlope(const float& x1, const float& y1, const float& x2, const float& y2) {
  float dphi = Dphi(x1,y1,x2,y2);
  float dr   = Dr(x1,y1,x2,y2);

  float phi_slope = -999;

  if( fabs(dr)>0 ){
      phi_slope = dphi / dr; 
  } else {
    if (dphi > 0) {
      // set to max value of float
      phi_slope = 3.402823466e+38F; 
    } else if (dphi < 0) {
      // set to min value of float
      phi_slope = -3.402823466e+38F; 
    } else {
      // set to 0
      phi_slope = 0; 
    }
  }
  return phi_slope;
}

__device__ float Dz(const float &z1, const float &z2) { return z2 - z1; }

__device__ float Z0(const float& x1, const float& y1, const float& z1,
          const float& x2, const float& y2, const float& z2) {
  float dz = Dz(z1,z2);
    float dr = Dr(x1,y1,x2,y2);
    float r1 = R(x1,y1);

    float z0 = -999;

    if( fabs(dr)>0 ){
        z0 = z1 - r1 * dz / dr;
    } else {
      z0 = dz >= 0 ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return z0;
}

__device__ float Deta(const float& x1, const float& y1, const float& z1,
            const float& x2, const float& y2, const float& z2) {
  float eta1 = Eta(x1,y1,z1);
  float eta2 = Eta(x2,y2,z2);
  // TODO: understand why we do 1-2, while for the other (dphi, dz) we do 2-1
  return eta1 - eta2;
}

__device__ float DiffDzDr(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3,
            const float& z1, const float& z2, const float& z3) {
  float dz_12 = Dz(z1,z2);
    float dz_23 = Dz(z2,z3);
    float dr_12 = Dr(x1, y1, x2, y2);
    float dr_23 = Dr(x2, y2, x3, y3);

    float diff_dzdr = 999;

    if (dr_12!=0 && dr_23!=0) {
        diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
    } else if (dr_12 == 0 && dr_23 == 0) {
      diff_dzdr = 0;
    } else if (dr_12 == 0) {
      diff_dzdr = ((-dz_23 / dr_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dzdr = ((dz_12 / dr_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return diff_dzdr;
}

__device__ float DiffDzDr(const float& dz_12, const float& dz_23,
            const float& dr_12, const float& dr_23) {
  float diff_dzdr = 999;

    if (dr_12!=0 && dr_23!=0) {
        diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
    } else if (dr_12 == 0 && dr_23 == 0) {
      diff_dzdr = 0;
    } else if (dr_12 == 0) {
      diff_dzdr = ((-dz_23 / dr_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dzdr = ((dz_12 / dr_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return diff_dzdr;
}

__device__ float DiffDyDx(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3) {
  float dy_12 = y2 - y1;
    float dy_23 = y2 - y3;
    float dx_12 = x1 - x2;
    float dx_23 = x2 - x3;

    float diff_dydx= 999;
    if (dx_12!=0 && dx_23!=0) {
        diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
    } else if (dx_12 == 0 && dx_23 == 0) {
      diff_dydx = 0;
    } else if (dx_12 == 0) {
      diff_dydx = ((-dy_23 / dx_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dydx = ((dy_12 / dx_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return diff_dydx;
}

__device__ float DiffDyDx(const float& dy_12, const float& dy_23,
            const float& dx_12, const float& dx_23) {
  float diff_dydx= 999;
    if (dx_12!=0 && dx_23!=0) {
        diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
    } else if (dx_12 == 0 && dx_23 == 0) {
      diff_dydx = 0;
    } else if (dx_12 == 0) {
      diff_dydx = ((-dy_23 / dx_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dydx = ((dy_12 / dx_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
}
/*

__device__ void
doublet_finding_kernal(HitContainer hit_container, unsigned n_mod_pairs,
                       ModuleDoublet *module_pairs, unsigned *hits_offsets,
                       unsigned *hits_pairs_acc,
                       unsigned *hits_pairs_acc_offsets, unsigned *hits_sum,
                       unsigned *hits_sum_offsets, unsigned *hits_module_sum) {
  // Define thread ID and warp ID
  unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned warp_id = tid / WARP_SIZE;

  // Warp reduce
  typedef cub::WarpReduce<unsigned> WarpReduce;
  __shared__ typename WarpReduce::TempStorage
      temp_storage[32]; // We have 32 warps (1024 threads)
  auto warp_reduce = WarpReduce(temp_storage[warp_id]);
  __shared__ unsigned temp_hit_sum[32][WARP_SIZE];

  // Block scan
  typedef cub::BlockScan<unsigned, 32, cub::BLOCK_SCAN_RAKING, 32> BlockScan;
  __shared__ typename BlockScan::TempStorage block_temp_storage;
  auto block_scan = BlockScan(block_temp_storage);

  // Kernal hit containers
  HitSubContainer<WARP_SIZE, cub::WARP_LOAD_STRIPED> i_hits(hit_container, tid);
  HitSubContainer<WARP_SIZE, cub::WARP_LOAD_DIRECT> j_hits(hit_container, tid);

  // Loop over all the module pairs
  //-------------------------------------------------------------------------
  for (unsigned mod_idx = blockIdx.x;
       mod_idx < n_mod_pairs, mod_idx += gridDim.x) {
    ModuleDoublet mod_dbl = module_pairs[mod_idx];
    unsigned mod_i = mod_dbl.module_a;
    unsigned mod_j = mod_dbl.module_b;

    unsigned hits_i_offset = hits_offsets[mod_i];
    unsigned hits_j_offset = hits_offsets[mod_j];

    unsigned n_hits_i = hits_offsets[mod_i + 1] - hits_i_offset;
    unsigned n_hits_j = hits_offsets[mod_j + 1] - hits_j_offset;

    unsigned hits_pairs_offset = hits_pairs_offsets[mod_idx];
    unsigned *is_hits_pair = &hits_pairs[hits_pairs_offset];

    unsigned hits_sum_offset = hits_sum_offsets[mod_idx];
    unsigned *h_sum = hits_sum + hits_sum_offset + 1;

    // Loop over all the hit pairs in the module pair and check if they are
    // pass the selection
    //-------------------------------------------------------------------------
    if (n_hits_i <= WARP_SIZE && n_hits_j <= WARP_SIZE) {
      unsigned *row_hits_pairs = is_hits_pair + i * n_hits_j;
      i_hits.load(hits_i_offset, n_hits_i);
      j_hits.load(hits_j_offset, n_hits_j);

      bool accept = doublet_selection(i_hits.hit(), j_hits.hit(), mod_dbl)
          row_hits_pairs[threadIdx.x] = accept;
      temp_hit_sum[threadIdx.y][threadIdx.x] = accept;
    } else if (n_hits_i > WARP_SIZE && n_hits_j <= WARP_SIZE) {
      j_hits.load(hits_j_offset, n_hits_j);

      unsigned n_loops = (n_hits_i + WARP_SIZE - 1) / WARP_SIZE;
      unsigned i_max = min((l + 1) * WARP_SIZE, n_hits_i);
      unsigned i_start = l * WARP_SIZE;
      for (unsigned i = l * WARP_SIZE + threadIdx.x; i < i_max;
           i += blockDim.y) {
        unsigned *row_hits_pairs = is_hits_pair + i * n_hits_j;
        i_hits.load(hits_i_offset, i_max - i_start);

        bool accept = doublet_selection(i_hits.hit(), j_hits.hit(), mod_dbl)
            row_hits_pairs[threadIdx.x] = accept;
        temp_hit_sum[threadIdx.y][threadIdx.x] += accept;
      }
    } else if (n_hits_i <= WARP_SIZE && n_hits_j > WARP_SIZE) {
      i_hits.load(hits_i_offset, n_hits_i);
      unsigned n_loops = (n_hits_j + WARP_SIZE - 1) / WARP_SIZE;
      for (unsigned l = 0; l < n_loops; l++) {
        unsigned j_max = min((l + 1) * WARP_SIZE, n_hits_j);
        unsigned j_start = l * WARP_SIZE;
        for (unsigned j = l * WARP_SIZE + threadIdx.x; j < j_max;
             j += blockDim.y) {
          unsigned *row_hits_pairs = is_hits_pair + i * n_hits_j;
          j_hits.load(hits_j_offset, j_max - j_start);

          bool accept = doublet_selection(i_hits.hit(), j_hits.hit(), mod_dbl)
              row_hits_pairs[threadIdx.x] = accept;
          temp_hit_sum[threadIdx.y][threadIdx.x] += accept;
        }
      }

    } else {
      unsigned n_i_loops = (n_hits_i + WARP_SIZE - 1) / WARP_SIZE;
      unsigned n_j_loops = (n_hits_j + WARP_SIZE - 1) / WARP_SIZE;
      for (unsigned i_l = 0; i_l < n_i_loops; i_l++) {
        unsigned i_max = min((i_l + 1) * WARP_SIZE, n_hits_i);
        unsigned i_start = i_l * WARP_SIZE;
        for (unsigned i = i_l * WARP_SIZE + threadIdx.x; i < i_max;
             i += blockDim.y) {
          unsigned *row_hits_pairs = is_hits_pair + i * n_hits_j;
          i_hits.load(hits_i_offset, i_max - i_start);
          for (unsigned j_l = 0; j_l < n_j_loops; j_l++) {
            unsigned j_max = min((j_l + 1) * WARP_SIZE, n_hits_j);
            unsigned j_start = j_l * WARP_SIZE;
            for (unsigned j = j_l * WARP_SIZE + threadIdx.x; j < j_max;
                 j += blockDim.y) {
              j_hits.load(hits_j_offset, j_max - j_start);

              bool accept =
                  doublet_selection(i_hits.hit(), j_hits.hit(), mod_dbl)
                      row_hits_pairs[threadIdx.x] = accept;
              temp_hit_sum[threadIdx.y][threadIdx.x] += accept;
            }
          }
        }
      }
    }
    __syncthreads();

    // Count how many pairs each hit from the first module is part of
    // --------------------------------------------------------------
    unsigned thread_data = temp_hit_sum[threadIdx.y][threadIdx.x];
    unsigned warp_id = tid / WARP_SIZE;
    unsigned aggregate = warp_reduce.Sum(thread_data);
    unsigned i = tid / WARP_SIZE + (l * WARP_SIZE);
    if (tid % WARP_SIZE == 0 && i < n_hits_i) {
      atomicAdd(&counter2, aggregate);
      h_sum[i] = aggregate;
    }
    __syncthreads();

    // Compute prefix sum of the number of pairs each hit from the first module
    // is part of
    // -----------------------------------------------------------------------
    if (n_hits_i <= 1024) {
      unsigned thread_data;
      if (tid < n_hits_i) {
        thread_data = h_sum[tid];
      } else {
        thread_data = 0;
      }

      unsigned block_aggregate;
      block_scan.InclusiveSum(thread_data, thread_data, block_aggregate);

      if (tid < n_hits_i) {
        h_sum[tid] = thread_data;
      }

      // Store the total number of pairs for this module pair
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        hits_module_sum[mod_idx] = block_aggregate;
      }
    } else {
      printf("Error: n_hits_i > 1024!");
    }
  }
}
*/