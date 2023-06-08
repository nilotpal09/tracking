#include "../include/triplet_finder.cuh"
#include "../include/utility.cuh"

__device__ unsigned counter = 0;
__device__ unsigned long long counter2 = 0;
__device__ unsigned long long counter3 = 0;
__device__ unsigned long long counter4 = 0;

__global__ void print_counter() { printf("Counter1: %u \n", counter); }

__global__ void print_counter2() { printf("Counter2: %ld \n", counter2); }

__global__ void print_counter3() { printf("Counter3: %ld \n", counter3); }

__global__ void print_counter4() { printf("Counter4: %ld \n", counter4); }


__global__ void print_all(unsigned n_mod_triplets,
                          ModuleTriplet *module_triplets,
                          Hit *hits, unsigned *hits_offsets,
                          unsigned *hits_pairs_acc,
                          unsigned *hits_pairs_acc_offsets) {
  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_triplets; mod_idx += gridDim.x) {
    ModuleTriplet mod_trip = module_triplets[mod_idx];

    unsigned hits_a_offset = hits_offsets[mod_trip.module_a];
    unsigned hits_b_offset = hits_offsets[mod_trip.module_b];
    unsigned hits_c_offset = hits_offsets[mod_trip.module_c];

    unsigned n_hits_a = hits_offsets[mod_trip.module_a + 1] - hits_a_offset;
    unsigned n_hits_b = hits_offsets[mod_trip.module_b + 1] - hits_b_offset;
    unsigned n_hits_c = hits_offsets[mod_trip.module_c + 1] - hits_c_offset;

    if (n_hits_a * n_hits_b * n_hits_c != 1) {
      continue;
    }
    if (!(mod_trip.module_a == 6798 && mod_trip.module_b == 359 && mod_trip.module_c == 461)) {
      continue;
    }
    printf("Module triplet: %u %u %u \n", mod_trip.module_a, mod_trip.module_b, mod_trip.module_c);
    printf("Number of hits in the first module: %u \n", n_hits_a);
    printf("Number of hits in the second module: %u \n", n_hits_b);
    printf("Number of hits in the third module: %u \n", n_hits_c);

    printf("Min dphi_12: %f \n", mod_trip.dphi_min_ab);
    printf("Max dphi_12: %f \n", mod_trip.dphi_max_ab);
    printf("Min dphi_23: %f \n", mod_trip.dphi_min_bc);
    printf("Max dphi_23: %f \n", mod_trip.dphi_max_bc);
    printf("Min deta_12: %f \n", mod_trip.deta_min_ab);
    printf("Max deta_12: %f \n", mod_trip.deta_max_ab);
    printf("Min deta_23: %f \n", mod_trip.deta_min_bc);
    printf("Max deta_23: %f \n", mod_trip.deta_max_bc);
    printf("Min z0_12: %f \n", mod_trip.z0_min_ab);
    printf("Max z0_12: %f \n", mod_trip.z0_max_ab);
    printf("Min z0_23: %f \n", mod_trip.z0_min_bc);
    printf("Max z0_23: %f \n", mod_trip.z0_max_bc);
    printf("Min phiSlopemin_12: %f \n", mod_trip.phi_slope_min_ab);
    printf("Max phiSlopemax_12: %f \n", mod_trip.phi_slope_max_ab);
    printf("Min phiSlopemin_23: %f \n", mod_trip.phi_slope_min_bc);
    printf("Max phiSlopemax_23: %f \n", mod_trip.phi_slope_max_bc);

    for (unsigned i = threadIdx.y; i < n_hits_a; i += blockDim.y) {
      for (unsigned j = threadIdx.x; j < n_hits_b; j += blockDim.x) {
        float x_centralSP = hits[hits_b_offset + i].x;
        float y_centralSP = hits[hits_b_offset + i].y;
        float z_centralSP = hits[hits_b_offset + i].z;
        float x_bottomSP = hits[hits_a_offset + j].x;
        float y_bottomSP = hits[hits_a_offset + j].y;
        float z_bottomSP = hits[hits_a_offset + j].z;

        printf("dphi_12: %f \n", Dphi(x_bottomSP, y_bottomSP, x_centralSP, y_centralSP));
        printf("deta_12: %f \n", Deta(x_bottomSP, y_bottomSP, z_bottomSP, x_centralSP, y_centralSP, z_centralSP));
        printf("z0_12: %f \n", Z0(x_bottomSP, y_bottomSP, z_bottomSP, x_centralSP, y_centralSP, z_centralSP));
        printf("phiSlope_12: %f \n", PhiSlope(x_bottomSP, y_bottomSP, x_centralSP, y_centralSP));
      }
      for (unsigned k = threadIdx.z; k < n_hits_c; k += blockDim.z) {
        float x_centralSP = hits[hits_b_offset + i].x;
        float y_centralSP = hits[hits_b_offset + i].y;
        float z_centralSP = hits[hits_b_offset + i].z;
        float x_topSP = hits[hits_c_offset + k].x;
        float y_topSP = hits[hits_c_offset + k].y;
        float z_topSP = hits[hits_c_offset + k].z;

        printf("dphi_23: %f \n", Dphi(x_centralSP, y_centralSP, x_topSP, y_topSP));
        printf("deta_23: %f \n", Deta(x_centralSP, y_centralSP, z_centralSP, x_topSP, y_topSP, z_topSP));
        printf("z0_23: %f \n", Z0(x_centralSP, y_centralSP, z_centralSP, x_topSP, y_topSP, z_topSP));
        printf("phiSlope_23: %f \n", PhiSlope(x_centralSP, y_centralSP, x_topSP, y_topSP));
      }
    }
    float x_centralSP = hits[hits_b_offset].x;
    float y_centralSP = hits[hits_b_offset].y;
    float z_centralSP = hits[hits_b_offset].z;
    float x_bottomSP = hits[hits_a_offset].x;
    float y_bottomSP = hits[hits_a_offset].y;
    float z_bottomSP = hits[hits_a_offset].z;
    float x_topSP = hits[hits_c_offset].x;
    float y_topSP = hits[hits_c_offset].y;
    float z_topSP = hits[hits_c_offset].z;
    printf("diff_dydx: %f \n", DiffDyDx(x_bottomSP, x_centralSP, x_topSP, y_bottomSP, y_centralSP, y_topSP));
    printf("diff_dzdr: %f \n", DiffDzDr(x_bottomSP, x_centralSP, x_topSP, y_bottomSP, y_centralSP, y_topSP, z_bottomSP, z_centralSP, z_topSP));
  return;
  }
}


__global__ void doublet_finding(unsigned n_mod_pairs,
                                ModuleDoublet *module_pairs, Hit *hits,
                                unsigned *hits_offsets, unsigned *hits_pairs,
                                unsigned *hits_pairs_offsets,
                                unsigned *hits_sum, unsigned *hits_sum_offsets,
                                unsigned *hits_module_sum) {
                                  typedef cub::WarpReduce<unsigned> WarpReduce;
  __shared__ typename WarpReduce::TempStorage
      temp_storage[32]; // We have 32 warps (1024 threads)
  __shared__ unsigned temp_hit_sum[32][WARP_SIZE];
  __shared__ Hit temp_hits[32][WARP_SIZE];
  
  typedef cub::BlockScan<unsigned, 512, cub::BLOCK_SCAN_RAKING, 1>
          BlockScan;
  __shared__ typename BlockScan::TempStorage block_temp_storage;
  /*
  typedef cub::WarpReduce<unsigned> WarpReduce;
  __shared__ typename WarpReduce::TempStorage
      temp_storage[32]; // We have 32 warps (1024 threads)
  __shared__ unsigned temp_hit_sum[32][WARP_SIZE];
  
  typedef cub::BlockScan<unsigned, 32, cub::BLOCK_SCAN_RAKING, 32>
          BlockScan;
  __shared__ typename BlockScan::TempStorage block_temp_storage;
  */
  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_pairs;
       mod_idx += gridDim.x) {
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
    //    &hits_sum[hits_sum_offset];

    unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;

    /*
    // Loop over all hit pairs and see if they pass the selection criteria
    //--------------------------------------------------------------------
    unsigned n_loops = n_hits_i / WARP_SIZE;
    if (n_hits_i % WARP_SIZE != 0) {
      n_loops += 1;
    }
    for (unsigned l = 0; l < n_loops; l++) {
      temp_hit_sum[threadIdx.y][threadIdx.x] = 0;
      unsigned i_max =
          (l + 1) * WARP_SIZE < n_hits_i ? (l + 1) * WARP_SIZE : n_hits_i;
      for (unsigned i = threadIdx.y + (l * WARP_SIZE); i < i_max;
           i += blockDim.y) {
        unsigned *row_hits_pairs = is_hits_pair + i * n_hits_j;
        //&is_hits_pair[i * n_hits_j];
        for (unsigned j = threadIdx.x; j < n_hits_j; j += blockDim.x) {
          bool accept = doublet_selection_baseline(hits[hits_i_offset + i],
                                                   hits[hits_j_offset + j], mod_dbl);
          row_hits_pairs[j] = accept;
          temp_hit_sum[threadIdx.y][threadIdx.x] += accept;
          atomicAdd(&counter, accept);
        }
      }
    __syncthreads();
    
    // Count how many pairs each hit from the first module is part of
    // --------------------------------------------------------------
      unsigned thread_data = temp_hit_sum[threadIdx.y][threadIdx.x];
      unsigned warp_id = tid / WARP_SIZE;
      unsigned aggregate = WarpReduce(temp_storage[warp_id]).Sum(thread_data);
      unsigned i = tid / WARP_SIZE + (l * WARP_SIZE);
      if (tid % WARP_SIZE == 0 && i < n_hits_i) {
        atomicAdd(&counter2, aggregate);
        h_sum[i] = aggregate;
      }
    }
    __syncthreads();
    */
    // Loop over all hit pairs and see if they pass the selection criteria
    //--------------------------------------------------------------------
    for (unsigned i = threadIdx.y; i < n_hits_i; i += blockDim.y) {
      unsigned *row_hits_pairs = is_hits_pair + i * n_hits_j;
      //&is_hits_pair[i * n_hits_j];
      for (unsigned j = threadIdx.x; j < n_hits_j; j += blockDim.x) {
        bool accept = doublet_selection_baseline(hits[hits_i_offset + i],
                                     hits[hits_j_offset + j], mod_dbl);
        row_hits_pairs[j] = accept;
        atomicAdd(&counter, accept);
        atomicAdd(&h_sum[i], accept);
      }
    }
    __syncthreads();


    // Compute prefix sum of the number of pairs each hit from the first module
    // is part of
    // -----------------------------------------------------------------------
    // TODO: Tidy up this part
    /*
      Algorithms:
      - BLOCK_SCAN_RAKING
      - BLOCK_SCAN_RAKING_MEMOIZE
      - BLOCK_SCAN_WARP_SCANS
    */
    /*
    if (n_hits_i <= 1024) {
      unsigned thread_data;
      if (tid < n_hits_i) {
        thread_data = h_sum[tid];
      } else {
        thread_data = 0;
      }

      unsigned block_aggregate;
      BlockScan(block_temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);

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
    */
    if (n_hits_i <= 512) {
      unsigned thread_data;
      if (tid < n_hits_i) {
        thread_data = h_sum[tid];
      } else {
        thread_data = 0;
      }

      unsigned block_aggregate;
      BlockScan(block_temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);

      if (tid < n_hits_i) {
        h_sum[tid] = thread_data;
      }

      // Store the total number of pairs for this module pair
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        hits_module_sum[mod_idx] = block_aggregate;
        atomicAdd(&counter2, block_aggregate);
        //if (block_aggregate != n_hits_i * n_hits_j) {
        //  printf("Error there: %d %d \n", block_aggregate, n_hits_i * n_hits_j);
        //}
      }
    } else {
      printf("Error: n_hits_i > 1024!");
    }
  }
}

__global__ void triplet_finding(unsigned n_mod_triplets,
                                ModuleTriplet *module_triplets, Hit *hits,
                                unsigned *hits_offsets, unsigned *hits_pairs,
                                unsigned *hits_pairs_offsets,
                                unsigned *hits_a, unsigned *hits_b,
                                unsigned *hits_module_offsets,
                                unsigned *hits_cumsum, unsigned *hits_sum_offsets) {

  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_triplets;
       mod_idx += gridDim.x) {
    ModuleTriplet mod_trip = module_triplets[mod_idx];
    unsigned mod_i = mod_trip.module_a;
    unsigned mod_j = mod_trip.module_b;
    unsigned mod_k = mod_trip.module_b;

    unsigned hits_i_offset = hits_offsets[mod_i];
    unsigned hits_j_offset = hits_offsets[mod_j];
    unsigned hits_k_offset = hits_offsets[mod_k];

    unsigned n_hits_i = hits_offsets[mod_i + 1] - hits_i_offset;
    unsigned n_hits_j = hits_offsets[mod_j + 1] - hits_j_offset;
    unsigned n_hits_k = hits_offsets[mod_k + 1] - hits_k_offset;

    unsigned doublet_l_offset = hits_module_offsets[mod_trip.pair_i];
    unsigned n_pairs_l = hits_module_offsets[mod_trip.pair_i + 1] -
                         doublet_l_offset;

    unsigned doublet_m_offset = hits_module_offsets[mod_trip.pair_j];
    
    unsigned hits_sum_offset = hits_sum_offsets[mod_trip.pair_j];
    unsigned *h_sum = hits_cumsum + hits_sum_offset;

    unsigned hits_pairs_ij_offset = hits_pairs_offsets[mod_trip.pair_i];
    unsigned *is_hits_pair_ij = hits_pairs + hits_pairs_ij_offset;
    unsigned hits_pairs_jk_offset = hits_pairs_offsets[mod_trip.pair_j];
    unsigned *is_hits_pair_jk = hits_pairs + hits_pairs_jk_offset;

    for (unsigned l = threadIdx.y; l < n_pairs_l; l += blockDim.y) {
      unsigned hit_i_ind = hits_a[doublet_l_offset + l];
      unsigned hit_j_ind = hits_b[doublet_l_offset + l];

      unsigned n_pairs_m = h_sum[hit_j_ind + 1] - h_sum[hit_j_ind];
      for (unsigned m = threadIdx.x; m < n_pairs_m; m += blockDim.x) {
        unsigned hit_k_ind = hits_b[doublet_m_offset + h_sum[hit_j_ind] + m];
        bool accept = triplet_selection(
          hits[hits_i_offset + hit_i_ind], hits[hits_j_offset + hit_j_ind], hits[hits_k_offset + hit_k_ind], mod_trip);
        atomicAdd(&counter4, accept);
        is_hits_pair_ij[hit_i_ind * n_hits_j + hit_j_ind] |= accept;
        is_hits_pair_jk[hit_j_ind * n_hits_k + hit_k_ind] |= accept;
      }
    }
  }
}

__global__ void triplet_finding_baseline(unsigned n_mod_triplets,
                                         ModuleTriplet *module_triplets,
                                         Hit *hits, unsigned *hits_offsets,
                                         unsigned *hits_pairs_acc,
                                         unsigned *hits_pairs_acc_offsets) {

  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_triplets;
       mod_idx += gridDim.x) {
    ModuleTriplet mod_trip = module_triplets[mod_idx];

    unsigned hits_a_offset = hits_offsets[mod_trip.module_a];
    unsigned hits_b_offset = hits_offsets[mod_trip.module_b];
    unsigned hits_c_offset = hits_offsets[mod_trip.module_c];

    unsigned n_hits_a = hits_offsets[mod_trip.module_a + 1] - hits_a_offset;
    unsigned n_hits_b = hits_offsets[mod_trip.module_b + 1] - hits_b_offset;
    unsigned n_hits_c = hits_offsets[mod_trip.module_c + 1] - hits_c_offset;

    unsigned hits_pairs_i_acc_offset = hits_pairs_acc_offsets[mod_trip.pair_i];
    unsigned *is_hits_i_pair = &hits_pairs_acc[hits_pairs_i_acc_offset];

    unsigned hits_pairs_j_acc_offset = hits_pairs_acc_offsets[mod_trip.pair_j];
    unsigned *is_hits_j_pair = &hits_pairs_acc[hits_pairs_j_acc_offset];

    for (unsigned i = threadIdx.y; i < n_hits_a; i += blockDim.y) {
      unsigned *row_i_hits_pairs = &is_hits_i_pair[i * n_hits_b];
      for (unsigned j = threadIdx.x; j < n_hits_b; j += blockDim.x) {
        unsigned *row_j_hits_pairs = &is_hits_j_pair[j * n_hits_c];
        for (unsigned k = threadIdx.z; k < n_hits_c; k += blockDim.z) {
          bool accept = triplet_selection_baseline(
              hits[hits_a_offset + i], hits[hits_b_offset + j],
              hits[hits_c_offset + k], mod_trip);
          //atomicAdd(&counter4, accept);
          //atomicOr(&row_i_hits_pairs[j], accept);
          //atomicOr(&row_j_hits_pairs[k], accept);
          row_i_hits_pairs[j] |= accept;
          row_j_hits_pairs[k] |= accept;
        }
      }
    }
  }
}

__global__ void count_reduced_triplets(unsigned n_mod_triplets,
                                ModuleTriplet *module_triplets, 
                                ModuleDoublet *module_doublets,
                                Hit *hits,
                                unsigned *hits_offsets,
                                unsigned *hits_a, unsigned *hits_b,
                                unsigned *hits_module_offsets,
                                unsigned *hits_sum, unsigned *hits_sum_offsets) {
  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_triplets;
       mod_idx += gridDim.x) {
    ModuleTriplet mod_trip = module_triplets[mod_idx];
    //unsigned mod_i = mod_trip.module_a;
    //unsigned mod_j = mod_trip.module_b;
    //unsigned mod_k = mod_trip.module_c;

    // There seems to be a missmatch between pairs in the tripletd
    // and the doublets!!!
    ModuleDoublet mod_doub_a = module_doublets[mod_trip.pair_i];
    ModuleDoublet mod_doub_b = module_doublets[mod_trip.pair_j];

    unsigned mod_i = mod_doub_a.module_a;
    unsigned mod_j = mod_doub_a.module_b;
    unsigned mod_k = mod_doub_b.module_b;

    unsigned hits_i_offset = hits_offsets[mod_i];
    unsigned hits_j_offset = hits_offsets[mod_j];
    unsigned hits_k_offset = hits_offsets[mod_k];

    unsigned n_hits_i = hits_offsets[mod_i + 1] - hits_i_offset;
    unsigned n_hits_j = hits_offsets[mod_j + 1] - hits_j_offset;
    unsigned n_hits_k = hits_offsets[mod_k + 1] - hits_k_offset;

    
    unsigned doublet_l_offset = hits_module_offsets[mod_trip.pair_i];
    unsigned n_pairs_l = hits_module_offsets[mod_trip.pair_i + 1] -
                         doublet_l_offset;

    unsigned doublet_m_offset = hits_module_offsets[mod_trip.pair_j];
    
    unsigned hits_sum_offset = hits_sum_offsets[mod_trip.pair_j];
    unsigned *h_sum = hits_sum + hits_sum_offset;

    for (unsigned l = threadIdx.x; l < n_pairs_l; l += blockDim.x) {
      unsigned hit_i_ind = hits_a[doublet_l_offset + l];
      unsigned hit_j_ind = hits_b[doublet_l_offset + l];

      //printf("hit_i_ind: %u, hit_j_ind: %u in mod_ix: %u \n", hit_i_ind, hit_j_ind, mod_idx);

      unsigned n_pairs_m = h_sum[hit_j_ind + 1] - h_sum[hit_j_ind];
      if (threadIdx.y == 0) {
        atomicAdd(&counter3, n_pairs_m);
      }
      //for (unsigned m = threadIdx.y; m < n_pairs_m; m += blockDim.y) {
      //  unsigned hit_j2_ind = hits_a[doublet_m_offset + h_sum[hit_j_ind] + m];
      //  unsigned hit_k_ind = hits_b[doublet_m_offset + h_sum[hit_j_ind] + m];
      //  //if (hit_j2_ind != hit_j_ind) {
      //  //  printf("hit_i_ind: %u, hit_j_ind: %u, hit_j2_ind %u, in mod_ix: %u \n", hit_i_ind, hit_j_ind, hit_j2_ind, mod_idx);
      //  //}
      //  atomicAdd(&counter3, 1);
      //}
    }
  }
}

__device__ bool doublet_selection(Hit &hit_a, Hit &hit_b,
                                  ModuleDoublet &mod_dbl) {
  /*
  float dphi = hit_b.phi - hit_a.phi;
  dphi -= dphi > CUDART_PI_F ? 2 * CUDART_PI_F : 0;
  dphi += dphi < -CUDART_PI_F ? 2 * CUDART_PI_F : 0;

  float dz = hit_b.z - hit_a.z;
  float dr = hit_b.r - hit_a.r;
  float z0 = hit_a.z - hit_a.r * dz / dr;

  // TODO: understand why we do 1-2, while for the other (dphi, dz) we do 2-1
  float deta = hit_a.eta - hit_b.eta;

  float phi_slope = dphi / dr;
  bool accept = (dphi > mod_dbl.dphi_min) && (dphi < mod_dbl.dphi_max) &&
                (z0 > mod_dbl.z0_min) && (z0 < mod_dbl.z0_max) &&
                (deta > mod_dbl.deta_min) && (deta < mod_dbl.deta_max) &&
                (phi_slope > mod_dbl.phi_slope_min) &&
                (phi_slope < mod_dbl.phi_slope_max);

  return accept;
  */
  return true;
}

__device__ bool doublet_selection_baseline(Hit& hit_a, Hit& hit_b, ModuleDoublet& mod_dbl) {
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

__device__ bool triplet_selection(Hit &hit_bottom, Hit &hit_central, Hit &hit_top,
                                  ModuleTriplet &mod_trip) {
  
  float dphi_ab = hit_central.phi - hit_bottom.phi;

  float dz_ab = hit_central.z - hit_bottom.z;
  float dr_ab = hit_central.r - hit_bottom.r;
  float r1_ab = hit_bottom.r;
  float z1_ab = hit_bottom.z;

  float z0_ab = fabs(dr_ab) > 0 ? z1_ab - r1_ab * dz_ab / dr_ab : 3.402823466e+38F;
  //float z0_ab = -999;
  //if (fabs(dr_ab) > 0) {
  //  z0_ab = z1_ab - r1_ab * dz_ab / dr_ab;
  //} else {
  //  z0_ab = dz_ab >= 0 ? 3.402823466e+38F : -3.402823466e+38F;
  //}

  float deta_ab = hit_bottom.eta - hit_central.eta;

  float phi_slope_ab = fabs(dr_ab) > 0 ? dphi_ab / dr_ab : 3.402823466e+38F;
  //float phi_slope_ab = -999;
  //if (fabs(dr_ab) > 0) {
  //  phi_slope_ab = dphi_ab / dr_ab;
  //} else {
  //  if (dphi_ab > 0) {
  //    phi_slope_ab = 3.402823466e+38F;
  //  } else if (dphi_ab < 0) {
  //    phi_slope_ab = -3.402823466e+38F;
  //  } else {
  //    phi_slope_ab = 0;
  //  }
  //}

  bool accept = (dphi_ab > mod_trip.dphi_min_ab) && (dphi_ab < mod_trip.dphi_max_ab) &&
                (z0_ab > mod_trip.z0_min_ab) && (z0_ab < mod_trip.z0_max_ab) &&
                (deta_ab > mod_trip.deta_min_ab) && (deta_ab < mod_trip.deta_max_ab) &&
                (phi_slope_ab > mod_trip.phi_slope_min_ab) &&
                (phi_slope_ab < mod_trip.phi_slope_max_ab);

  float dphi_bc = hit_top.phi - hit_central.phi;

  float dz_bc = hit_top.z - hit_central.z;
  float dr_bc = hit_top.r - hit_central.r;
  float r1_bc = hit_central.r;
  float z1_bc = hit_central.z;

  float z0_bc = fabs(dr_bc) > 0 ? z1_bc - r1_bc * dz_bc / dr_bc : 3.402823466e+38F;
  //float z0_bc = -999;
  //if (fabs(dr_bc) > 0) {
  //  z0_bc = z1_bc - r1_bc * dz_bc / dr_bc;
  //} else {
  //  z0_bc = dz_bc >= 0 ? 3.402823466e+38F : -3.402823466e+38F;
  //}

  float deta_bc = hit_central.eta - hit_top.eta;

  float phi_slope_bc = fabs(dr_bc) > 0 ? dphi_bc / dr_bc : 3.402823466e+38F;
  //float phi_slope_bc = -999;
  //if (fabs(dr_bc) > 0) {
  //  phi_slope_bc = dphi_bc / dr_bc;
  //} else {
  //  if (dphi_bc > 0) {
  //    phi_slope_bc = 3.402823466e+38F;
  //  } else if (dphi_bc < 0) {
  //    phi_slope_bc = -3.402823466e+38F;
  //  } else {
  //    phi_slope_bc = 0;
  //  }
  //}

  accept &= (dphi_bc > mod_trip.dphi_min_bc) && (dphi_bc < mod_trip.dphi_max_bc) &&
            (z0_bc > mod_trip.z0_min_bc) && (z0_bc < mod_trip.z0_max_bc) &&
            (deta_bc > mod_trip.deta_min_bc) && (deta_bc < mod_trip.deta_max_bc) &&
            (phi_slope_bc > mod_trip.phi_slope_min_bc) &&
            (phi_slope_bc < mod_trip.phi_slope_max_bc);

  float dy_ab = hit_central.y - hit_bottom.y;
  float dy_bc = hit_top.y - hit_central.y;
  float dx_ab = hit_central.x - hit_bottom.x;
  float dx_bc = hit_top.x - hit_central.x;

  float diff_dydx = (dx_ab != 0 && dx_bc != 0) ? (dy_ab / dx_ab) - (dy_bc / dx_bc) : 3.402823466e+38F;
  //float diff_dydx = 999;
  //if (dx_ab != 0 && dx_bc != 0) {
  //  diff_dydx = (dy_ab / dx_ab) - (dy_bc / dx_bc);
  //} else if (dx_ab == 0 && dx_bc == 0) {
  //  diff_dydx = 0;
  //} else if (dx_ab == 0) {
  //  diff_dydx = ((-dy_bc / dx_bc) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
  //} else if (dx_bc == 0) {
  //  diff_dydx = ((dy_ab / dx_ab) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
  //}

  float diff_dzdr = (dr_ab != 0 && dr_bc != 0) ? (dz_ab / dr_ab) - (dz_bc / dr_bc) : 3.402823466e+38F;
  //float diff_dzdr = 999;
  //if (dr_ab != 0 && dr_bc != 0) {
  //  diff_dzdr = (dz_ab / dr_ab) - (dz_bc / dr_bc);
  //} else if (dr_ab == 0 && dr_bc == 0) {
  //  diff_dzdr = 0;
  //} else if (dr_ab == 0) {
  //  diff_dzdr = ((-dz_bc / dr_bc) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
  //} else if (dr_bc == 0) {
  //  diff_dzdr = ((dz_ab / dr_ab) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
  //}

  accept &= (diff_dydx > mod_trip.diff_dydx_min) &&
            (diff_dydx < mod_trip.diff_dydx_max) &&
            (diff_dzdr > mod_trip.diff_dzdr_min) &&
            (diff_dzdr < mod_trip.diff_dzdr_max);
  
  return accept;
}

__device__ bool triplet_selection_baseline(Hit &hit_bottom, Hit &hit_central, Hit &hit_top,
                                           ModuleTriplet &mod_trip) {
  float dphi_ab = Dphi(hit_bottom.x, hit_bottom.y, hit_central.x, hit_central.y);

  float z0_ab = Z0(hit_bottom.x, hit_bottom.y, hit_bottom.z, hit_central.x, hit_central.y, hit_central.z);

  float deta_ab = Deta(hit_bottom.x, hit_bottom.y, hit_bottom.z, hit_central.x, hit_central.y, hit_central.z);

  float phi_slope_ab = PhiSlope(hit_bottom.x, hit_bottom.y, hit_central.x, hit_central.y);

  bool accept = (dphi_ab > mod_trip.dphi_min_ab) && (dphi_ab < mod_trip.dphi_max_ab) &&
                (z0_ab > mod_trip.z0_min_ab) && (z0_ab < mod_trip.z0_max_ab) &&
                (deta_ab > mod_trip.deta_min_ab) && (deta_ab < mod_trip.deta_max_ab) &&
                (phi_slope_ab > mod_trip.phi_slope_min_ab) &&
                (phi_slope_ab < mod_trip.phi_slope_max_ab);

  float dphi_bc = Dphi(hit_central.x, hit_central.y, hit_top.x, hit_top.y);

  float z0_bc = Z0(hit_central.x, hit_central.y, hit_central.z, hit_top.x, hit_top.y, hit_top.z);

  float deta_bc = Deta(hit_central.x, hit_central.y, hit_central.z, hit_top.x, hit_top.y, hit_top.z);

  float phi_slope_bc = PhiSlope(hit_central.x, hit_central.y, hit_top.x, hit_top.y);

  accept &= (dphi_bc > mod_trip.dphi_min_bc) && (dphi_bc < mod_trip.dphi_max_bc) &&
            (z0_bc > mod_trip.z0_min_bc) && (z0_bc < mod_trip.z0_max_bc) &&
            (deta_bc > mod_trip.deta_min_bc) && (deta_bc < mod_trip.deta_max_bc) &&
            (phi_slope_bc > mod_trip.phi_slope_min_bc) &&
            (phi_slope_bc < mod_trip.phi_slope_max_bc);
  
  float diff_dydx =
      DiffDyDx(hit_bottom.x, hit_central.x, hit_top.x, hit_bottom.y, hit_central.y, hit_top.y);

  float diff_dzdr =
      DiffDzDr(hit_bottom.x, hit_central.x, hit_top.x, hit_bottom.y, hit_central.y, hit_top.y, hit_bottom.z, hit_central.z, hit_top.z);

  accept &= (diff_dydx > mod_trip.diff_dydx_min) &&
            (diff_dydx < mod_trip.diff_dydx_max) &&
            (diff_dzdr > mod_trip.diff_dzdr_min) &&
            (diff_dzdr < mod_trip.diff_dzdr_max);

  return accept;
}


// Edge features
__device__ float Phi(const float &x, const float &y) {
  return atan2(y, x);
}

__device__ float R(const float &x, const float &y) {
  return sqrtf(x * x + y * y);
}
__device__ float Eta(const float &x, const float &y, const float &z) {
  float r = R(x, y);
  float r3 = sqrtf(r * r + z * z);
  float theta = acos(z / r3);

  return -log(tan(theta * 0.5));
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