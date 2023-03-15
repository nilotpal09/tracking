#include "../include/triplet_finder.cuh"
#include "../include/utility.cuh"

__device__ bool doublet_selection(Hit &hit_a, Hit &hit_b,
                                  ModuleDoublet &mod_dbl) {
  // bool accept = true;

  float dphi = hit_b.phi - hit_a.phi;
  dphi += dphi > CUDART_PI_F ? -2 * CUDART_PI_F : 2 * CUDART_PI_F;
  // float dphi = Dphi(hit_a.phi, hit_a.phi);
  // accept &= (dphi > mod_dbl.dphi_min) && (dphi < mod_dbl.dphi_max);

  float dz = hit_b.z - hit_a.z;
  float dr = hit_b.r - hit_a.r;
  float z0 = hit_a.z - hit_a.r * dz / dr;
  // float z0 = Z0(hit_a.r, hit_b.r, hit_a.z, hit_b.z);
  // accept &= (z0 > mod_dbl.z0_min) && (z0 < mod_dbl.z0_max);

  // TODO: understand why we do 1-2, while for the other (dphi, dz) we do 2-1
  float deta = hit_a.eta - hit_b.eta;
  // float deta = Deta(hit_a.eta, hit_b.eta);
  // accept &= (deta > mod_dbl.deta_min) && (deta < mod_dbl.deta_max);

  float phi_slope = dphi / dr;
  // float phi_slope = PhiSlope(hit_a.phi, hit_a.r, hit_b.phi, hit_b.r);
  bool accept = (dphi > mod_dbl.dphi_min) && (dphi < mod_dbl.dphi_max) &&
                (z0 > mod_dbl.z0_min) && (z0 < mod_dbl.z0_max) &&
                (deta > mod_dbl.deta_min) && (deta < mod_dbl.deta_max) &&
                (phi_slope > mod_dbl.phi_slope_min) &&
                (phi_slope < mod_dbl.phi_slope_max);

  return accept;
}

__device__ bool triplet_selection(Hit &hit_a, Hit &hit_b, Hit &hit_c,
                                  ModuleTriplet &mod_trip) {

  float dphi_ab = hit_b.phi - hit_a.phi;
  dphi_ab += dphi_ab > CUDART_PI_F ? -2 * CUDART_PI_F : 2 * CUDART_PI_F;
  float dz_ab = hit_b.z - hit_a.z;
  float dr_ab = hit_b.r - hit_a.r;
  float z0_ab = hit_a.z - hit_a.r * dz_ab / dr_ab;
  float deta_ab = hit_a.eta - hit_b.eta;
  float phi_slope_ab = dphi_ab / dr_ab;

  float dphi_bc = hit_c.phi - hit_b.phi;
  dphi_bc += dphi_bc > CUDART_PI_F ? -2 * CUDART_PI_F : 2 * CUDART_PI_F;
  float dz_bc = hit_c.z - hit_b.z;
  float dr_bc = hit_c.r - hit_b.r;
  float z0_bc = hit_b.z - hit_b.r * dz_bc / dr_bc;
  float deta_bc = hit_b.eta - hit_c.eta;
  float phi_slope_bc = dphi_bc / dr_bc;

  float diff_dydx =
      DiffDyDx(hit_a.x, hit_b.x, hit_c.x, hit_a.y, hit_b.y, hit_c.y);

  float diff_dzdr =
      DiffDzDr(hit_a.r, hit_b.r, hit_c.r, hit_a.z, hit_b.z, hit_c.z);

  bool accept =
      (dphi_ab > mod_trip.dphi_min_ab) && (dphi_ab < mod_trip.dphi_max_ab) &&
      (z0_ab > mod_trip.z0_min_ab) && (z0_ab < mod_trip.z0_max_ab) &&
      (deta_ab > mod_trip.deta_min_ab) && (deta_ab < mod_trip.deta_max_ab) &&
      (phi_slope_ab > mod_trip.phi_slope_min_ab) &&
      (phi_slope_ab < mod_trip.phi_slope_max_ab) &&
      (dphi_bc > mod_trip.dphi_min_bc) && (dphi_bc < mod_trip.dphi_max_bc) &&
      (z0_bc > mod_trip.z0_min_bc) && (z0_bc < mod_trip.z0_max_bc) &&
      (deta_bc > mod_trip.deta_min_bc) && (deta_bc < mod_trip.deta_max_bc) &&
      (phi_slope_bc > mod_trip.phi_slope_min_bc) &&
      (phi_slope_bc < mod_trip.phi_slope_max_bc) &&
      (diff_dydx > mod_trip.diff_dydx_min) &&
      (diff_dydx < mod_trip.diff_dydx_max) &&
      (diff_dzdr > mod_trip.diff_dzdr_min) &&
      (diff_dzdr < mod_trip.diff_dzdr_max);

  return accept;
}

__device__ bool triplet_selection_baseline(Hit &hit_a, Hit &hit_b, Hit &hit_c,
                                           ModuleTriplet &mod_trip) {
  /*
  float phi_a = Phi(hit_a.x, hit_a.y);
  float eta_a = Eta(hit_a.x, hit_a.y, hit_a.z);

  bool accept = true;

  float dphi_ab = Dphi(hit_a.x, hit_a.y, hit_b.x, hit_b.y);
  accept &= (dphi_ab > mod_trip.dphi_min_ab) && (dphi_ab <
  mod_trip.dphi_max_ab);

  float z0_ab = Z0(hit_a.x, hit_a.y, hit_a.z, hit_b.x, hit_b.y, hit_b.z);
  accept &= (z0_ab > mod_trip.z0_min_ab) && (z0_ab < mod_trip.z0_max_ab);

  float deta_ab = Deta(hit_a.x, hit_a.y, hit_a.z, hit_b.x, hit_b.y, hit_b.z);
  accept &= (deta_ab > mod_trip.deta_min_ab) && (deta_ab <
  mod_trip.deta_max_ab);

  float phi_slope_ab = PhiSlope(hit_a.x, hit_a.y, hit_b.x, hit_b.y);
  accept &= (phi_slope_ab > mod_trip.phi_slope_min_ab) &&
            (phi_slope_ab < mod_trip.phi_slope_max_ab);

  float dphi_bc = Dphi(hit_b.x, hit_b.y, hit_c.x, hit_c.y);
  accept &= (dphi_bc > mod_trip.dphi_min_bc) && (dphi_bc <
  mod_trip.dphi_max_bc);

  float z0_bc = Z0(hit_b.x, hit_b.y, hit_b.z, hit_c.x, hit_c.y, hit_c.z);
  accept &= (z0_bc > mod_trip.z0_min_bc) && (z0_bc < mod_trip.z0_max_bc);

  float deta_bc = Deta(hit_b.x, hit_b.y, hit_b.z, hit_c.x, hit_c.y, hit_c.z);
  accept &= (deta_bc > mod_trip.deta_min_bc) && (deta_bc <
  mod_trip.deta_max_bc);

  float phi_slope_bc = PhiSlope(hit_b.x, hit_b.y, hit_c.x, hit_c.y);
  accept &= (phi_slope_bc > mod_trip.phi_slope_min_bc) &&
            (phi_slope_bc < mod_trip.phi_slope_max_bc);

  float diff_dydx = DiffDyDx(hit_a.x, hit_b.x, hit_c.x, hit_a.y, hit_b.y,
                             hit_c.y);
  accept &= (diff_dydx > mod_trip.diff_dydx_min) &&
            (diff_dydx < mod_trip.diff_dydx_max);

  float diff_dzdr = DiffDzDr(hit_a.x, hit_b.x, hit_c.x, hit_a.y, hit_b.y,
                             hit_c.y, hit_a.z, hit_b.z, hit_c.z);
  accept &= (diff_dzdr > mod_trip.diff_dzdr_min) &&
            (diff_dzdr < mod_trip.diff_dzdr_max);

  return accept;
  */
  return true;
}

__global__ void
doublet_finding(unsigned n_mod_pairs, ModuleDoublet *module_pairs, Hit *hits,
                unsigned *hits_offsets, unsigned *hits_pairs_acc,
                unsigned *hits_pairs_acc_offsets, unsigned *hits_sum,
                unsigned *hits_sum_offsets, unsigned *hits_module_sum) {
  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_pairs;
       mod_idx += gridDim.x) {
    ModuleDoublet mod_dbl = module_pairs[mod_idx];
    unsigned mod_i = mod_dbl.module_a;
    unsigned mod_j = mod_dbl.module_b;

    unsigned hits_i_offset = hits_offsets[mod_i];
    unsigned hits_j_offset = hits_offsets[mod_j];

    unsigned n_hits_i = hits_offsets[mod_i + 1] - hits_i_offset;
    unsigned n_hits_j = hits_offsets[mod_j + 1] - hits_j_offset;

    unsigned hits_pairs_acc_offset = hits_pairs_acc_offsets[mod_idx];
    unsigned *is_hits_pair = &hits_pairs_acc[hits_pairs_acc_offset];

    unsigned hits_sum_offset = hits_sum_offsets[mod_idx];
    unsigned *h_sum =
        &hits_sum[hits_sum_offset + 1]; // 0th entry needs to be 0;

    typedef cub::WarpReduce<unsigned> WarpReduce;
    __shared__ typename WarpReduce::TempStorage
        temp_storage[WARP_SIZE]; // We have 32 warps (1024 threads)

    __shared__ unsigned temp_hit_sum[32][32];
    unsigned n_loops = n_hits_i / WARP_SIZE;
    if (n_hits_i % WARP_SIZE != 0) {
      n_loops += 1;
    }
    for (unsigned l = 0; l < n_loops; l++) {
      temp_hit_sum[threadIdx.y][threadIdx.x] = 0;
      unsigned i = (threadIdx.y + (l * WARP_SIZE));
      if (i < n_hits_i) {
        unsigned *row_hits_pairs = &is_hits_pair[i * n_hits_j];
        for (unsigned j = threadIdx.x; j < n_hits_j; j += blockDim.x) {
          row_hits_pairs[j] = doublet_selection(
              hits[hits_i_offset + i], hits[hits_j_offset + j], mod_dbl);
          temp_hit_sum[threadIdx.y][threadIdx.x] += row_hits_pairs[j];
        }
        __syncthreads();
        unsigned thread_data =
            temp_hit_sum[threadIdx.y][threadIdx.x]; // row_hits_pairs[j];
        unsigned warp_id = (threadIdx.y * blockDim.x + threadIdx.y) / WARP_SIZE;
        unsigned aggregate = WarpReduce(temp_storage[warp_id]).Sum(thread_data);

        if (threadIdx.x == 0) {
          h_sum[i] = aggregate;
        }
      }
    }
    __syncthreads();
    // Max number of hits was found to be ~300 so this is safe for now.
    // But if the number of hits is larger than 1024 this will become an issue.
    typedef cub::BlockScan<unsigned, 32, cub::BLOCK_SCAN_RAKING, 32> BlockScan;
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned thread_data = tid < n_hits_i ? h_sum[tid] : 0;

    // Run exclusive prefix-scan (aka scan) across the thread block
    BlockScan(block_temp_storage).InclusiveSum(thread_data, thread_data);

    if (tid < n_hits_i) {
      h_sum[tid] = thread_data;
      if (tid == n_hits_i - 1) {
        hits_module_sum[mod_idx] = h_sum[tid];
      }
    }
    __syncthreads();
  }
}

__global__ void triplet_finding(unsigned n_mod_triplets,
                                ModuleTriplet *module_triplets, Hit *hits,
                                unsigned *hits_offsets,
                                unsigned *hits_pairs_acc, unsigned *hits_sum,
                                unsigned *hits_sum_offsets, unsigned *hits_a,
                                unsigned *hits_b,
                                unsigned *hit_module_offsets) {

  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_triplets;
       mod_idx += gridDim.x) {
    ModuleTriplet mod_trip = module_triplets[mod_idx];

    unsigned hits_a_offset = hits_offsets[mod_trip.module_a];
    unsigned hits_b_offset = hits_offsets[mod_trip.module_b];
    unsigned hits_c_offset = hits_offsets[mod_trip.module_c];

    unsigned hit_pair_i_offset = hit_module_offsets[mod_trip.pair_i];
    unsigned n_pairs_i =
        hit_module_offsets[mod_trip.pair_i + 1] - hit_pair_i_offset;

    unsigned hit_pair_j_offset = hit_module_offsets[mod_trip.pair_j];

    unsigned hits_sum_offset = hits_sum_offsets[mod_trip.pair_j];
    unsigned *h_sum = &hits_sum[hits_sum_offset];

    unsigned *is_hits_pair_i = &hits_pairs_acc[hit_pair_i_offset];
    unsigned *is_hits_pair_j = &hits_pairs_acc[hit_pair_j_offset];

    for (unsigned i = threadIdx.y; i < n_pairs_i; i += blockDim.y) {
      unsigned hit_a_idx = hits_a[hit_pair_i_offset + i];
      unsigned hit_b_idx = hits_b[hit_pair_i_offset + i];

      unsigned hit_j_offset = h_sum[hit_b_idx];

      unsigned n_pairs_j = h_sum[hit_b_idx + 1] - hit_pair_j_offset;

      if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf(
            "mod_idx: %d, 'hit_a_idx': %d, 'hit_b_idx': %d, 'n_pairs_j': %d\n",
            mod_idx, hit_a_idx, hit_b_idx, n_pairs_j);
      }

      for (unsigned j = blockIdx.x; j < n_pairs_j; j += blockDim.x) {
        unsigned hit_c_idx = hits_b[hit_pair_j_offset + hit_j_offset + j];
        bool accept = triplet_selection(
            hits[hits_a_offset + hit_a_idx], hits[hits_b_offset + hit_b_idx],
            hits[hits_c_offset + hit_c_idx], mod_trip);
        // is_hits_pair_i[i] = accept;
        //  is_hits_pair_j[i] = accept;

        // Create graph triplet here?
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
          row_i_hits_pairs[j] = accept;
          row_j_hits_pairs[k] = accept;
        }
      }
    }
  }
}

// Edge features
__device__ float Dphi(const float &phi1, const float &phi2) {
  float dphi = phi2 - phi1;

  dphi += dphi > CUDART_PI_F ? -2 * CUDART_PI_F : 2 * CUDART_PI_F;
  return dphi;
}

__device__ float Dr(const float &r1, const float &r2) { return r2 - r1; }
__device__ float PhiSlope(const float &phi1, const float &r1, const float &phi2,
                          const float &r2) {
  float dphi = Dphi(phi1, phi2);
  float dr = Dr(r1, r2);

  // TODO: Fix this for edge cases
  return dphi / dr;
}
__device__ float Dz(const float &z1, const float &z2) { return z2 - z1; }

__device__ float Z0(const float &r1, const float &r2, const float &z1,
                    const float &z2) {
  float dz = Dz(z1, z2);
  float dr = Dr(r1, r2);

  // TODO: Fix this for edge cases
  return z1 - r1 * dz / dr;
}

__device__ float Deta(const float &eta1, const float &eta2) {
  // TODO: understand why we do 1-2, while for the other (dphi, dz) we do 2-1
  return eta1 - eta2;
}

__device__ float DiffDzDr(const float &r1, const float &r2, const float &r3,
                          const float &z1, const float &z2, const float &z3) {
  float dz_12 = Dz(z1, z2);
  float dz_23 = Dz(z2, z3);
  float dr_12 = Dr(r1, r2);
  float dr_23 = Dr(r2, r3);

  // TODO: Fix this for edge cases
  return (dz_12 / dr_12) - (dz_23 / dr_23);
}
__device__ float DiffDzDr(const float &dz_12, const float &dz_23,
                          const float &dr_12, const float &dr_23) {
  // TODO: Fix this for edge cases
  return (dz_12 / dr_12) - (dz_23 / dr_23);
}

__device__ float DiffDyDx(const float &x1, const float &x2, const float &x3,
                          const float &y1, const float &y2, const float &y3) {
  float dy_12 = y2 - y1;
  float dy_23 = y2 - y3;
  float dx_12 = x1 - x2;
  float dx_23 = x2 - x3;

  // TODO: Fix this for edge cases
  return (dy_12 / dx_12) - (dy_23 / dx_23);
}

__device__ float DiffDyDx(const float &dy_12, const float &dy_23,
                          const float &dx_12, const float &dx_23) {
  // TODO: Fix this for edge cases
  return (dy_12 / dx_12) - (dy_23 / dx_23);
}