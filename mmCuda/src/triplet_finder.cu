#include "../include/triplet_finder.cuh"

__device__ bool doublet_selection(Hit hit_a, Hit hit_b) { return true; }

__device__ bool triplet_selection(Hit hit_a, Hit hit_b, Hit hit_c) {

  return true;
}

__global__ void
doublet_finding(unsigned n_mod_pairs, ModuleDoublet *module_pairs, Hit *hits,
                unsigned *hits_offsets, unsigned *hits_pairs_acc,
                unsigned *hits_pairs_acc_offsets, unsigned *hits_sum,
                unsigned *hits_sum_offsets, unsigned *hits_module_sum) {
  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_pairs;
       mod_idx += gridDim.x) {
    unsigned mod_i = module_pairs[mod_idx].module_a;
    unsigned mod_j = module_pairs[mod_idx].module_b;

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
          row_hits_pairs[j] = doublet_selection(hits[hits_i_offset + i],
                                                hits[hits_j_offset + j]);
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

      for (unsigned j = blockIdx.x; j < n_pairs_j; j += blockDim.x) {
        unsigned hit_c_idx = hits_b[hit_pair_j_offset + hit_j_offset + j];
        bool accept = triplet_selection(hits[hits_a_offset + hit_a_idx],
                                        hits[hits_b_offset + hit_b_idx],
                                        hits[hits_c_offset + hit_c_idx]);

        is_hits_pair_i[i] = accept;
        is_hits_pair_j[i] = accept;

        // Create graph triplet here?
      }
    }
  }
}
