#include "../include/device_kernals.cuh"

#define WARP_SIZE 32

__device__
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
