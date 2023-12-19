#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "../include/cuda_kernels.h"

// kernel function (compute upperlimmit on edge count)
//----------------------

__global__ void kernel_edge_count(
    unsigned int *flatten_doublets, unsigned int *input_hits_chunk_idx,
    unsigned int *edge_count, // could use int here, but same vector will be
                              // used to store cumsum
    unsigned int n) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threds = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += total_threds) {

    unsigned int m1_idx = flatten_doublets[i * 2];
    unsigned int m2_idx = flatten_doublets[i * 2 + 1];

    unsigned int m1_start_idx = input_hits_chunk_idx[m1_idx],
                 m1_end_idx = input_hits_chunk_idx[m1_idx + 1],

                 m2_start_idx = input_hits_chunk_idx[m2_idx],
                 m2_end_idx = input_hits_chunk_idx[m2_idx + 1];

    edge_count[i] = (m1_end_idx - m1_start_idx) * (m2_end_idx - m2_start_idx);
  }
}

void launch_kernel_edge_count(
    unsigned int *flatten_doublets, unsigned int *input_hits_chunk_idx,
    unsigned int *edge_count, // could use int here, but same vector will be
                              // used to store cumsum
    unsigned int n, int grid_size, int block_size) {

  kernel_edge_count<<<grid_size, block_size>>>(
      flatten_doublets, input_hits_chunk_idx, edge_count, n);
}

// kernel function (compute num loop-iterations on each triplet)
//----------------------

__global__ void kernel_triplet_iter_count(unsigned int *flatten_triplets,
                                          unsigned int *input_hits_chunk_idx,
                                          unsigned int *iter_count,
                                          unsigned int n) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threds = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += total_threds) {

    unsigned int m1_idx = flatten_triplets[i * 3];
    unsigned int m2_idx = flatten_triplets[i * 3 + 1];
    unsigned int m3_idx = flatten_triplets[i * 3 + 2];

    unsigned int m1_start_idx = input_hits_chunk_idx[m1_idx],
                 m1_end_idx = input_hits_chunk_idx[m1_idx + 1],

                 m2_start_idx = input_hits_chunk_idx[m2_idx],
                 m2_end_idx = input_hits_chunk_idx[m2_idx + 1],

                 m3_start_idx = input_hits_chunk_idx[m3_idx],
                 m3_end_idx = input_hits_chunk_idx[m3_idx + 1];

    iter_count[i] = (m1_end_idx - m1_start_idx) * (m2_end_idx - m2_start_idx) *
                    (m3_end_idx - m3_start_idx);
  }
}

void launch_kernel_triplet_iter_count(unsigned int *flatten_triplets,
                                      unsigned int *input_hits_chunk_idx,
                                      unsigned int *iter_count, unsigned int n,
                                      int grid_size, int block_size) {

  kernel_triplet_iter_count<<<grid_size, block_size>>>(
      flatten_triplets, input_hits_chunk_idx, iter_count, n);
}

// kernel function (apply doublet cuts)
//----------------------

__global__ void kernel_apply_doublet_cuts(unsigned int *flatten_doublets,
                                          unsigned int *input_hits_chunk_idx,
                                          unsigned int *edge_count,
                                          unsigned int n) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threds = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += total_threds) {

    unsigned int m1_idx = flatten_doublets[i * 2];
    unsigned int m2_idx = flatten_doublets[i * 2 + 1];

    unsigned int m1_start_idx = input_hits_chunk_idx[m1_idx],
                 m1_end_idx = input_hits_chunk_idx[m1_idx + 1],

                 m2_start_idx = input_hits_chunk_idx[m2_idx],
                 m2_end_idx = input_hits_chunk_idx[m2_idx + 1];

    edge_count[i] = (m1_end_idx - m1_start_idx) * (m2_end_idx - m2_start_idx);
  }
}

// kernel function (make edges)
//----------------------

// Each thread takes one triplet
// Cons:
//     - Less equal workload distribution (some triplets are cheaper)
//     - Not taking advantage of the doublet cuts (run the doublet cuts again
//     and again)

__global__ void kernel_make_edges(unsigned int *flatten_triplets,
                                  unsigned int *flatten_t2d_links,
                                  unsigned int *edge_count, float *input_hits,
                                  unsigned int *input_hits_chunk_idx,
                                  unsigned int *srcs, unsigned int *dsts,
                                  unsigned int n) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threds = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += total_threds) {

    unsigned int m1_idx = flatten_triplets[i * 3];
    unsigned int m2_idx = flatten_triplets[i * 3 + 1];
    unsigned int m3_idx = flatten_triplets[i * 3 + 2];

    unsigned int m1_start_idx = input_hits_chunk_idx[m1_idx],
                 m1_end_idx = input_hits_chunk_idx[m1_idx + 1],

                 m2_start_idx = input_hits_chunk_idx[m2_idx],
                 m2_end_idx = input_hits_chunk_idx[m2_idx + 1],

                 m3_start_idx = input_hits_chunk_idx[m3_idx],
                 m3_end_idx = input_hits_chunk_idx[m3_idx + 1];

    unsigned int n_hits_m1 = m1_end_idx - m1_start_idx;
    unsigned int n_hits_m2 = m2_end_idx - m2_start_idx;

    // // -1 coz there is a a zero in the beginning
    // unsigned int offset_d1 = edge_count[flatten_t2d_links[i*2] - 1];
    // unsigned int offset_d2 = edge_count[flatten_t2d_links[i*2+1] - 1];

    // no -1
    unsigned int offset_d1 = edge_count[flatten_t2d_links[i * 2]];
    unsigned int offset_d2 = edge_count[flatten_t2d_links[i * 2 + 1]];

    // loop over m1 hits
    for (unsigned int j = m1_start_idx; j < m1_end_idx; j++) {

      float x1 = input_hits[j * 3];
      float y1 = input_hits[j * 3 + 1];
      float z1 = input_hits[j * 3 + 2];

      // loop over m2 hits
      for (unsigned int k = m2_start_idx; k < m2_end_idx; k++) {

        float x2 = input_hits[k * 3];
        float y2 = input_hits[k * 3 + 1];
        float z2 = input_hits[k * 3 + 2];

        // loop over m3 hits
        for (unsigned int l = m3_start_idx; l < m3_end_idx; l++) {

          float x3 = input_hits[l * 3];
          float y3 = input_hits[l * 3 + 1];
          float z3 = input_hits[l * 3 + 2];

          // doublet_cuts
          bool pass_12 = (x2 * y2 * z2) > (x1 * y1 * z1);
          if (!pass_12) {
            continue;
          }

          bool pass_23 = (x3 * y3 * z3) > (x2 * y2 * z2);
          if (!pass_23) {
            continue;
          }

          // triplet cuts
          bool pass_123 = (x1 * y1 * z1 * x2 * y2 * z2 * x3 * y3 * z3) > 0;
          if (!pass_123) {
            continue;
          }

          // // DOUBLE CHECK THESE INDICES!!!

          srcs[0] = 1;
          dsts[0] = 1;

          // // edges in pair_12
          // unsigned int edge_ind_12 = (n_hits_m1 * j + k) + offset_d1;
          // srcs[edge_ind_12] =  1;
          // dsts[edge_ind_12] =  1;

          // // // edges in pair_23
          // unsigned int edge_ind_23 = (n_hits_m2 * k + l) + offset_d2;
          // srcs[edge_ind_23] =  1;
          // dsts[edge_ind_23] =  1;

          // input_hits[j*3]   = input_hits[j*3] + 1;
          // input_hits[j*3+1] = input_hits[j*3+1] + 1;
          // input_hits[j*3+2] = input_hits[j*3+2] + 1;

          // input_hits[k*3]   = input_hits[k*3] + 1;
          // input_hits[k*3+1] = input_hits[k*3+1] + 1;
          // input_hits[k*3+2] = input_hits[k*3+2] + 1;

          // input_hits[l*3]   = input_hits[l*3] + 1;
          // input_hits[l*3+1] = input_hits[l*3+1] + 1;
          // input_hits[l*3+2] = input_hits[l*3+2] + 1;
        }
      }
    }
  }
}

void launch_kernel_make_edges(unsigned int *flatten_triplets,
                              unsigned int *flatten_t2d_links,
                              unsigned int *edge_count, float *input_hits,
                              unsigned int *input_hits_chunk_idx,
                              unsigned int *srcs, unsigned int *dsts,
                              unsigned int n, int grid_size, int block_size) {

  kernel_make_edges<<<grid_size, block_size>>>(
      flatten_triplets, flatten_t2d_links, edge_count, input_hits,
      input_hits_chunk_idx, srcs, dsts, n);
}

// kernel function (make edges with doublets) - with Stream
//----------------------

// Each thread takes one Hit doublet
// Pros:
//     - Much better workload distribution (some triplets are cheaper)
// Cons:
//     - Not taking advantage of the doublet cuts (run the doublet cuts again
//     and again)

__global__ void kernel_make_edges(unsigned int *flatten_triplets,
                                  unsigned int *flatten_t2d_links,
                                  unsigned int *edge_count, float *input_hits,
                                  unsigned int *input_hits_chunk_idx,
                                  unsigned int *srcs, unsigned int *dsts,
                                  unsigned int m_triplet_idx,
                                  unsigned long int offset) {

  unsigned long int tid =
      offset + (unsigned long int)blockDim.x * (unsigned long int)blockIdx.x +
      (unsigned long int)threadIdx.x;

  // get the module indices
  unsigned int m1_idx = flatten_triplets[m_triplet_idx * 3];
  unsigned int m2_idx = flatten_triplets[m_triplet_idx * 3 + 1];
  unsigned int m3_idx = flatten_triplets[m_triplet_idx * 3 + 2];

  // get hit count in each module
  unsigned int n_hits_m1 =
      input_hits_chunk_idx[m1_idx + 1] - input_hits_chunk_idx[m1_idx];
  unsigned int n_hits_m2 =
      input_hits_chunk_idx[m2_idx + 2] - input_hits_chunk_idx[m2_idx];
  unsigned int n_hits_m3 =
      input_hits_chunk_idx[m3_idx + 3] - input_hits_chunk_idx[m3_idx];

  // exit if needed
  if (tid < n_hits_m1 * n_hits_m2 * n_hits_m3) {

    // figure out the hit indices in the corresponding modules
    unsigned int tmp = tid % (n_hits_m2 * n_hits_m3);

    unsigned int i = tid / (n_hits_m2 * n_hits_m3);
    unsigned int j = tmp / n_hits_m3;
    unsigned int k = tmp % n_hits_m3;

    float x1 = input_hits[i], y1 = input_hits[i + 1], z1 = input_hits[i + 2];
    float x2 = input_hits[j], y2 = input_hits[j + 1], z2 = input_hits[j + 2];
    float x3 = input_hits[k], y3 = input_hits[k + 1], z3 = input_hits[k + 2];

    // doublet_cuts
    bool pass_12 = (x2 * y2 * z2) > (x1 * y1 * z1);
    if (pass_12) {

      bool pass_23 = (x3 * y3 * z3) > (x2 * y2 * z2);
      if (!pass_23) {

        // triplet cuts
        bool pass_123 = (x1 * y1 * z1 * x2 * y2 * z2 * x3 * y3 * z3) > 0;
        if (!pass_123) {

          // DOUBLE CHECK THESE INDICES!!!
          // // -1 coz there is a a zero in the beginning
          // unsigned int offset_d1 = edge_count[flatten_t2d_links[i*2] - 1];
          // unsigned int offset_d2 = edge_count[flatten_t2d_links[i*2+1] - 1];

          // // no -1
          // unsigned int offset_d1 = edge_count[flatten_t2d_links[i*2]];
          // unsigned int offset_d2 = edge_count[flatten_t2d_links[i*2+1]];

          srcs[0] = 1;
          dsts[0] = 1;
        }
      }
    }
  }
}

void launch_kernel_make_edges(
    unsigned int *flatten_triplets, unsigned int *flatten_t2d_links,
    unsigned int *edge_count, float *input_hits,
    unsigned int *input_hits_chunk_idx, unsigned int *srcs, unsigned int *dsts,
    unsigned int m_triplet_idx, unsigned long int offset, int grid_size,
    int block_size, unsigned int shared_mem_size, cudaStream_t stream) {

  kernel_make_edges<<<grid_size, block_size, shared_mem_size, stream>>>(
      flatten_triplets, flatten_t2d_links, edge_count, input_hits,
      input_hits_chunk_idx, srcs, dsts, m_triplet_idx, offset);
}
