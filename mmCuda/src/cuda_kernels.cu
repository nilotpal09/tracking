#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../include/cuda_kernels.h"




// kernel function (compute upperlimmit on edge count)
//----------------------

__global__ void kernel_edge_count(
    unsigned long int *flatten_doublets, 
    unsigned long int *input_hits_chunk_idx,
    unsigned long int *edge_count, // could use int here, but same vector will be used to store cumsum
    unsigned long int n){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threds = blockDim.x * gridDim.x;

    for (int i=tid; i<n; i+=total_threds){

        unsigned long int m1_idx = flatten_doublets[i*2];
        unsigned long int m2_idx = flatten_doublets[i*2 + 1];

        unsigned long int
        m1_start_idx = input_hits_chunk_idx[m1_idx], 
        m1_end_idx   = input_hits_chunk_idx[m1_idx + 1],

        m2_start_idx = input_hits_chunk_idx[m2_idx], 
        m2_end_idx   = input_hits_chunk_idx[m2_idx + 1];

        edge_count[i] = (m1_end_idx - m1_start_idx) * (m2_end_idx - m2_start_idx);
    }
}

void launch_kernel_edge_count(
    unsigned long int *flatten_doublets, 
    unsigned long int *input_hits_chunk_idx,
    unsigned long int *edge_count, // could use int here, but same vector will be used to store cumsum
    unsigned long int n,
    int grid_size,
    int block_size){

    kernel_edge_count<<<grid_size, block_size>>>(flatten_doublets, input_hits_chunk_idx, edge_count, n);
}






// kernel function (make edges)
//----------------------

__global__ void kernel_make_edges(
    unsigned long int *flatten_triplets,
    unsigned long int *flatten_t2d_links,
    unsigned long int *edge_count,
    float *input_hits, 
    unsigned long int *input_hits_chunk_idx,
    unsigned long int *srcs,
    unsigned long int *dsts,
    unsigned long int n) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threds = blockDim.x * gridDim.x;

    for (int i=tid; i<n; i+=total_threds){

        unsigned long int m1_idx = flatten_triplets[i*3];
        unsigned long int m2_idx = flatten_triplets[i*3 + 1];
        unsigned long int m3_idx = flatten_triplets[i*3 + 2];

        unsigned long int
        m1_start_idx = input_hits_chunk_idx[m1_idx], 
        m1_end_idx   = input_hits_chunk_idx[m1_idx + 1],

        m2_start_idx = input_hits_chunk_idx[m2_idx], 
        m2_end_idx   = input_hits_chunk_idx[m2_idx + 1],

        m3_start_idx = input_hits_chunk_idx[m3_idx], 
        m3_end_idx   = input_hits_chunk_idx[m3_idx + 1];

        int n_hits_m1 = m1_end_idx - m1_start_idx;
        int n_hits_m2 = m2_end_idx - m2_start_idx;
        // int n_hits_m3 = m3_end_idx - m3_start_idx;

        // -1 coz there is a a zero in the beginning
        unsigned long int offset_d1 = edge_count[flatten_t2d_links[i*2] - 1];
        unsigned long int offset_d2 = edge_count[flatten_t2d_links[i*2+1] - 1];

        // loop over m2 hits
        for (unsigned long int j=m2_start_idx; j<m2_end_idx; j++){

            float x2 = input_hits[j*3];
            float y2 = input_hits[j*3+1];
            float z2 = input_hits[j*3+2];

            // no dynamic allocation allowed (!! need a better solution probably !!)
            unsigned long int pair_12[200], pair_23[200]; // holds the node (hit) indices
            int pair_12_count = 0, pair_23_count = 0;

            // loop over m1 hits
            for (unsigned long int k=m1_start_idx; k<m1_end_idx; k++){

                float x1 = input_hits[k*3];
                float y1 = input_hits[k*3+1];
                float z1 = input_hits[k*3+2];

                bool pass = (x1*x1 + y1*y1 + z1*z1) > (x2*x2 + y2*y2 + z2*z2);

                if (pass){
                    pair_12[pair_12_count*2]     = k;
                    pair_12[pair_12_count*2 + 1] = j;
                    pair_12_count++;
                }
            }

            // loop over m3 hits
            for (unsigned long int k=m3_start_idx; k<m3_end_idx; k++){

                float x3 = input_hits[k*3];
                float y3 = input_hits[k*3+1];
                float z3 = input_hits[k*3+2];

                bool pass = (x3*x3 + y3*y3 + z3*z3) > (x2*x2 + y2*y2 + z2*z2);

                if (pass){
                    pair_23[pair_23_count*2]     = j;
                    pair_23[pair_23_count*2 + 1] = k;
                    pair_23_count++;
                }
            }

            // loop over the pairs
            for (unsigned long int k=0; k<pair_12_count; k++){
                for (unsigned long int l=0; l<pair_23_count; l++){
                    
                    // some more selections here
                    // need to store the hits in these 3 modules temporarily
                    // or some more indexing...

                    // edges in pair_12 (flipped, annoying)
                    unsigned long int edge_ind_12 = (n_hits_m1 * pair_12[k*2] + pair_12[k*2+1]) + offset_d1;
                    srcs[edge_ind_12] =  pair_12[k*2];
                    dsts[edge_ind_12] =  pair_12[k*2+1];

                    // edges in pair_23
                    unsigned long int edge_ind_23 = (n_hits_m2 * pair_23[l*2] + pair_23[l*2+1]) + offset_d2;
                    srcs[edge_ind_23] =  pair_23[l*2];
                    dsts[edge_ind_23] =  pair_23[l*2+1];
                }               
            }
        }
    }
}


void launch_kernel_make_edges(
    unsigned long int *flatten_triplets,
    unsigned long int *flatten_t2d_links,
    unsigned long int *edge_count,
    float *input_hits, 
    unsigned long int *input_hits_chunk_idx,
    unsigned long int *srcs,
    unsigned long int *dsts,
    unsigned long int n,
    int grid_size,
    int block_size){

    kernel_make_edges<<<grid_size, block_size>>>(
        flatten_triplets, flatten_t2d_links, edge_count, input_hits, input_hits_chunk_idx, srcs, dsts, n);

}