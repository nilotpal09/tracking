#ifndef _CUDA_KERNELS_
#define _CUDA_KERNELS_


#include <vector>


// void kernel_edge_count(
//     unsigned long int *flatten_doublets, 
//     unsigned long int *input_hits_chunk_idx,
//     unsigned long int *edge_count, // could use int here, but same vector will be used to store cumsum
//     unsigned long int n);

void launch_kernel_edge_count(
    unsigned long int *flatten_doublets, 
    unsigned long int *input_hits_chunk_idx,
    unsigned long int *edge_count, // could use int here, but same vector will be used to store cumsum
    unsigned long int n,
    int grid_size,
    int block_size);



// void kernel_make_edges(
//     unsigned long int *flatten_triplets,
//     unsigned long int *flatten_t2d_links,
//     unsigned long int *edge_count,
//     float *input_hits, 
//     unsigned long int *input_hits_chunk_idx,
//     unsigned long int *srcs,
//     unsigned long int *dsts,
//     unsigned long int n);

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
    int block_size);

#endif